"""
backend/main.py

Phase 4 of Drishti: the backend orchestration layer. Wires Phases 1-3
(detection, depth, fusion) together behind a real streaming WebSocket
server, so a browser/phone can send camera frames and get back spoken
guidance in near-real-time. No new AI logic lives here -- this file is
pure plumbing.

WEBSOCKET CONTRACT (Phase 5's frontend must match this):
  - Connect to ws://<host>:<port>/ws/stream
  - Send each frame as a TEXT message: a plain base64-encoded JPEG string
    (no "data:image/jpeg;base64," prefix, no JSON wrapper -- just the
    base64 characters themselves).
  - Receive back a JSON text message:
        {"speech": <string or null>, "detections": [...], "latency_ms": N}
  - GET /health -> {"status": "ok", "device": "cpu"}

INTEGRATION NOTE (check this first if you get a KeyError):
this is the FIRST time detection/detector.py's real output actually flows
into fusion/engine.py -- Phase 3's demo only ever used hardcoded fake
observations. fusion/engine.py's build_observations() expects each
detection dict to have keys "class_name", "confidence", "bbox". If your
detector.py's detect() uses different key names, you'll get a KeyError
the first time a real detection comes through -- either rename the keys in
detector.py's return value, or tell me and I'll adjust build_observations()
instead.
"""

import asyncio
import base64
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from detection.detector import detect
from depth.depth_estimator import estimate_depth, load_calibration
from fusion.engine import Debouncer, build_observations, generate_frame_speech, score_and_rank

app = FastAPI(title="Drishti backend")

# ---------------------------------------------------------------------------
# Loaded ONCE per server process, at import time -- not per frame, not per
# connection. detection.detector and depth.depth_estimator already load
# their models at THEIR OWN import time (see those files' module-level
# code), so importing them above already triggered both (slow) model
# loads before this server starts accepting connections. That's why the
# server's first startup takes several seconds -- it's loading YOLOv8n and
# Depth Anything V2, not a bug.
# ---------------------------------------------------------------------------
CALIBRATION = load_calibration()

# A single-worker thread pool for the CPU-heavy synchronous pipeline calls
# (YOLO inference, depth model inference). Deliberately max_workers=1, not
# more: this is a CPU-only laptop demo (Section 3 of PROJECT_CONTEXT.md) --
# extra worker threads wouldn't add real parallelism on the same CPU cores
# the model inference already saturates, and using exactly one worker also
# serializes access to the shared, stateful model objects loaded once at
# import time in detector.py / depth_estimator.py, which is the safer
# assumption without having specifically verified those libraries are
# thread-safe under truly concurrent inference calls.
_executor = ThreadPoolExecutor(max_workers=1)


def _run_pipeline_sync(frame: np.ndarray, debouncer: Debouncer, current_time: float) -> dict:
    """
    The actual detection -> depth -> fusion pipeline for ONE frame.
    Synchronous / blocking on purpose -- this is only ever called via
    loop.run_in_executor() below, so it runs on a worker thread and never
    blocks the asyncio event loop that every other WebSocket connection
    also depends on.
    """
    t0 = time.time()
    detections = detect(frame)
    depth_map = estimate_depth(frame)
    frame_width = frame.shape[1]

    observations = build_observations(detections, depth_map, CALIBRATION, frame_width)
    ranked = score_and_rank(observations)

    speech = None
    if debouncer.should_speak(ranked, current_time):
        speech = generate_frame_speech(ranked)
        debouncer.mark_spoken(ranked, current_time)

    # Cast to plain JSON-safe types -- numpy int64/float32 and tuples
    # don't serialize cleanly through FastAPI's default JSON encoder.
    detections_out = [
        {
            "class_name": o["class_name"],
            "distance_m": round(float(o["distance_m"]), 2),
            "bbox": [int(v) for v in o["bbox"]],
            "frame_position": o["frame_position"],
            "urgency": round(float(o["urgency"]), 3),
        }
        for o in ranked
    ]
    latency_ms = round((time.time() - t0) * 1000, 1)
    return {"speech": speech, "detections": detections_out, "latency_ms": latency_ms}


def _decode_frame(b64_str: str):
    """base64 JPEG string -> BGR np.ndarray, or None if decoding fails."""
    try:
        jpg_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # None if cv2 can't decode it
    except Exception:
        return None


@app.get("/health")
async def health():
    return {"status": "ok", "device": "cpu"}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()

    # Per-CONNECTION state. Each client (each phone) gets its OWN debounce
    # timer -- sharing one Debouncer across multiple simultaneous clients
    # would be wrong, since one phone's announcement timing has nothing to
    # do with another's.
    debouncer = Debouncer()
    loop = asyncio.get_event_loop()

    # "Latest frame" mailbox, size 1. THIS is the frame-skip/throttle
    # requirement (Phase 4 spec item 3): the receiver task below always
    # OVERWRITES this slot with whatever frame just arrived, discarding
    # whatever was in it before. The processing loop further down only
    # ever looks at "whatever's in the slot right now" -- so if 3 frames
    # arrive while one slow frame is still being processed, the first 2
    # are silently dropped, and only the newest is ever processed next.
    # We always want the LATEST reality, never a backlog (Section 6 of
    # PROJECT_CONTEXT.md) -- a visually-impaired user needs to hear about
    # what's in front of them NOW, not a stale queue of what used to be.
    mailbox = {"frame": None, "seq": 0}
    processed_seq = -1

    async def receiver():
        """
        Runs concurrently with the processing loop below. Its ONLY job is
        to keep pulling incoming frames off the socket as fast as they
        arrive and drop each one into the mailbox -- it never waits on
        the (slow) AI pipeline itself.
        """
        try:
            while True:
                b64_str = await websocket.receive_text()
                frame = _decode_frame(b64_str)
                if frame is not None:
                    mailbox["frame"] = frame
                    mailbox["seq"] += 1
                # if decode failed: silently ignore this one bad frame and
                # keep waiting for the next one -- never let one corrupt
                # frame kill the connection.
        except WebSocketDisconnect:
            pass  # normal disconnect; the processing loop below will
                  # notice via a failed send and clean up.
        except Exception as e:
            print(f"[backend] receiver task error: {type(e).__name__}: {e}")

    receiver_task = asyncio.create_task(receiver())

    try:
        while True:
            if mailbox["seq"] != processed_seq and mailbox["frame"] is not None:
                frame = mailbox["frame"]
                seq = mailbox["seq"]
                try:
                    result = await loop.run_in_executor(
                        _executor, _run_pipeline_sync, frame, debouncer, time.time()
                    )
                    print(result)
                    await websocket.send_json(result)
                except Exception as e:
                    # One bad/slow frame must never crash the server or
                    # the connection -- log it, tell the client, keep going.
                    print(f"[backend] pipeline error on a frame: {type(e).__name__}: {e}")
                    await websocket.send_json(
                        {"speech": None, "detections": [], "latency_ms": None, "error": str(e)}
                    )
                processed_seq = seq
            else:
                await asyncio.sleep(0.01)  # nothing new yet -- brief yield, not a busy-spin
    except WebSocketDisconnect:
        print("[backend] client disconnected from /ws/stream")
    except Exception as e:
        print(f"[backend] unexpected error on /ws/stream: {type(e).__name__}: {e}")
    finally:
        receiver_task.cancel()


# Serve the frontend (Phase 5) as static files from this same app/port.
# Mounted LAST, deliberately: routes registered above (/health, /ws/stream)
# are matched first by Starlette's routing, so this catch-all mount at "/"
# can never shadow them, even though it's also rooted at "/". Right now
# frontend/ is still empty (Phase 5 hasn't been built) -- that's fine,
# StaticFiles just returns 404s for any file request until then.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
