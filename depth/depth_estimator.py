"""
depth/depth_estimator.py

Phase 2 of Drishti: standalone monocular depth estimation.

WHAT THIS FILE DOES
--------------------
1. Loads a pretrained monocular depth model. We try Depth Anything V2 Small
   first (via the `transformers` pipeline); if that can't load for any
   reason (no internet on first run, missing deps, out-of-memory, etc.) we
   fall back to MiDaS Small (via torch.hub). Whichever one loads is printed
   clearly to the console, along with WHY, if we had to fall back.

2. estimate_depth(frame) -> a per-pixel RELATIVE depth map, same H/W as the
   input frame.

   IMPORTANT CONCEPT (you will be asked this in your viva):
   Neither model outputs metres. Both output values that are monotonically
   related to *inverse* depth (closer objects -> larger raw value), but the
   scale is arbitrary and shifts frame-to-frame / scene-to-scene. That is
   why we need calibration (see below) before we can say "2.3 metres" out
   loud to a user.

3. get_object_distance(depth_map, bbox) -> the MEDIAN relative-depth value
   inside a bounding box. Median instead of mean because a bbox from YOLO
   is rarely a tight, pixel-perfect mask around the object -- it usually
   includes some background at the edges/corners. A few background pixels
   can drag a mean depth value badly off; the median is robust to that kind
   of edge contamination as long as most of the box is genuinely "object".

4. calibrate(ref_points) + relative_to_meters(value, calibration):
   the bridge from "arbitrary relative units" to "metres", built from a
   small number of real, tape-measured reference points. See the big
   comment block above `calibrate()` for the full reasoning -- that
   comment is written so you can basically read it out loud in your
   evaluation.

5. A __main__ block that:
   - loads test_data/sample1.jpg (falls back to a synthetic gradient image
     with a clear warning if that file doesn't exist yet, so the script is
     still runnable standalone before you've dropped in a real photo)
   - prints distance estimates for a couple of hardcoded bboxes
   - displays the depth map as a colour heatmap for a visual sanity check
   - runs `calibrate()` on two placeholder (relative, metres) pairs, and
     tells you loudly that you must replace them with your own
     tape-measured numbers before trusting any metre output.

HONEST LIMITATION OF THIS HANDOFF
-----------------------------------
This file was generated in a network-isolated dev sandbox, so it could NOT
actually be run here -- there was no way to download either model's
weights or test against a real webcam/photo in this environment. Treat it
as reviewed-and-ready-to-run code, not "confirmed working" code. Run it on
your own machine first and paste back the console output (or any
traceback) if anything looks off -- especially the fallback path and the
depth map orientation (near vs far), which are the two things most likely
to need a small tweak on a real machine.
"""

import os
import time
import warnings

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Model loading (with fallback)
# ---------------------------------------------------------------------------
# These are module-level globals so the (potentially slow) model load only
# happens once, the first time this module is imported -- not once per call
# to estimate_depth().

_MODEL_BACKEND = None     # 'depth_anything_v2' or 'midas_small' once loaded
_da_pipe = None            # transformers pipeline object, if using Depth Anything V2
_midas_model = None        # torch model, if using MiDaS
_midas_transform = None    # matching preprocessing transform for MiDaS


def _load_model():
    """
    Try Depth Anything V2 Small first. On ANY failure, fall back to MiDaS
    Small and explain why. This runs once at import time.
    """
    global _MODEL_BACKEND, _da_pipe, _midas_model, _midas_transform

    print("[depth_estimator] Loading depth model ...")
    t0 = time.time()
    try:
        from transformers import pipeline

        _da_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=-1,  # force CPU; change to 0 if you have a CUDA GPU
        )
        _MODEL_BACKEND = "depth_anything_v2"
        print(
            f"[depth_estimator] Loaded Depth Anything V2 Small "
            f"in {time.time() - t0:.1f}s (device=cpu)."
        )
        return
    except Exception as e:  # noqa: BLE001 - we deliberately want a broad catch here
        print(
            "[depth_estimator] Could not load Depth Anything V2 Small "
            f"({type(e).__name__}: {e}). Falling back to MiDaS Small."
        )

    # ---- Fallback: MiDaS Small via torch.hub -----------------------------
    t0 = time.time()
    import torch

    _midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    _midas_model.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    _midas_transform = midas_transforms.small_transform
    _MODEL_BACKEND = "midas_small"
    print(f"[depth_estimator] Loaded MiDaS Small in {time.time() - t0:.1f}s (device=cpu).")


_load_model()


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def estimate_depth(frame: np.ndarray) -> np.ndarray:
    """
    frame: BGR uint8 image, shape (H, W, 3) -- e.g. from cv2.imread() or a
           cv2.VideoCapture frame.

    Returns: float32 array, shape (H, W), of RELATIVE depth.
             Convention used throughout this file: LARGER value = CLOSER.
             (Both backends natively follow this convention already, so no
             flipping is needed -- but if you ever swap in a different
             model, check this assumption first; it silently breaks
             `calibrate()` and every distance number downstream if wrong.)
    """
    if _MODEL_BACKEND is None:
        raise RuntimeError("Depth model failed to load; see console output above.")

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if _MODEL_BACKEND == "depth_anything_v2":
        from PIL import Image

        pil_img = Image.fromarray(rgb)
        result = _da_pipe(pil_img)
        depth = np.array(result["depth"], dtype=np.float32)
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        return depth

    # MiDaS Small path
    import torch

    input_batch = _midas_transform(rgb)
    with torch.no_grad():
        prediction = _midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy().astype(np.float32)


def get_object_distance(depth_map: np.ndarray, bbox) -> float:
    """
    depth_map: output of estimate_depth() (or a crop of it) -- (H, W) float32.
    bbox: (x1, y1, x2, y2) in pixel coordinates, same frame as depth_map.

    Returns the MEDIAN relative-depth value inside the box.

    Why median, not mean: a YOLO bbox is a rectangle, not a mask -- for any
    non-rectangular object (a person, a bicycle) some corner pixels are
    background, not object. A mean gets pulled toward those background
    pixels; a handful of stray corner pixels can shift a mean noticeably.
    The median is unaffected as long as more than half the box is the
    actual object, which is true for the vast majority of real detections.
    """
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return float("nan")

    region = depth_map[y1:y2, x1:x2]
    return float(np.median(region))


# ---------------------------------------------------------------------------
# Calibration: relative depth -> real metres
# ---------------------------------------------------------------------------
#
# THE PROBLEM
# Depth Anything V2 and MiDaS are trained with a *scale- and shift-invariant*
# loss. That's precisely what makes them generalise so well across random
# internet photos, camera types, and scenes -- but it also means the model
# is only claiming "this is closer than that", not "this is 2.3 metres
# away". Two different photos of the exact same 2-metre-away object could
# come back with completely different raw numbers.
#
# THE FIX
# Both models' raw output behaves like an INVERSE-depth (disparity) signal:
# roughly proportional to 1 / real_distance, up to an unknown per-scene
# scale and offset. So instead of fitting real_distance directly against
# the raw value (which is a poor, non-linear fit), we fit:
#
#       1 / real_distance_m  ≈  m * relative_depth_value + c
#
# a plain straight line, using known (relative_value, real_distance_m)
# pairs you collect by: pointing the camera at an object, running
# estimate_depth() + get_object_distance() to get its relative_value, and
# separately measuring the true distance to that object with a tape
# measure. Two such pairs give you an exact solution for (m, c); three or
# more give a least-squares best fit, which is more robust to any one
# measurement being slightly off -- collect 3-4 if you have time before
# your evaluation, not just the minimum 2.
#
# WHY THIS IS DEFENSIBLE IN A VIVA
# - It's the standard approach for turning affine-invariant relative depth
#   into metric depth (this is literally what "metric-depth fine-tuned"
#   variants of these models try to bake in at training time instead of
#   doing manually -- we're doing the lightweight, no-fine-tuning version).
# - It's transparent: you can show the two tape-measured numbers, show the
#   fitted line, and show the resulting predictions next to reality in
#   evaluation/eval_depth.py (Phase 7).
# - It's honest about its limits: it's a single global linear fit per
#   camera setup / rough scene depth range, not a physically exact model.
#   If your evaluator asks "does this hold at 20 metres if you only
#   calibrated with points at 1-4 metres?" -- the honest answer is "no,
#   linear extrapolation beyond the calibrated range is unreliable, which
#   is a documented limitation", not "yes, perfectly".


def calibrate(ref_points):
    """
    ref_points: list of (relative_depth_value, real_meters) tuples.
                Needs >= 2 points; 3-4 spread across your expected working
                distance range (e.g. 0.5m, 1.5m, 3m, 5m) is noticeably more
                robust than exactly 2.

    Returns: calibration dict {'m': float, 'c': float} to pass into
             relative_to_meters().
    """
    if len(ref_points) < 2:
        raise ValueError("calibrate() needs at least 2 (relative_value, real_meters) points.")

    rel = np.array([p[0] for p in ref_points], dtype=np.float64)
    meters = np.array([p[1] for p in ref_points], dtype=np.float64)

    if np.any(meters <= 0):
        raise ValueError("real_meters values must all be positive.")

    inv_meters = 1.0 / meters
    # degree-1 fit: exact solution for 2 points, least-squares for 3+.
    m, c = np.polyfit(rel, inv_meters, 1)
    return {"m": float(m), "c": float(c)}


def relative_to_meters(relative_depth_value: float, calibration: dict) -> float:
    """
    Applies a calibration dict from calibrate() to convert one relative
    depth value into an estimated real-world distance in metres.
    """
    inv_m = calibration["m"] * relative_depth_value + calibration["c"]
    inv_m = max(inv_m, 1e-6)  # guard against non-physical (negative/zero) inverse-distance
    return 1.0 / inv_m


# ---------------------------------------------------------------------------
# Calibration persistence
# ---------------------------------------------------------------------------
# WHY THIS EXISTS: Phase 2's __main__ block computes a calibration dict
# {m, c} once, from whatever CALIBRATION_OBJECTS + reference photo were
# used at the time -- but that dict lived only in local variables and
# vanished when the script exited. Phase 4's backend runs continuously and
# needs a calibration dict available at STARTUP, without re-running the
# whole depth-model-load + reference-photo dance every time the server
# boots. So: __main__ now saves its computed calibration to a small JSON
# file, and load_calibration() reads it back. Re-running depth_estimator.py
# after a fresh round of real tape measurements automatically updates this
# file -- no code edits needed elsewhere.

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

# Bootstrap fallback: the actual calibration fit from this project's Phase 2
# run (chair/desk/bed/back-wall-shelf provisional values), so the backend
# can start up even before calibration.json exists on a fresh machine.
# Still provisional -- same caveat as before, replace with real
# measurements before trusting the numbers in an evaluation.
_BOOTSTRAP_DEFAULT_CALIBRATION = {"m": 0.01745352743998648, "c": -0.5103083743909744}
_BOOTSTRAP_IS_PROVISIONAL = True


def save_calibration(calibration: dict, is_provisional: bool, path: str = None) -> str:
    """Writes a calibration dict to disk as JSON. Returns the path used."""
    import json

    path = path or CALIBRATION_FILE
    payload = {**calibration, "provisional": is_provisional}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_calibration(path: str = None) -> dict:
    """
    Loads a calibration dict from disk. Falls back to a baked-in bootstrap
    default (still marked provisional) if the file doesn't exist yet, so
    the backend never hard-crashes on startup just because Phase 2's demo
    hasn't been (re)run on this particular machine.
    """
    import json

    path = path or CALIBRATION_FILE
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data.get("provisional"):
            print(
                f"[depth_estimator] Loaded PROVISIONAL calibration from {path} "
                "-- not tape-measured yet, fine for dev, not for evaluation."
            )
        return {"m": data["m"], "c": data["c"]}

    print(
        f"[depth_estimator] No calibration file at {path} -- using the "
        "built-in bootstrap default (also provisional). Run "
        "'python depth/depth_estimator.py' once to generate a real one."
    )
    return dict(_BOOTSTRAP_DEFAULT_CALIBRATION)


# ---------------------------------------------------------------------------
# Standalone demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    test_path = "test_data/sample1room.jpeg"
    frame = None

    if os.path.exists(test_path):
        frame = cv2.imread(test_path)
        print(f"[demo] Using real test image: {test_path}")
    else:
        # No saved test photo -- try grabbing one real frame from the
        # webcam instead (same fallback pattern as Phase 1's detector.py),
        # so this demo actually looks at something real rather than a
        # meaningless synthetic gradient.
        print(f"[demo] '{test_path}' not found -- trying webcam instead.")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ok, webcam_frame = cap.read()
            cap.release()
            if ok:
                frame = webcam_frame
                print("[demo] Captured one frame from webcam.")
        if frame is None:
            warnings.warn(
                f"No '{test_path}' AND no webcam available -- using a "
                "synthetic placeholder gradient image so the script still "
                "runs end-to-end. This gradient is NOT a real scene, so the "
                "depth/heatmap output on it is not meaningful -- it's only "
                "here so the code path doesn't crash. Drop a real "
                "phone/webcam photo into test_data/sample1.jpg (or enable a "
                "webcam) and re-run for an actual sanity check."
            )
            h, w = 480, 640
            gradient = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
            frame = cv2.merge([gradient, gradient, gradient])

    t0 = time.time()
    depth_map = estimate_depth(frame)
    print(f"[demo] estimate_depth() took {time.time() - t0:.2f}s on a {frame.shape[1]}x{frame.shape[0]} frame "
          f"(backend: {_MODEL_BACKEND})")

    # A couple of hardcoded example bboxes (x1, y1, x2, y2) -- adjust these
    # to actually land on objects once you're using a real photo.
    h, w = frame.shape[:2]
    example_bboxes = [
        (int(w * 0.05), int(h * 0.30), int(w * 0.25), int(h * 0.90)),  # near-ish, left
        (int(w * 0.60), int(h * 0.40), int(w * 0.95), int(h * 0.70)),  # further, right
    ]
    for i, bbox in enumerate(example_bboxes):
        rel = get_object_distance(depth_map, bbox)
        print(f"[demo] bbox {i} {bbox} -> median relative depth = {rel:.4f}")

    # --- Calibration -----------------------------------------------------
    # FILL THIS IN with your own real, tape-measured objects.
    # For each one:
    #   1. Find its pixel box (x1, y1, x2, y2) by hovering over its corners
    #      in Windows Paint and reading the coordinates in the status bar.
    #   2. Physically tape-measure the real distance from the camera to it.
    # Use 3+ objects spread across near/mid/far distances for a better fit.
    CALIBRATION_OBJECTS = [
        # !!! PROVISIONAL CALIBRATION !!!
        # These real_meters values are EYEBALLED ESTIMATES from the room
        # photo, NOT tape-measured. This unblocks Phase 3+ for now, but
        # these numbers are not defensible in an evaluation as-is. Before
        # Phase 7 (accuracy evaluation), come back and replace these 4 with
        # real measurements -- even a phone AR measure app or a known-size
        # object (a ruler, an A4 sheet, a floor tile) laid end-to-end is
        # enough; you don't need an actual tape measure. Any real
        # measurement beats an eyeballed guess.
        {"name": "chair (nearest)",    "bbox": (340, 560, 830, 963),  "real_meters": 0.7},  # PROVISIONAL
        {"name": "desk/monitor",       "bbox": (75,  355, 220, 480),  "real_meters": 1.8},  # PROVISIONAL
        {"name": "bed",                "bbox": (500, 420, 950, 750),  "real_meters": 2.2},  # PROVISIONAL
        {"name": "back wall shelf",    "bbox": (960, 95,  1270, 260), "real_meters": 4.0},  # PROVISIONAL
    ]
    # Flip this to False once every entry above is a real measurement, not
    # a guess -- it's what triggers the loud console reminder below.
    CALIBRATION_IS_PROVISIONAL = True

    if not CALIBRATION_OBJECTS:
        print(
            "\n[demo] CALIBRATION_OBJECTS is empty -- skipping real calibration. "
            "Edit the list near the top of this __main__ block with your own "
            "tape-measured (bbox, real_meters) entries, then rerun."
        )
    else:
        if CALIBRATION_IS_PROVISIONAL:
            print(
                "\n[demo] *** WARNING: CALIBRATION_IS_PROVISIONAL = True *** "
                "The real_meters values below are eyeballed estimates, not "
                "measured distances. Fine for unblocking Phase 3+ dev work, "
                "NOT fine to quote in your evaluation. Replace with real "
                "measurements (phone AR app or a known-size object as a "
                "ruler) before Phase 7."
            )
        ref_points = []
        print("\n[demo] Reading relative depth for your calibration objects:")
        for obj in CALIBRATION_OBJECTS:
            rel = get_object_distance(depth_map, obj["bbox"])
            ref_points.append((rel, obj["real_meters"]))
            print(f"  {obj['name']}: bbox={obj['bbox']} -> relative={rel:.4f}, real={obj['real_meters']}m")

        calibration = calibrate(ref_points)
        print(f"[demo] calibration fit = {calibration}")

        saved_path = save_calibration(calibration, CALIBRATION_IS_PROVISIONAL)
        print(f"[demo] Saved calibration to {saved_path} -- Phase 4's backend will load this at startup.")

        print("\n[demo] Checking fit against the same objects (sanity check only --")
        print("       a true test would use a held-out object not in the fit):")
        for obj in CALIBRATION_OBJECTS:
            rel = get_object_distance(depth_map, obj["bbox"])
            predicted = relative_to_meters(rel, calibration)
            print(f"  {obj['name']}: predicted={predicted:.2f}m vs real={obj['real_meters']}m")

    # --- Heatmap visualisation (with boxes drawn on, so you can SEE what
    # each bbox is actually landing on -- this is what was missing before) --
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    def draw_box(img, bbox, label, color):
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # green = the two hardcoded placeholder demo boxes (not real objects)
    for i, bbox in enumerate(example_bboxes):
        draw_box(heatmap, bbox, f"ex{i}", (0, 255, 0))

    # yellow = your real calibration objects, once you've filled them in
    for obj in CALIBRATION_OBJECTS:
        draw_box(heatmap, obj["bbox"], obj["name"], (0, 255, 255))

    cv2.imshow("Depth heatmap (brighter/warmer = closer)", heatmap)
    print("\n[demo] Displaying depth heatmap with boxes drawn on. Green = placeholder")
    print("       demo boxes, yellow = your CALIBRATION_OBJECTS (if any). Press any")
    print("       key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
