# Drishti — Phase-Wise Build Prompts

## How to use this file

1. Save **`PROJECT_CONTEXT.md`** in your project root. If you're using **Claude Code**, rename it to `CLAUDE.md` — it loads automatically at the start of every session, so you never have to re-paste it. If you're pasting into a regular Claude chat instead, paste the full content of `PROJECT_CONTEXT.md` once at the start of each new conversation.
2. Work through the phases **in order**. Copy the prompt block for the current phase, send it as-is (edit any bracketed placeholders first), and let the assistant build.
3. Before moving to the next phase, check it against **"Definition of done."** Don't skip this — a broken Phase 2 makes Phase 4 much harder to debug.
4. Tick the box in `PROJECT_CONTEXT.md` Section 9 once a phase is done, so the next session picks up exactly where you left off.
5. Phase 9 is optional — do it only if time allows after Phase 8 is solid. A working core system beats a half-built stretch feature, every time an evaluator is scoring you.

| Phase | What it builds | Est. time |
|---|---|---|
| 0 | Environment & repo scaffolding | 1 day |
| 1 | Object detection module | 3–4 days |
| 2 | Depth estimation module | 3–4 days |
| 3 | Fusion engine (the "brain") | 4–5 days |
| 4 | Backend API (FastAPI + WebSocket) | 3–4 days |
| 5 | Frontend PWA (phone camera + speech) | 4–5 days |
| 6 | End-to-end integration & tuning | 3–4 days |
| 7 | Accuracy & performance evaluation | 4–5 days |
| 8 | Deployment (local demo + cloud) | 2–3 days |
| 9 | Stretch features *(optional)* | flexible |
| 10 | Documentation, report, demo video | 3–4 days |

That's ~6–7 weeks core + buffer — matches the "5 weeks or more" you put on the proposal form, with room to breathe.

---

## Phase 0 — Environment & Repo Setup

**Goal:** a clean, version-controlled skeleton with nothing AI-related in it yet.

**Prompt:**
```
I'm building "Drishti" — I've pasted/attached the full project context above.
Right now I only need Phase 0: scaffolding. Please:

1. Create the folder structure exactly as laid out in Section 5 of the context
   file, with empty placeholder files.
2. Set up a Python virtual environment (venv) with instructions for my OS.
3. Write requirements.txt with: ultralytics, fastapi, "uvicorn[standard]",
   opencv-python, torch, torchvision, transformers, websockets,
   python-multipart, numpy, Pillow.
4. Add a .gitignore suited for Python + downloaded model weights + venv.
5. Initialize git and make an initial commit.
6. Write a one-paragraph README stub (I'll expand it fully in Phase 10).

Don't implement any AI logic yet — this phase is pure scaffolding. Show me
the exact commands you ran and the final folder tree.
```

**Definition of done:** `git log` shows one commit, `pip install -r requirements.txt` succeeds in a fresh venv, folder tree matches Section 5.

**Common pitfalls:** torch installs can be huge/slow on some connections — if it times out, install torch separately first with the CPU-only wheel before the rest of requirements.txt.

---

## Phase 1 — Object Detection Module

**Goal:** YOLOv8n detecting relevant objects, standalone — no depth, no backend yet.

**Prompt:**
```
Phase 1: object detection only, standalone.

Build detection/detector.py that:
1. Loads a pretrained YOLOv8n model via the ultralytics package
   (auto-downloads weights on first run).
2. Exposes detect(frame: np.ndarray) -> list[dict], returning for each
   detected object: class name, confidence, and bbox (x1,y1,x2,y2) in pixels.
3. Filters to a whitelist relevant to a visually-impaired pedestrian: person,
   bicycle, car, motorcycle, bus, truck, traffic light, stop sign, bench,
   chair, dog, backpack, suitcase — check which of these exist in the COCO
   class list ultralytics ships with and use the exact names.
4. Has a __main__ block: run on test_data/sample1.jpg if it exists, else open
   my laptop webcam. Draw labeled boxes on the frame and display with OpenCV
   so I can visually confirm it's working. Print inference time per frame
   in ms.

Keep it CPU-only, no CUDA assumptions. Tell me the exact command to run it
and what I should see.
```

**Definition of done:** a window pops up with boxes + labels on real objects; console prints per-frame ms.

**Common pitfalls:** if no webcam and no sample image, ask for a phone photo to drop into `test_data/` first rather than guessing.

---

## Phase 2 — Depth Estimation Module

**Goal:** standalone monocular depth, with real-world distance calibration.

**Prompt:**
```
Phase 2: monocular depth estimation, still standalone.

Build depth/depth_estimator.py that:
1. Loads Depth Anything V2 Small via the transformers pipeline
   (model id "depth-anything/Depth-Anything-V2-Small-hf"). If that's too
   heavy to download/run on my machine, fall back to MiDaS Small via
   torch.hub.load('intel-isl/MiDaS', 'MiDaS_small') — try Depth Anything V2
   first and tell me clearly if you had to fall back, and why.
2. Exposes estimate_depth(frame) -> per-pixel relative depth map, same H/W
   as input.
3. Exposes get_object_distance(depth_map, bbox) -> float, using the MEDIAN
   depth value inside the bbox (more robust to edge noise than the mean).
4. These models output RELATIVE inverse depth, not metric meters. Add
   calibrate(relative_depth, ref_points: list[tuple[float,float]]) that
   fits relative-depth-to-real-meters using at least 2 known reference
   pairs — I'll physically measure a couple of objects with a tape measure
   and give you the (relative_value, real_meters) pairs. Explain this
   calibration step clearly, I need to defend it in my evaluation.
5. __main__ block: run on a test image, print estimated distance for a
   couple of hardcoded test bboxes, and display the depth map as a colored
   heatmap so I can sanity-check it visually.

Tell me how long model download + first inference took, and the accuracy
tradeoff between Depth Anything V2 Small and MiDaS Small.
```

**Definition of done:** heatmap displays and looks sensible (near objects clearly different from far background); calibration function runs on at least 2 real tape-measured points without crashing.

**Common pitfalls:** don't skip calibration — an evaluator asking "how do you know that's 2 meters and not 5?" with no answer is worse than a slightly-off number you can explain.

---

## Phase 3 — Fusion Engine (the "brain")

**Goal:** the actual decision-making system — this is what makes it more than "two models bolted together."

**Prompt:**
```
Phase 3: fusion engine — take your time here, this is the core intelligence.

Build fusion/engine.py that:
1. Combines detection + depth outputs for a single frame into a list of
   observations: {class_name, distance_m, bbox, frame_position}, where
   frame_position is 'left'/'center'/'right' based on bbox center.
2. Implements the urgency scoring formula from Section 6 of the context
   file exactly: urgency = 0.5*proximity + 0.3*class_criticality +
   0.2*centrality, using the class-criticality tiers table given there.
   Expose the weights and the criticality dict as constants I can tune later.
3. Sorts by urgency descending, keeps top 3 per frame.
4. Generates a spoken sentence via deterministic templates, e.g.
   "{class} very close, {position}" if distance<1m, else
   "{class} {distance:.1f} meters, {position}". No LLM call — this must work
   instantly and fully offline. Structure the interface so an optional LLM
   rephraser could later plug in behind a flag, without implementing it now.
5. Implements the debounce logic from Section 6: don't repeat an
   announcement unless urgency shifts meaningfully or a new object enters
   the top 3, with a hard minimum ~2 second gap between spoken outputs.
6. __main__ block: feed 3-4 hardcoded fake observations, print the resulting
   sentence, then simulate a sequence of frames to show the debounce working
   (some frames should NOT trigger new speech).

Comment thoroughly — I need to explain *why* median depth, these specific
weights, and the debounce exist, in my viva.
```

**Definition of done:** running the `__main__` block shows at least one frame where a new object should be announced and one where debounce correctly suppresses a repeat.

**Common pitfalls:** resist the urge to make this "smarter" with a real LLM call yet — get the deterministic version rock-solid first; it's what runs during your live demo.

---

## Phase 4 — Backend API (FastAPI + WebSocket)

**Goal:** wire the three modules together behind a real streaming server.

**Prompt:**
```
Phase 4: backend orchestration.

Build backend/main.py that:
1. Exposes a WebSocket endpoint /ws/stream: receives a base64-encoded JPEG
   frame, decodes it with OpenCV, runs it through detection -> depth ->
   fusion (import Phases 1-3 as modules, don't reimplement), and sends back
   JSON: {"speech": <string or null>, "detections": [...], "latency_ms": N}.
2. Runs the pipeline off the main event loop (thread pool executor or
   async-friendly wrapper) so one slow frame doesn't block other clients.
3. Adds a frame-skip/throttle: if still processing when a new frame arrives,
   drop the new one rather than queueing — we always want the LATEST
   reality, not a backlog.
4. Serves the frontend (Phase 5) as static files from the same app, so
   everything runs from one process/port.
5. Adds GET /health returning {"status": "ok", "device": "cpu"}.
6. Handles disconnects/exceptions gracefully — one bad frame must never
   crash the server.

Give me the exact uvicorn command to run this, and a quick way to test the
WebSocket endpoint (command line or a small Python test client) before we
build the real frontend.
```

**Definition of done:** a test client can send one frame and get back a valid JSON response; `/health` returns 200; killing a client mid-stream doesn't crash the server.

---

## Phase 5 — Frontend PWA (phone camera + speech)

**Goal:** the actual thing a visually impaired user's phone runs.

**Prompt:**
```
Phase 5: phone-facing frontend, plain HTML/JS PWA, no framework.

Build frontend/index.html + app.js + manifest.json + a minimal sw.js that:
1. Requests camera access via getUserMedia({video:{facingMode:'environment'}})
   and shows the live feed in a <video> element.
2. Every ~500ms, grabs a frame onto an offscreen canvas, encodes it as a
   reduced-resolution (~480px wide) base64 JPEG, and sends it over a
   WebSocket to /ws/stream.
3. On response, if "speech" isn't null, speaks it via
   SpeechSynthesisUtterance, cancelling any currently-speaking utterance
   first so announcements don't queue and lag behind reality.
4. Shows a minimal high-contrast on-screen overlay of the last sentence +
   current latency, for sighted debugging/demo purposes — the end user
   relies on audio only.
5. Adds manifest.json + a basic service worker so it's installable as a PWA.
6. IMPORTANT: browsers only allow camera access over HTTPS or on exactly
   "localhost" — connecting from a phone to a laptop's LAN IP over plain
   http:// WILL be blocked. Don't try to work around this in frontend code;
   I'll handle it in Phase 8. For now just build it assuming eventual HTTPS,
   and tell me how to test camera + speech on my own laptop browser via
   localhost in the meantime.

Also tell me: can I mock the WebSocket response first, to test speech output
independently of the backend pipeline?
```

**Definition of done:** on `http://localhost:8000` in a laptop browser, camera feed shows, and a mocked "speech" response is audibly spoken.

---

## Phase 6 — End-to-End Integration & Tuning

**Goal:** everything from Phases 1–5 actually working together, at usable speed.

**Prompt:**
```
Phase 6: real end-to-end integration and latency tuning.

1. Wire the frontend to the real backend WebSocket — remove Phase 5's mocks.
2. Add server-side per-stage latency logging (detection ms, depth ms,
   fusion ms, total ms) so we can see exactly where time goes.
3. If total latency is too high for comfortable use on CPU (>1.5-2 sec),
   propose and implement concrete fixes — e.g. run depth estimation only
   every Nth frame while detection runs every frame, reduce input
   resolution further, or export YOLOv8n to ONNX/OpenVINO for faster CPU
   inference — and let me pick which tradeoffs to accept.
4. Add config.yaml (or .env) for frame interval, resolution, urgency
   weights, and debounce timing, so I can retune for the demo without
   touching code.
5. Help me do a real test connecting my phone to my laptop over WiFi (I'll
   sort the HTTPS requirement in Phase 8) and confirm detections + speech
   feel responsive while walking around a room.

Tell me what latency you'd consider "good enough for a real-time safety
demo," and what we're actually hitting.
```

**Definition of done:** a real phone, real WiFi, real walk-around test produces timely, sensible spoken output — not stale or wildly delayed.

---

## Phase 7 — Accuracy & Performance Evaluation

**Goal:** the numbers that prove the "accuracy" part of your grade.

**Prompt:**
```
Phase 7: rigorous evaluation — this is what proves the AI-accuracy part of
my grade.

Build an evaluation/ folder with:
1. eval_detection.py: run YOLOv8n's built-in validation (model.val())
   against a COCO128 subset AND help me set up labeling ~100-150 of my own
   photos (Roboflow free tier or LabelImg, YOLO format) for a second,
   real-world validation set. Report mAP@0.5, precision, recall per
   relevant class.
2. eval_depth.py: reads a CSV I'll fill in (object_name, real_distance_m,
   predicted_distance_m) from ~20-30 tape-measured test objects at varying
   distances. Compute Mean Absolute Error and Mean Absolute Percentage
   Error, and plot predicted vs real distance with a y=x reference line.
3. eval_latency.py: run the full pipeline on a batch of N test frames, log
   per-stage + total latency, report mean/median/p95 as a bar chart.
4. An EVAL_RESULTS.md template pulling all three together, plus a section
   for the qualitative usability test (a volunteer navigating a short
   obstacle course with eyes closed, guided only by the system's audio —
   success rate, near-misses, subjective feedback) that I'll fill in by hand
   after recording it.

Make each script runnable with a single command so I can re-run after any
tuning change.
```

**Definition of done:** you can produce a filled `EVAL_RESULTS.md` with real numbers, not placeholders, after running each script once.

---

## Phase 8 — Deployment (Local Demo + Cloud)

**Goal:** a version that works reliably in front of an evaluator, and (optionally) a public URL for your resume.

**Prompt:**
```
Phase 8: deployment for both the live demo and public access.

1. Local demo (no internet dependency risk on the day): set up BOTH
   (a) a quick tunnel via `cloudflared tunnel --url http://localhost:8000`
   or `ngrok http 8000` for a temporary HTTPS URL I can QR-code, AND
   (b) an mkcert-based self-signed certificate for my laptop's LAN IP so my
   phone can connect directly over WiFi with HTTPS, with no internet
   dependency, in case the venue has none. Document exact steps for both so
   I can choose on the day based on venue WiFi.
2. Public deployment (for the "anyone can use it" resume claim): step-by-step
   for deploying the FastAPI backend to Render or Railway's free tier,
   including a Dockerfile if needed, env var handling, and confirming HTTPS
   works out of the box for camera access.
3. Add a clear fallback message in the frontend if the WebSocket disconnects,
   so the demo never looks silently broken/frozen.

Walk me through testing the deployed version from my own phone before the
actual evaluation day.
```

**Definition of done:** you've personally tested the exact demo-day path (tunnel or LAN cert) from your own phone, start to finish, at least once.

---

## Phase 9 — Stretch Features *(optional — pick 1–2, time allowing)*

**Prompt:**
```
Phase 9 (optional): I want to add [PICK ONE OR TWO]:

1. Text/sign reading — integrate EasyOCR (or Tesseract) for on-demand OCR
   (triggered by a button/long-press, not every frame — too slow for
   real-time) with speech output of the read text.
2. Currency note recognition — a small transfer-learned classifier
   (MobileNet/EfficientNet backbone) on currency photos I'll collect.
3. Indoor/outdoor scene classification — a lightweight Places365-pretrained
   classifier to shift urgency weighting by context (outdoor: prioritize
   vehicles higher; indoor: prioritize furniture/stairs higher).
4. Multilingual speech — let me pick a Web Speech API voice/language and add
   Hindi sentence templates alongside English.
5. Haptic feedback — Vibration API, buzz pattern scaled to urgency, as a
   redundant non-audio channel.

Build it as a separate module that plugs into fusion/engine.py's
observation list — don't rewrite the core pipeline to fit it in.
```

**Definition of done:** the core system (Phases 0–8) still works with the stretch feature disabled — it should never become a single point of failure for your demo.

---

## Phase 10 — Documentation, Report & Demo Video

**Goal:** the materials that actually get you marks, beyond the code.

**Prompt:**
```
Phase 10: documentation and submission materials.

1. Write a complete README.md: problem statement, architecture diagram
   (mermaid or ASCII), setup instructions, how to run locally and demo it,
   a summary of evaluation results, and placeholders for screenshots/GIFs.
2. Draft a project report structure covering: problem, existing solutions
   and their limits, system design, per-module implementation details,
   evaluation methodology + results, limitations, and future scope — pull
   future scope from my original proposal (multilingual, offline TFLite,
   wearable integration, India-specific fine-tuning, haptic feedback, GPS).
3. Give me a script/checklist for a 2-3 minute demo video: problem framing
   first, then the live walkthrough (ideally the eyes-closed obstacle-course
   test), then a quick glance at one evaluation chart.
4. Suggest 3-4 resume bullet points for this project, focused on concrete
   technical contributions (multimodal fusion pipeline, real-time
   constraint handling, quantified accuracy) rather than vague claims.
```

**Definition of done:** README renders cleanly on GitHub, report draft exists, demo video script is ready to shoot against.
