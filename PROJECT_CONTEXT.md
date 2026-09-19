# Drishti — Project Context
*(Paste this whole file at the start of any new AI chat, or save it as `CLAUDE.md` in your project root if you're using Claude Code — it auto-loads as context every session, so you won't need to paste it manually.)*

This file is the single source of truth for what this project is, why it's built this way, and how far along it is. Update **Section 9** every time you finish a phase — that's what makes this a *living* doc instead of a one-time brief.

---

## 0. Quick facts

| | |
|---|---|
| **Course** | Applied AI and Prototype Development (AAIPD), 3rd year BTech CSE |
| **Grading** | No exam — 100% final evaluation of the working prototype |
| **Graded on** | (a) how much genuine AI is embedded — not one pretrained model wrapped in a UI, and (b) how accurate/measurable the results are |
| **Secondary goal** | Strong enough for a resume/portfolio — code quality, docs, and a real demo all matter |
| **Project title** | Drishti — AI-Powered Real-Time Scene Description and Obstacle-Alert System for the Visually Impaired |
| **GitHub** | https://github.com/hemantk-codes/Drishti (private) |

---

## 1. The problem

India has an estimated ~5 million visually impaired people who navigate daily using a white cane or guide dog. Both solve *"is something there?"* but not *"what is it, how far, and how urgent?"* — a cane doesn't tell you a car is approaching from the right or that there's a step down 2 meters ahead. This project closes that contextual gap using only a smartphone camera — no dedicated hardware, no cost barrier.

## 2. What we're building

A phone (any phone, via browser — no app install) streams its camera feed to a laptop/server. The server runs a real AI pipeline — object detection, depth estimation, and a priority/language engine — and sends back a short spoken sentence the phone speaks aloud in near-real-time. Example: *"Person very close, center. Car 4 meters, right."*

```
 ┌─────────────┐   frame every ~500ms    ┌────────────────────────┐
 │ Phone (PWA) │ ───────────────────────▶│  FastAPI + WebSocket    │
 │ - camera    │                         │       server            │
 │ - speaker   │◀─────────────────────── │                         │
 └─────────────┘   spoken sentence (JSON)└───────────┬────────────┘
                                                      │ frame (np array)
                              ┌───────────────────────┼───────────────────────┐
                              ▼                                               ▼
                    ┌──────────────────┐                          ┌────────────────────────┐
                    │  detection/       │                          │  depth/                  │
                    │  YOLOv8n           │                          │  Depth Anything V2 /     │
                    │  → objects+bboxes  │                          │  MiDaS → depth map       │
                    └─────────┬──────────┘                          └────────────┬────────────┘
                              └─────────────────────┬──────────────────────────┘
                                                     ▼
                                        ┌─────────────────────────┐
                                        │   fusion/engine.py        │
                                        │  urgency scoring +        │
                                        │  debounce + sentence gen  │
                                        └─────────────────────────┘
```

## 3. Tech stack, and why

| Layer | Tool | Why this one |
|---|---|---|
| Frontend | Plain HTML/JS PWA — `getUserMedia`, WebSocket, Web Speech API | No app store, no APK. Any phone browser works. Installable as a home-screen PWA. |
| Backend | Python, FastAPI + WebSockets | Async-friendly, handles streaming naturally, easy to demo locally |
| Object detection | YOLOv8n (Ultralytics), pretrained on COCO | Fast enough for CPU, well-documented, easy to fine-tune later |
| Depth estimation | **Depth Anything V2 (Small)** — primary; **MiDaS Small** — fallback | Depth Anything V2 is a newer, more accurate open monocular depth model; MiDaS is the simpler, more battle-tested fallback if setup gets in the way |
| Fusion / decision logic | Custom priority-scoring algorithm (Section 6) | This is what turns "two models bolted together" into an actual *system* — it's the part you'll defend hardest in your viva |
| Speech output | Browser's built-in Web Speech API | Zero extra latency — text goes back over the socket, the phone speaks it locally, no audio streaming needed |

**Hardware assumption:** everything is scoped to run on a CPU-only laptop by default (nano/small model variants, frame throttling). If you have an NVIDIA GPU, every phase notes how to scale up (YOLOv8s/m, larger depth model, higher frame rate).

## 4. The "AI core" — three models + one algorithm

Grading explicitly rewards "how much AI is injected," so be able to name all four pieces, not just "I used YOLO":

1. **Object detection** — YOLOv8n, a CNN-based detector
2. **Depth estimation** — a transformer-based monocular depth model (Depth Anything V2)
3. **Priority/urgency scoring** — a weighted decision algorithm fusing both models' outputs (below)
4. **Natural language generation** — deterministic templated NLG, structured so an LLM-based rephraser can be swapped in later as an enhancement, not a dependency (a real-time safety tool can't depend on a cloud API's latency or uptime for its core function)

### Urgency scoring formula

```
urgency = 0.5 * proximity_score + 0.3 * class_criticality + 0.2 * centrality_score
```

- **proximity_score** = `1 / max(distance_m, 0.3)`, normalized 0–1 across the current frame's objects
- **class_criticality** — lookup table:

  | Tier | Classes | Weight |
  |---|---|---|
  | Highest | car, bus, truck, motorcycle, bicycle (moving hazards) | 1.0 |
  | High | person, dog | 0.8 |
  | Medium | stairs, doors (drop-offs / static obstacles in path) | 0.6 |
  | Lower | chair, bench, backpack, suitcase (furniture-like) | 0.4 |

- **centrality_score** = 1.0 if the object's bbox center falls in the middle third of frame width, tapering to 0.5 near the edges

Only the top 3 objects by urgency get spoken per frame, and a **debounce**: don't re-announce unless urgency shifts meaningfully or a new object enters the top 3, with a hard minimum ~2 second gap between announcements — otherwise the audio becomes noise instead of guidance.

## 5. Repo structure

```
drishti/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml                  # tunable params: frame interval, resolution, weights, debounce
├── test_data/
│   └── sample1.jpg
├── detection/
│   └── detector.py
├── depth/
│   └── depth_estimator.py
├── fusion/
│   └── engine.py
├── backend/
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── manifest.json
│   └── sw.js
├── evaluation/
│   ├── eval_detection.py
│   ├── eval_depth.py
│   ├── eval_latency.py
│   ├── depth_test_log.csv
│   └── EVAL_RESULTS.md
└── docs/
    └── report_draft.md
```

## 6. Hard constraints — don't let these slip

- **Secure-context rule**: browsers only allow camera access over HTTPS, or on exactly `localhost`. A phone hitting your laptop's LAN IP over plain `http://192.168.x.x:8000` will have the camera **blocked** — this is a browser security rule, not a bug to route around. Fix is in Phase 8 (tunnel or local cert), not in frontend code.
- **Never block the event loop** — one slow frame (a big depth model call) must not freeze the whole WebSocket server for other connections.
- **Graceful degradation** — if the pipeline errors on one frame, log it and continue; never let the server crash mid-demo.
- **CPU-first** — default to nano/small model variants; note the GPU upgrade path but don't require it.

## 7. Evaluation plan — this is what proves "accuracy" to the evaluator

| Metric | How |
|---|---|
| Detection accuracy | mAP@0.5, precision/recall on a COCO subset *and* ~100–150 self-photographed, self-labeled real images (the self-collected set is what proves this solves a *real* problem, not a benchmark) |
| Depth accuracy | Mean Absolute Error against ~20–30 tape-measured real distances |
| Latency | Mean / median / p95 end-to-end time, frame-capture to spoken output |
| Real-world usability | A volunteer navigates a short obstacle course with eyes closed, guided only by the system's audio — success rate + qualitative feedback. This single demo video will do more for your marks than any metrics table. |

## 8. Explicitly out of scope (for now)

Native mobile app, multi-camera setups, cross-session person re-identification, full offline operation. These are legitimate **Future Scope** items for the report, not things to build now.

## 9. Progress tracker — update as you complete each phase

- [✅] Phase 0 — Environment & repo setup
- [✅] Phase 1 — Object detection module
      → `detection/detector.py` built and tested via webcam.
      YOLOv8n, CPU-only, ~55ms/frame inference (~18fps).
      13-class COCO whitelist (person, car, bicycle, bus, truck, motorcycle,
      traffic light, stop sign, bench, chair, dog, backpack, suitcase).
      Confirmed: boxes + labels display correctly, console prints per-frame ms.
- [✅] Phase 2 — Depth estimation module
- [✅] Phase 3 — Fusion engine (priority + NLG)
- [✅] Phase 4 — Backend API (FastAPI + WebSocket)
- [✅] Phase 5 — Frontend PWA
- [✅] Phase 6 — End-to-end integration & tuning (Cloudflare tunnel used for phone/WiFi demo; ~450ms total latency)
- [ ] Phase 7 — Accuracy & performance evaluation
- [ ] Phase 8 — Deployment (local demo + cloud)
- [ ] Phase 9 — Stretch features *(optional)*
- [ ] Phase 10 — Documentation, report, demo video

## 10. How to frame this for the report / resume / viva

Don't describe it as "an app that uses YOLO." Describe it as: *a real-time multimodal AI pipeline that fuses object detection and monocular depth estimation through a custom urgency-scoring algorithm to generate prioritized, spoken guidance for visually impaired users — deployed as an installable, device-agnostic web app.* That sentence alone signals system-design thinking, not model-calling.

## 11. Instructions for whichever AI assistant is reading this file

- Check Section 9 first — only build the next unchecked phase unless told otherwise.
- Prefer small, runnable, independently-testable increments over big multi-file drops.
- Comment code enough that a student can explain every non-obvious line in a viva.
- Default to CPU-only assumptions unless told a GPU is available.
- Ask before any architecture deviation from Sections 2–6 — don't silently redesign the pipeline.
- Never introduce a hard dependency on a paid/cloud API for the *core* pipeline (LLM rephrasing is opt-in only, per Section 4).
