# Drishti

Drishti is a real-time, multimodal AI pipeline for the visually impaired: a phone browser streams its camera feed to a FastAPI/WebSocket server that fuses YOLOv8n object detection with monocular depth estimation (Depth Anything V2 / MiDaS) through a custom urgency-scoring algorithm, then speaks a short prioritized sentence (e.g. *"Person very close, center. Car 4 meters, right."*) back to the phone via the Web Speech API — no app install, no dedicated hardware, CPU-only by default.

*(Full setup instructions, architecture diagram, and evaluation results will be added in Phase 10.)*
