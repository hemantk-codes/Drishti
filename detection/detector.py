"""
detection/detector.py

Phase 1 of Drishti: standalone object detection.

Loads a pretrained YOLOv8n model (via the `ultralytics` package) and exposes
a simple detect() function that returns class name, confidence, and pixel
bounding box for each object of interest found in a frame.

This module has NO knowledge of depth or fusion yet -- it only answers
"what objects are in this frame, and where?" Phases 2 and 3 build on top
of this.
"""

import time

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Whitelist of classes relevant to a visually-impaired pedestrian.
#
# YOLOv8n ships pretrained on COCO (80 classes). We don't want to announce
# every COCO class (e.g. "toaster", "wine glass") -- only the ones that
# matter for real-world street/indoor navigation. These names must match
# the COCO class names EXACTLY as ultralytics defines them (model.names),
# otherwise the filter below will silently drop everything.
#
# Verified against the COCO-80 list that ships with ultralytics:
#   'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
#   'traffic light', 'stop sign', 'bench', 'chair', 'dog',
#   'backpack', 'suitcase'
# All 13 of these exist verbatim in COCO -- no renaming needed.
# ---------------------------------------------------------------------------
RELEVANT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "chair",
    "dog",
    "backpack",
    "suitcase",
}

# Loaded once at import time, not per-call -- loading the model on every
# detect() call would be extremely slow. "yolov8n.pt" auto-downloads from
# Ultralytics' servers the first time this runs, then caches locally
# (usually in the working directory or ~/.cache) for every run after that.
_model = YOLO("yolov8n.pt")


def detect(frame: np.ndarray) -> list[dict]:
    """
    Run YOLOv8n on a single frame and return only the objects we care about.

    Args:
        frame: a BGR image as a NumPy array (the format OpenCV uses),
               e.g. straight from cv2.imread() or cv2.VideoCapture().read().

    Returns:
        A list of dicts, one per detected object that's in RELEVANT_CLASSES:
            {
                "class_name": str,           # e.g. "person"
                "confidence": float,          # 0.0-1.0
                "bbox": (x1, y1, x2, y2),     # pixel coords, top-left/bottom-right
            }
        Objects outside RELEVANT_CLASSES are silently dropped -- this keeps
        the fusion engine (Phase 3) from ever having to reason about a
        "toaster" urgency score.
    """
    # verbose=False stops ultralytics from printing its own per-frame log
    # line -- we do our own timing/printing in __main__ instead.
    results = _model(frame, conf=0.5, verbose=False)[0]

    detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = _model.names[class_id]

        if class_name not in RELEVANT_CLASSES:
            continue

        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            }
        )

    return detections


def _draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw labeled bounding boxes on a copy of the frame, for visual
    confirmation during development. Not used by the real pipeline later
    (the phone doesn't need to SEE the boxes, only hear the sentence) --
    this is purely a debugging aid for this standalone phase.
    """
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put the label text just above the box; if the box is near the
        # top edge, put it just inside instead so it doesn't get clipped.
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return annotated


if __name__ == "__main__":
    import os

    SAMPLE_PATH = os.path.join("test_data", "sample1.jpg")

    if os.path.exists(SAMPLE_PATH):
        # ---- Single-image mode ----
        print(f"Found {SAMPLE_PATH} -- running detection on a static image.")
        frame = cv2.imread(SAMPLE_PATH)
        if frame is None:
            raise RuntimeError(
                f"OpenCV couldn't read {SAMPLE_PATH} -- is it a valid image file?"
            )

        start = time.perf_counter()
        detections = detect(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"Inference time: {elapsed_ms:.1f} ms")
        print(f"Detected {len(detections)} relevant object(s):")
        for det in detections:
            print(f"  - {det['class_name']} ({det['confidence']:.2f}) at {det['bbox']}")

        annotated = _draw_detections(frame, detections)
        cv2.imshow("Drishti - Phase 1 Detection (press any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # ---- Webcam mode ----
        print(f"{SAMPLE_PATH} not found -- falling back to webcam.")
        print("Press 'q' in the video window to quit.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open webcam (index 0). If you don't have a "
                "webcam either, drop a photo at test_data/sample1.jpg "
                "and re-run this script."
            )

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the webcam -- stopping.")
                break

            start = time.perf_counter()
            detections = detect(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000

            print(f"Inference: {elapsed_ms:.1f} ms | {len(detections)} object(s)", end="\r")

            annotated = _draw_detections(frame, detections)
            cv2.putText(
                annotated,
                f"{elapsed_ms:.0f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Drishti - Phase 1 Detection (press 'q' to quit)", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
