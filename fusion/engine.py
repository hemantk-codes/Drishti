"""
fusion/engine.py -- Phase 3: combines detection + depth into prioritized,
debounced, spoken guidance.
"""

from typing import Callable, Dict, List, Optional, Tuple

URGENCY_WEIGHTS = {"proximity": 0.5, "class_criticality": 0.3, "centrality": 0.2}

CLASS_CRITICALITY = {
    "car": 1.0, "bus": 1.0, "truck": 1.0, "motorcycle": 1.0, "bicycle": 1.0,
    "person": 0.8, "dog": 0.8,
    "stairs": 0.6, "door": 0.6, "traffic light": 0.6, "stop sign": 0.6,
    "chair": 0.4, "bench": 0.4, "backpack": 0.4, "suitcase": 0.4,
}
DEFAULT_CRITICALITY = 0.5
DEFAULT_MIN_INTERVAL_S = 2.0
DEFAULT_URGENCY_SHIFT_THRESHOLD = 0.15
TOP_N = 3


def _bbox_center_x(bbox) -> float:
    x1, _, x2, _ = bbox
    return (x1 + x2) / 2.0


def get_frame_position(bbox, frame_width: float) -> str:
    cx = _bbox_center_x(bbox)
    third = frame_width / 3.0
    if cx < third:
        return "left"
    elif cx < 2 * third:
        return "center"
    return "right"


def compute_centrality_score(bbox, frame_width: float) -> float:
    cx = _bbox_center_x(bbox)
    third = frame_width / 3.0
    if third <= cx <= 2 * third:
        return 1.0
    frac = (third - cx) / third if cx < third else (cx - 2 * third) / third
    frac = min(max(frac, 0.0), 1.0)
    return 1.0 - 0.5 * frac


def get_class_criticality(class_name: str) -> float:
    return CLASS_CRITICALITY.get(class_name, DEFAULT_CRITICALITY)


def annotate_observation(obs: Dict, frame_width: float) -> Dict:
    obs["frame_position"] = get_frame_position(obs["bbox"], frame_width)
    obs["centrality_score"] = compute_centrality_score(obs["bbox"], frame_width)
    obs["class_criticality"] = get_class_criticality(obs["class_name"])
    return obs


def build_observations(detections: List[Dict], depth_map, calibration: Dict, frame_width: float) -> List[Dict]:
    from depth.depth_estimator import get_object_distance, relative_to_meters
    observations = []
    for det in detections:
        bbox = det["bbox"]
        rel = get_object_distance(depth_map, bbox)
        distance_m = relative_to_meters(rel, calibration)
        obs = {"class_name": det["class_name"], "confidence": det.get("confidence"),
               "distance_m": distance_m, "bbox": bbox}
        annotate_observation(obs, frame_width)
        observations.append(obs)
    return observations


def compute_proximity_scores(observations: List[Dict]) -> List[float]:
    if not observations:
        return []
    raw = [1.0 / max(o["distance_m"], 0.3) for o in observations]
    lo, hi = min(raw), max(raw)
    if hi == lo:
        return [1.0 for _ in raw]
    return [(r - lo) / (hi - lo) for r in raw]


def score_and_rank(observations: List[Dict], weights: Dict = URGENCY_WEIGHTS, top_n: int = TOP_N) -> List[Dict]:
    if not observations:
        return []
    proximity_scores = compute_proximity_scores(observations)
    for obs, prox in zip(observations, proximity_scores):
        obs["proximity_score"] = prox
        obs["urgency"] = (weights["proximity"] * prox
                           + weights["class_criticality"] * obs["class_criticality"]
                           + weights["centrality"] * obs["centrality_score"])
    return sorted(observations, key=lambda o: o["urgency"], reverse=True)[:top_n]


def generate_sentence(obs: Dict, use_llm_rephraser: bool = False,
                       llm_rephraser_fn: Optional[Callable[[str, Dict], Optional[str]]] = None) -> str:
    class_name, distance_m, position = obs["class_name"], obs["distance_m"], obs["frame_position"]
    sentence = f"{class_name} very close, {position}" if distance_m < 1.0 else f"{class_name} {distance_m:.1f} meters, {position}"
    if use_llm_rephraser and llm_rephraser_fn is not None:
        try:
            rephrased = llm_rephraser_fn(sentence, obs)
            if rephrased:
                return rephrased
        except Exception:
            pass
    return sentence


def generate_frame_speech(ranked_observations: List[Dict], **kwargs) -> Optional[str]:
    if not ranked_observations:
        return None
    return ". ".join(generate_sentence(o, **kwargs) for o in ranked_observations) + "."


class Debouncer:
    def __init__(self, min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
                 urgency_shift_threshold: float = DEFAULT_URGENCY_SHIFT_THRESHOLD):
        self.min_interval_s = min_interval_s
        self.urgency_shift_threshold = urgency_shift_threshold
        self.last_spoken_time: Optional[float] = None
        self.last_identities: set = set()
        self.last_urgencies: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def _identity(obs: Dict) -> Tuple[str, str]:
        return (obs["class_name"], obs["frame_position"])

    def should_speak(self, ranked_observations: List[Dict], current_time: float) -> bool:
        if not ranked_observations:
            return False
        if self.last_spoken_time is not None and (current_time - self.last_spoken_time) < self.min_interval_s:
            return False
        current_ids = {self._identity(o) for o in ranked_observations}
        if current_ids != self.last_identities:
            return True
        for obs in ranked_observations:
            prev = self.last_urgencies.get(self._identity(obs))
            if prev is None or abs(obs["urgency"] - prev) >= self.urgency_shift_threshold:
                return True
        return False

    def mark_spoken(self, ranked_observations: List[Dict], current_time: float) -> None:
        self.last_spoken_time = current_time
        self.last_identities = {self._identity(o) for o in ranked_observations}
        self.last_urgencies = {self._identity(o): o["urgency"] for o in ranked_observations}


def process_frame(detections, depth_map, calibration, frame_width, debouncer: Debouncer, current_time: float) -> Dict:
    observations = build_observations(detections, depth_map, calibration, frame_width)
    ranked = score_and_rank(observations)
    speech = None
    if debouncer.should_speak(ranked, current_time):
        speech = generate_frame_speech(ranked)
        debouncer.mark_spoken(ranked, current_time)
    return {"speech": speech, "observations": ranked}


if __name__ == "__main__":
    FRAME_WIDTH = 1280
    fake_observations = [
        {"class_name": "car", "distance_m": 4.0, "bbox": (900, 300, 1200, 600)},
        {"class_name": "person", "distance_m": 0.8, "bbox": (550, 200, 750, 900)},
        {"class_name": "chair", "distance_m": 2.5, "bbox": (50, 500, 250, 800)},
        {"class_name": "dog", "distance_m": 1.5, "bbox": (620, 400, 780, 700)},
    ]
    print("[demo] Part 1: one frame\n")
    ranked = score_and_rank([annotate_observation(dict(o), FRAME_WIDTH) for o in fake_observations])
    for o in ranked:
        print(f"  {o['class_name']:8s} dist={o['distance_m']:.1f}m pos={o['frame_position']:6s} urgency={o['urgency']:.3f}")
    print(f"\n[demo] Speech: \"{generate_frame_speech(ranked)}\"")

    print("\n[demo] Part 2: debounce across a frame sequence\n")
    debouncer = Debouncer()
    frame_sequence = [
        (0.0, fake_observations),
        (0.3, fake_observations),
        (2.5, fake_observations),
        (3.0, [{"class_name": "bicycle", "distance_m": 0.5, "bbox": (600, 300, 800, 700)}, *fake_observations]),
        (5.5, fake_observations),
    ]
    for t, obs_list in frame_sequence:
        ranked = score_and_rank([annotate_observation(dict(o), FRAME_WIDTH) for o in obs_list])
        speak = debouncer.should_speak(ranked, t)
        print(f"  t={t:>4.1f}s [{'SPEAK' if speak else 'suppressed'}] top3={[o['class_name'] for o in ranked]}")
        if speak:
            print(f"           -> \"{generate_frame_speech(ranked)}\"")
            debouncer.mark_spoken(ranked, t)
