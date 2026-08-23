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

# ---------------------------------------------------------------------------
# CHANGED (was 0.15): with real camera input -- as opposed to Phase 3's
# hardcoded demo observations -- distance_m jitters slightly from frame to
# frame even when nothing in the scene actually moved, because Depth
# Anything V2 re-infers the depth map from scratch on every frame with no
# temporal smoothing. That jitter alone was enough to cross a 0.15 urgency
# swing and re-trigger speech almost every cycle once the 2s floor cleared.
# 0.30 asks for a swing that reflects a real change (object genuinely
# got closer/further or more/less central), not sensor noise.
# ---------------------------------------------------------------------------
DEFAULT_URGENCY_SHIFT_THRESHOLD = 0.30
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
    # -----------------------------------------------------------------
    # NEW: which frame_position pairs count as "the same place" for
    # debounce purposes. get_frame_position() draws two hard lines at
    # exactly 1/3 and 2/3 of the frame width. An object sitting a few
    # pixels either side of one of those lines -- completely normal
    # detection jitter, not real motion -- flips its label between e.g.
    # "left" and "center" from one frame to the next. should_speak()
    # used to treat that label flip as "a new object entered the top 3",
    # which is wrong: it's the SAME object that never actually moved.
    # Only left<->right is excluded, since that genuinely means the
    # object crossed the entire frame and IS worth a fresh announcement.
    # -----------------------------------------------------------------
    _ADJACENT_POSITIONS = {
        ("left", "center"), ("center", "left"),
        ("center", "right"), ("right", "center"),
    }

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

    @classmethod
    def _same_identity(cls, a: Tuple[str, str], b: Tuple[str, str]) -> bool:
        """
        NEW: fuzzy identity match used instead of exact tuple equality.
        Same class AND (same position OR a one-bucket-adjacent position)
        counts as "the same object we already announced" -- see
        _ADJACENT_POSITIONS above for why.
        """
        class_a, pos_a = a
        class_b, pos_b = b
        if class_a != class_b:
            return False
        return pos_a == pos_b or (pos_a, pos_b) in cls._ADJACENT_POSITIONS

    def _find_prev_urgency(self, identity: Tuple[str, str]) -> Optional[float]:
        """
        NEW: fuzzy version of `self.last_urgencies.get(identity)`. A plain
        dict .get() does an EXACT key match, so if an object's position
        label flipped since it was last announced, the exact key would
        never be found -- prev would always come back None, and the loop
        below would treat it as "never seen before" and speak anyway.
        That silently defeated the fuzzy match in should_speak() above,
        which is exactly the bug this method closes.
        """
        for prev_identity, urgency in self.last_urgencies.items():
            if self._same_identity(identity, prev_identity):
                return urgency
        return None

    def should_speak(self, ranked_observations: List[Dict], current_time: float) -> bool:
        if not ranked_observations:
            return False
        if self.last_spoken_time is not None and (current_time - self.last_spoken_time) < self.min_interval_s:
            return False

        current_ids = {self._identity(o) for o in ranked_observations}

        # CHANGED: this used to be `if current_ids != self.last_identities`,
        # an exact set-equality check. That meant ANY position-label flip
        # on ANY object -- even one still sitting in the same physical
        # spot -- made the two sets unequal and forced a re-announcement.
        # This now asks a narrower, more correct question: is there a
        # CURRENT object that doesn't fuzzy-match ANY previously-announced
        # object? Only then is something genuinely new in the top 3.
        if any(not any(self._same_identity(cur, prev) for prev in self.last_identities)
               for cur in current_ids):
            return True

        # CHANGED: uses _find_prev_urgency() (fuzzy) instead of
        # self.last_urgencies.get(...) (exact-key). See that method's
        # docstring -- without this, a position-label flip made `prev`
        # always None here and forced a speak regardless of the fix above.
        for obs in ranked_observations:
            prev = self._find_prev_urgency(self._identity(obs))
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

    # -----------------------------------------------------------------
    # NEW: Part 3 demonstrates the exact bug this patch fixes -- an
    # object that never moved, but whose position label flickers across
    # a bucket boundary between frames due to ordinary detection jitter.
    # Before this patch, every one of these frames after the 2s floor
    # would have re-triggered speech. Now only genuinely new content does.
    # -----------------------------------------------------------------
    print("\n[demo] Part 3: position-boundary jitter (the real-camera bug)\n")
    jitter_debouncer = Debouncer()
    # A person standing still, right at the left/center boundary.
    # frame_width=1280 -> boundary sits at x=426.7. bbox center flickers
    # a few pixels either side of that line between frames -- the object
    # itself hasn't moved.
    left_bbox = (330, 200, 520, 900)    # center_x = 425 -> "left"
    center_bbox = (335, 200, 525, 900)  # center_x = 430 -> "center"
    jitter_sequence = [
        (0.0, left_bbox),
        (2.1, center_bbox),   # position label flips, object didn't move
        (4.3, left_bbox),     # flips back
        (6.5, center_bbox),
    ]
    for t, bbox in jitter_sequence:
        obs = annotate_observation({"class_name": "person", "distance_m": 1.2, "bbox": bbox}, FRAME_WIDTH)
        ranked = score_and_rank([obs])
        speak = jitter_debouncer.should_speak(ranked, t)
        print(f"  t={t:>4.1f}s pos={obs['frame_position']:6s} [{'SPEAK' if speak else 'suppressed'}]")
        if speak:
            jitter_debouncer.mark_spoken(ranked, t)
