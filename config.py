"""
config.py -- loads config.yaml once at import time. Falls back to the
current hardcoded values if config.yaml is missing or a key is absent,
so a fresh checkout without the file still boots instead of crashing.
"""
import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

_DEFAULTS = {
    "frame": {"interval_ms": 500, "capture_width": 480, "jpeg_quality": 0.6},
    "urgency_weights": {"proximity": 0.5, "class_criticality": 0.3, "centrality": 0.2},
    "debounce": {"min_interval_s": 2.0, "urgency_shift_threshold": 0.30},
}


def load_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        print(f"[config] No config.yaml at {_CONFIG_PATH} -- using built-in defaults.")
        return _DEFAULTS

    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f) or {}

    # shallow-merge each section over the defaults, so a partially-filled
    # config.yaml (e.g. you only added "debounce") doesn't KeyError elsewhere
    merged = dict(_DEFAULTS)
    for section, defaults in _DEFAULTS.items():
        merged[section] = {**defaults, **data.get(section, {})}
    return merged


CONFIG = load_config()