"""Suspicious pattern detectors (cycles, pass-through mules, fan-in/fan-out, structuring)."""

from src.detectors.base import DetectorConfig, Signal
from src.detectors.cycles import find_time_bounded_cycles
from src.detectors.fan import compute_fan_features
from src.detectors.passthrough import find_pass_through
from src.detectors.runner import run_all_detectors
from src.detectors.structuring import find_structuring

__all__ = [
    "DetectorConfig",
    "Signal",
    "find_time_bounded_cycles",
    "find_pass_through",
    "compute_fan_features",
    "find_structuring",
    "run_all_detectors",
]
