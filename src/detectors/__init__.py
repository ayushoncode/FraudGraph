"""Suspicious pattern detectors for FraudGraph."""

from src.detectors.base import DetectorConfig, Signal
from src.detectors.communities import (
    explain_not_flagged,
    find_suspicious_communities,
)
from src.detectors.cycles import find_time_bounded_cycles
from src.detectors.device_clusters import find_shared_device_clusters
from src.detectors.fan import compute_fan_features
from src.detectors.passthrough import find_pass_through
from src.detectors.runner import run_all_detectors
from src.detectors.structuring import find_structuring

__all__ = [
    "DetectorConfig",
    "Signal",
    "explain_not_flagged",
    "find_time_bounded_cycles",
    "find_pass_through",
    "compute_fan_features",
    "find_structuring",
    "find_suspicious_communities",
    "find_shared_device_clusters",
    "run_all_detectors",
]
