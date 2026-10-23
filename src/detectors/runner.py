"""Detector pipeline orchestrator and execution runner (runner.py).

Builds graph representations, executes all pattern detectors, benchmarks execution runtimes,
and serializes generated signals to JSON.
"""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import networkx as nx

from src.detectors.base import DetectorConfig, Signal
from src.detectors.cycles import find_time_bounded_cycles
from src.detectors.fan import compute_fan_features
from src.detectors.passthrough import find_pass_through
from src.detectors.structuring import find_structuring
from src.graph.builder import build_collapsed_graph, build_multigraph, load_data


def run_all_detectors(
    multigraph: nx.MultiDiGraph,
    collapsed: nx.DiGraph,
    config: DetectorConfig | None = None,
) -> Tuple[List[Signal], Dict[str, float]]:
    """Execute all pattern detectors and collect execution runtimes.

    Args:
        multigraph: Raw MultiDiGraph with individual timestamped transactions.
        collapsed: Aggregated DiGraph for topological searches.
        config: Centralized detector thresholds.

    Returns:
        Tuple of:
          - List[Signal]: All emitted fraud pattern signals.
          - Dict[str, float]: Benchmark runtimes in seconds for each detector.
    """
    cfg = config or DetectorConfig()
    all_signals: List[Signal] = []
    runtimes: Dict[str, float] = {}

    # 1. Circular flows (cycles)
    t0 = time.perf_counter()
    cycle_signals = find_time_bounded_cycles(multigraph, collapsed, cfg)
    runtimes["cycles"] = time.perf_counter() - t0
    all_signals.extend(cycle_signals)

    # 2. Pass-through accounts & mule chains
    t0 = time.perf_counter()
    mule_signals = find_pass_through(multigraph, cfg)
    runtimes["passthrough"] = time.perf_counter() - t0
    all_signals.extend(mule_signals)

    # 3. Fan-in / Fan-out aggregation-dispersal hubs
    t0 = time.perf_counter()
    _, fan_signals = compute_fan_features(multigraph, cfg)
    runtimes["fan_in_fan_out"] = time.perf_counter() - t0
    all_signals.extend(fan_signals)

    # 4. Structuring (smurfing)
    t0 = time.perf_counter()
    struct_signals = find_structuring(multigraph, cfg)
    runtimes["structuring"] = time.perf_counter() - t0
    all_signals.extend(struct_signals)

    return all_signals, runtimes


def print_detector_summary(
    signals: List[Signal], runtimes: Dict[str, float], out_path: Path
) -> None:
    """Format and print an executive summary of detection results and benchmark runtimes."""
    pattern_counts: Dict[str, int] = {}
    for sig in signals:
        pattern_counts[sig.pattern_type] = pattern_counts.get(sig.pattern_type, 0) + 1

    total_time = sum(runtimes.values())

    print("\n" + "=" * 65)
    print("           FRAUDGRAPH DETECTION PIPELINE SUMMARY")
    print("=" * 65)
    print(f"Total Signals Emitted: {len(signals):,}")
    print(f"Pipeline Runtime     : {total_time:.3f} seconds")
    print(f"Signals Output Path  : {out_path.resolve()}")
    print("-" * 65)
    print("DETECTION COUNTS BY PATTERN TYPE:")
    if not pattern_counts:
        print("  • (No signals generated)")
    else:
        for p_type, count in sorted(pattern_counts.items()):
            print(f"  • {p_type:<22}: {count:,} alerts")

    print("-" * 65)
    print("DETECTOR RUNTIME BENCHMARKS:")
    for detector_name, sec in runtimes.items():
        pct = (sec / total_time * 100) if total_time > 0 else 0
        print(f"  • {detector_name:<22}: {sec:.4f}s ({pct:5.1f}%)")
    print("=" * 65 + "\n")
