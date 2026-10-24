"""Detector pipeline orchestrator and execution runner (runner.py).

Builds graph representations, executes all pattern detectors (including Louvain communities
and shared hardware clusters), benchmarks runtimes, and outputs detection summaries.
"""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import pandas as pd

from src.detectors.base import DetectorConfig, Signal
from src.detectors.communities import find_suspicious_communities
from src.detectors.cycles import find_time_bounded_cycles
from src.detectors.device_clusters import find_shared_device_clusters
from src.detectors.fan import compute_fan_features
from src.detectors.passthrough import find_pass_through
from src.detectors.structuring import find_structuring


def run_all_detectors(
    multigraph: nx.MultiDiGraph,
    collapsed: nx.DiGraph,
    accounts_df: pd.DataFrame,
    config: DetectorConfig | None = None,
) -> Tuple[List[Signal], Dict[str, float], Dict[str, Any]]:
    """Execute all pattern detectors and collect execution runtimes and diagnostics.

    Args:
        multigraph: Raw MultiDiGraph with individual timestamped transactions.
        collapsed: Aggregated DiGraph for topological searches.
        accounts_df: Tabular account metadata for entity attribution.
        config: Centralized detector thresholds.

    Returns:
        Tuple of:
          - List[Signal]: All emitted fraud pattern signals.
          - Dict[str, float]: Benchmark runtimes in seconds for each detector.
          - Dict[str, Any]: Diagnostics (community breakdown, device cluster breakdown).
    """
    cfg = config or DetectorConfig()
    all_signals: List[Signal] = []
    runtimes: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}

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

    # 5. Suspicious dense communities (Louvain)
    t0 = time.perf_counter()
    communities_df, comm_signals = find_suspicious_communities(collapsed, accounts_df, cfg)
    runtimes["communities"] = time.perf_counter() - t0
    all_signals.extend(comm_signals)

    # 6. Shared device & IP clusters
    t0 = time.perf_counter()
    clusters_df, dev_signals = find_shared_device_clusters(
        multigraph, accounts_df, cfg, prior_signals=all_signals
    )
    runtimes["device_clusters"] = time.perf_counter() - t0
    all_signals.extend(dev_signals)

    # Aggregate diagnostics for communities
    if not communities_df.empty:
        total_comm = len(communities_df)
        flagged_comm = int(communities_df["is_flagged"].sum())
        too_small = sum(1 for r in communities_df["reason_not_flagged"] if "below minimum size" in r)
        too_large = sum(1 for r in communities_df["reason_not_flagged"] if "oversized" in r)
        not_insular = sum(1 for r in communities_df["reason_not_flagged"] if "insularity" in r)
        diagnostics["communities"] = {
            "total": total_comm,
            "flagged": flagged_comm,
            "skipped_too_small": too_small,
            "skipped_too_large": too_large,
            "skipped_not_insular": not_insular,
        }
    else:
        diagnostics["communities"] = {"total": 0, "flagged": 0}

    # Aggregate diagnostics for device clusters
    if not clusters_df.empty:
        total_clusters = len(clusters_df)
        flagged_clusters = int(clusters_df["is_flagged"].sum())
        diagnostics["device_clusters"] = {
            "total": total_clusters,
            "flagged": flagged_clusters,
        }
    else:
        diagnostics["device_clusters"] = {"total": 0, "flagged": 0}

    return all_signals, runtimes, diagnostics


def print_detector_summary(
    signals: List[Signal],
    runtimes: Dict[str, float],
    diagnostics: Dict[str, Any],
    out_path: Path,
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
            print(f"  • {p_type:<24}: {count:,} alerts")

    print("-" * 65)
    comm_diag = diagnostics.get("communities", {})
    print(f"COMMUNITY ANALYSIS (Louvain Partitioning):")
    print(f"  • Total Partitions Discovered   : {comm_diag.get('total', 0):,}")
    print(f"  • Flagged Suspicious (Insular)  : {comm_diag.get('flagged', 0):,}")
    print(f"  • Skipped (Too Small < 3)       : {comm_diag.get('skipped_too_small', 0):,}")
    print(f"  • Skipped (Oversized > 30)      : {comm_diag.get('skipped_too_large', 0):,}")
    print(f"  • Skipped (Not Insular < 70%)   : {comm_diag.get('skipped_not_insular', 0):,}")

    dev_diag = diagnostics.get("device_clusters", {})
    print("-" * 65)
    print(f"HARDWARE & IP CLUSTERS:")
    print(f"  • Total Co-Located Groups       : {dev_diag.get('total', 0):,}")
    print(f"  • Flagged Active Fraud Clusters : {dev_diag.get('flagged', 0):,}")

    print("-" * 65)
    print("DETECTOR RUNTIME BENCHMARKS:")
    for detector_name, sec in runtimes.items():
        pct = (sec / total_time * 100) if total_time > 0 else 0
        print(f"  • {detector_name:<24}: {sec:.4f}s ({pct:5.1f}%)")
    print("=" * 65 + "\n")
