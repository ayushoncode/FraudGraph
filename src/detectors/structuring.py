"""Structuring (smurfing) detector (structuring.py).

Detects sub-threshold structuring evasion: multiple transfers directed into a consolidation
account within a time window, each sized intentionally just below statutory reporting limits
(e.g., $35,000 - $49,999 under the $50,000 threshold), accumulating into a much larger sum.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Set
import networkx as nx
import pandas as pd

from src.detectors.base import DetectorConfig, Signal


def find_structuring(
    multigraph: nx.MultiDiGraph,
    config: DetectorConfig | None = None,
) -> List[Signal]:
    """Detect structuring: repeated sub-threshold transfers consolidating into a sink account.

    Args:
        multigraph: Directed multigraph with timestamped transaction edges.
        config: Detector threshold configuration.

    Returns:
        List of Signal objects with pattern_type='structuring'.
    """
    cfg = config or DetectorConfig()
    signals: List[Signal] = []
    signal_counter = 1

    lower_bound = cfg.reporting_threshold * cfg.structuring_lower_bound_ratio
    upper_bound = cfg.reporting_threshold
    window_delta = timedelta(hours=cfg.structuring_window_hours)

    for sink_node in multigraph.nodes():
        if multigraph.in_degree(sink_node) < cfg.structuring_min_transfers:
            continue

        # Collect incoming sub-threshold transactions
        sub_txns: List[Dict[str, Any]] = []
        for u, _, _, data in multigraph.in_edges(sink_node, keys=True, data=True):
            amt = float(data["amount"])
            if lower_bound <= amt < upper_bound:
                sub_txns.append({
                    "txn_id": data["txn_id"],
                    "src": u,
                    "amount": amt,
                    "timestamp": pd.to_datetime(data["timestamp"]),
                })

        if len(sub_txns) < cfg.structuring_min_transfers:
            continue

        sub_txns.sort(key=lambda t: t["timestamp"])

        # Sliding window over sub-threshold transactions
        n = len(sub_txns)
        processed_txn_ids: Set[str] = set()

        for i in range(n):
            w_start = sub_txns[i]["timestamp"]
            w_end = w_start + window_delta

            cluster = [t for t in sub_txns[i:] if t["timestamp"] <= w_end]

            if len(cluster) >= cfg.structuring_min_transfers:
                cluster_txn_ids = set(t["txn_id"] for t in cluster)
                # Avoid emitting duplicate subsets of already reported cluster
                if cluster_txn_ids.issubset(processed_txn_ids):
                    continue

                total_amt = sum(t["amount"] for t in cluster)
                if total_amt >= (upper_bound * 1.5):
                    processed_txn_ids.update(cluster_txn_ids)
                    senders = list(set(t["src"] for t in cluster))
                    all_accounts = [sink_node] + senders

                    signals.append(Signal(
                        signal_id=f"sig_structuring_{signal_counter:04d}",
                        pattern_type="structuring",
                        accounts=all_accounts,
                        txn_ids=sorted(list(cluster_txn_ids)),
                        start_ts=str(cluster[0]["timestamp"]),
                        end_ts=str(cluster[-1]["timestamp"]),
                        total_amount=round(total_amt, 2),
                        details={
                            "sink_account": sink_node,
                            "reporting_threshold": upper_bound,
                            "transfer_count": len(cluster),
                            "amounts": [t["amount"] for t in cluster],
                            "senders": senders,
                        },
                    ))
                    signal_counter += 1

    return signals
