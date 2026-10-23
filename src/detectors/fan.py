"""Fan-in and Fan-out topological feature calculator and detector (fan.py).

Computes rolling in-degree and out-degree metrics across time windows.
Emits a 'fan_in_fan_out' signal ONLY when an account exhibits BOTH high fan-in
(aggregation from many senders) AND high fan-out (dispersal to many receivers)
within the same window—a hallmark of smurfing hubs and laundering aggregators.

Legitimate lookalikes (merchants with high fan-in, or payroll with high fan-out)
exhibit high degree in only ONE direction, thereby avoiding false alarms.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import pandas as pd

from src.detectors.base import DetectorConfig, Signal


def compute_fan_features(
    multigraph: nx.MultiDiGraph,
    config: DetectorConfig | None = None,
) -> Tuple[pd.DataFrame, List[Signal]]:
    """Compute temporal fan-in/fan-out features and detect aggregation-dispersal hubs.

    Args:
        multigraph: Directed multigraph with transaction timestamps.
        config: Detector threshold configuration.

    Returns:
        Tuple of:
          - pd.DataFrame: Tabular feature table (one row per account-window).
          - List[Signal]: Signals for accounts exhibiting simultaneous high fan-in and fan-out.
    """
    cfg = config or DetectorConfig()
    signals: List[Signal] = []
    feature_rows: List[Dict[str, Any]] = []
    signal_counter = 1

    # Extract all transactions chronologically
    all_edges: List[Dict[str, Any]] = []
    for u, v, key, data in multigraph.edges(keys=True, data=True):
        all_edges.append({
            "txn_id": data["txn_id"],
            "src": u,
            "dst": v,
            "amount": float(data["amount"]),
            "timestamp": pd.to_datetime(data["timestamp"]),
        })

    if not all_edges:
        empty_df = pd.DataFrame(columns=[
            "account_id", "account_type", "window_start", "window_end",
            "fan_in", "fan_out", "in_amount", "out_amount"
        ])
        return empty_df, signals

    all_edges.sort(key=lambda e: e["timestamp"])
    min_time = all_edges[0]["timestamp"]
    max_time = all_edges[-1]["timestamp"]

    # Slice the timeline into non-overlapping or daily sliding windows
    window_delta = timedelta(hours=cfg.fan_window_hours)
    current_window_start = min_time

    # Pre-organize transactions per account for fast windowing
    account_in_txns: Dict[str, List[Dict[str, Any]]] = {}
    account_out_txns: Dict[str, List[Dict[str, Any]]] = {}

    for edge in all_edges:
        account_out_txns.setdefault(edge["src"], []).append(edge)
        account_in_txns.setdefault(edge["dst"], []).append(edge)

    # For each account, compute windowed degree features
    nodes = list(multigraph.nodes(data=True))

    for node, attrs in nodes:
        acc_type = attrs.get("account_type", "unknown")
        in_list = account_in_txns.get(node, [])
        out_list = account_out_txns.get(node, [])

        if not in_list and not out_list:
            continue

        # Find timeline range for this specific account
        acc_first = min(
            [e["timestamp"] for e in in_list + out_list]
        )
        acc_last = max(
            [e["timestamp"] for e in in_list + out_list]
        )

        w_start = acc_first
        while w_start <= acc_last:
            w_end = w_start + window_delta

            # Inflow transactions in window
            window_ins = [
                e for e in in_list if w_start <= e["timestamp"] < w_end
            ]
            # Outflow transactions in window
            window_outs = [
                e for e in out_list if w_start <= e["timestamp"] < w_end
            ]

            distinct_senders = set(e["src"] for e in window_ins)
            distinct_receivers = set(e["dst"] for e in window_outs)

            fan_in = len(distinct_senders)
            fan_out = len(distinct_receivers)
            in_amount = round(sum(e["amount"] for e in window_ins), 2)
            out_amount = round(sum(e["amount"] for e in window_outs), 2)

            # Record feature row
            if fan_in > 0 or fan_out > 0:
                feature_rows.append({
                    "account_id": node,
                    "account_type": acc_type,
                    "window_start": str(w_start),
                    "window_end": str(w_end),
                    "fan_in": fan_in,
                    "fan_out": fan_out,
                    "in_amount": in_amount,
                    "out_amount": out_amount,
                })

            # Check for suspicious simultaneous aggregation AND dispersal
            # Only accounts with BOTH high fan-in AND high fan-out trigger an alert
            if (
                fan_in >= cfg.fan_in_threshold
                and fan_out >= cfg.fan_out_threshold
                and (in_amount + out_amount) >= cfg.fan_min_amount
            ):
                all_window_txns = [e["txn_id"] for e in window_ins + window_outs]
                all_involved_accounts = list(distinct_senders.union(distinct_receivers).union({node}))

                signals.append(Signal(
                    signal_id=f"sig_fan_{signal_counter:04d}",
                    pattern_type="fan_in_fan_out",
                    accounts=all_involved_accounts,
                    txn_ids=all_window_txns,
                    start_ts=str(w_start),
                    end_ts=str(w_end),
                    total_amount=round(in_amount + out_amount, 2),
                    details={
                        "hub_account": node,
                        "account_type": acc_type,
                        "fan_in_count": fan_in,
                        "fan_out_count": fan_out,
                        "in_amount": in_amount,
                        "out_amount": out_amount,
                        "distinct_senders": sorted(list(distinct_senders)),
                        "distinct_receivers": sorted(list(distinct_receivers)),
                    },
                ))
                signal_counter += 1

            # Step forward by half the window to catch boundary-straddling patterns
            w_start += window_delta / 2

    features_df = pd.DataFrame(feature_rows)
    return features_df, signals
