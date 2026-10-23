"""Pass-through account and mule chain detector (passthrough.py).

Identifies money mules that receive inflows and rapidly disperse matching funds
outward with minimal dwell time and low balance retention.
Also stitches contiguous pass-through hops into end-to-end mule chains.
"""

from datetime import datetime
import statistics
from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import pandas as pd

from src.detectors.base import DetectorConfig, Signal


def find_pass_through(
    multigraph: nx.MultiDiGraph,
    config: DetectorConfig | None = None,
) -> List[Signal]:
    """Identify pass-through mule accounts and multi-hop mule chains.

    For each account:
    1. Matches each incoming transfer with subsequent outgoing transfers occurring
       within config.passthrough_minutes.
    2. Checks if pass_through_ratio = (amount_out / amount_in) >= passthrough_min_ratio.
    3. Emits individual 'pass_through' signals and stitches connected hops into
       'pass_through_chain' signals.

    Args:
        multigraph: Directed multigraph with timestamped transaction edges.
        config: Detector threshold configuration.

    Returns:
        List of Signal objects for individual pass-throughs and multi-hop chains.
    """
    cfg = config or DetectorConfig()
    signals: List[Signal] = []
    signal_counter = 1

    # Store individual matched hops: key = (mule_account, in_txn_id)
    # Value = dict of hop details
    account_matched_events: Dict[str, List[Dict[str, Any]]] = {}

    for node in multigraph.nodes():
        # Skip accounts that cannot be pass-throughs (need at least 1 in and 1 out)
        if multigraph.in_degree(node) == 0 or multigraph.out_degree(node) == 0:
            continue

        # Collect incoming transactions
        in_txns: List[Dict[str, Any]] = []
        for u, _, _, data in multigraph.in_edges(node, keys=True, data=True):
            in_txns.append({
                "txn_id": data["txn_id"],
                "src_account": u,
                "amount": float(data["amount"]),
                "timestamp": pd.to_datetime(data["timestamp"]),
            })

        # Collect outgoing transactions
        out_txns: List[Dict[str, Any]] = []
        for _, v, _, data in multigraph.out_edges(node, keys=True, data=True):
            out_txns.append({
                "txn_id": data["txn_id"],
                "dst_account": v,
                "amount": float(data["amount"]),
                "timestamp": pd.to_datetime(data["timestamp"]),
            })

        in_txns.sort(key=lambda t: t["timestamp"])
        out_txns.sort(key=lambda t: t["timestamp"])

        matched_for_node: List[Dict[str, Any]] = []

        for in_tx in in_txns:
            amt_in = in_tx["amount"]
            if amt_in < cfg.passthrough_min_amount:
                continue

            t_in = in_tx["timestamp"]
            # Look for subsequent outflows within passthrough_minutes
            candidate_outs = [
                out_tx for out_tx in out_txns
                if out_tx["timestamp"] > t_in
                and (out_tx["timestamp"] - t_in).total_seconds() / 60.0 <= cfg.passthrough_minutes
            ]

            if not candidate_outs:
                continue

            # Check individual candidate or immediate outbound match
            for out_tx in candidate_outs:
                amt_out = out_tx["amount"]
                ratio = amt_out / amt_in
                delay_minutes = (out_tx["timestamp"] - t_in).total_seconds() / 60.0

                if cfg.passthrough_min_ratio <= ratio <= cfg.passthrough_max_ratio:
                    matched_for_node.append({
                        "mule_account": node,
                        "src_account": in_tx["src_account"],
                        "dst_account": out_tx["dst_account"],
                        "in_txn_id": in_tx["txn_id"],
                        "out_txn_id": out_tx["txn_id"],
                        "amount_in": amt_in,
                        "amount_out": amt_out,
                        "ratio": round(ratio, 4),
                        "delay_minutes": round(delay_minutes, 2),
                        "start_ts": str(t_in),
                        "end_ts": str(out_tx["timestamp"]),
                    })
                    break  # Matched this incoming transaction to earliest matching outflow

        if len(matched_for_node) >= cfg.passthrough_min_events:
            account_matched_events[node] = matched_for_node

    # -------------------------------------------------------------------------
    # Chain Detection: Stitch A -> Mule 1 -> Mule 2 -> ... -> Destination
    # -------------------------------------------------------------------------
    # Build a lookup of out_txn_id -> event and in_txn_id -> event
    out_to_event = {}
    in_to_event = {}
    all_events: List[Dict[str, Any]] = []

    for node_events in account_matched_events.values():
        for ev in node_events:
            all_events.append(ev)
            out_to_event[ev["out_txn_id"]] = ev
            in_to_event[ev["in_txn_id"]] = ev

    # Discover multi-hop chains
    visited_events: Set[str] = set()
    chains: List[List[Dict[str, Any]]] = []

    for ev in all_events:
        ev_key = f"{ev['in_txn_id']}->{ev['out_txn_id']}"
        if ev_key in visited_events:
            continue

        # Trace forward from ev
        current_chain = [ev]
        curr = ev
        while curr["out_txn_id"] in in_to_event:
            next_ev = in_to_event[curr["out_txn_id"]]
            current_chain.append(next_ev)
            curr = next_ev

        if len(current_chain) > 1:
            for c_ev in current_chain:
                visited_events.add(f"{c_ev['in_txn_id']}->{c_ev['out_txn_id']}")
            chains.append(current_chain)

    # Emit Signal for detected chains
    chain_mules: Set[str] = set()
    for chain in chains:
        chain_accounts = [chain[0]["src_account"]]
        chain_txns = [chain[0]["in_txn_id"]]
        delays = []
        ratios = []

        for link in chain:
            chain_accounts.append(link["mule_account"])
            chain_txns.append(link["out_txn_id"])
            delays.append(link["delay_minutes"])
            ratios.append(link["ratio"])
            chain_mules.add(link["mule_account"])

        chain_accounts.append(chain[-1]["dst_account"])

        signals.append(Signal(
            signal_id=f"sig_chain_{signal_counter:04d}",
            pattern_type="pass_through_chain",
            accounts=chain_accounts,
            txn_ids=chain_txns,
            start_ts=chain[0]["start_ts"],
            end_ts=chain[-1]["end_ts"],
            total_amount=round(chain[0]["amount_in"], 2),
            details={
                "chain_length_hops": len(chain),
                "mule_accounts": [link["mule_account"] for link in chain],
                "median_delay_minutes": round(statistics.median(delays), 2),
                "ratios": ratios,
            },
        ))
        signal_counter += 1

    # Emit Signal for individual pass-through mule accounts (including those not in long chains)
    for mule_node, events in account_matched_events.items():
        all_node_txns = set()
        for ev in events:
            all_node_txns.add(ev["in_txn_id"])
            all_node_txns.add(ev["out_txn_id"])

        delays = [ev["delay_minutes"] for ev in events]
        ratios = [ev["ratio"] for ev in events]
        amounts = [ev["amount_in"] for ev in events]

        earliest_start = min(ev["start_ts"] for ev in events)
        latest_end = max(ev["end_ts"] for ev in events)

        # Distinguish whether it is part of a multi-hop chain
        is_in_chain = mule_node in chain_mules

        signals.append(Signal(
            signal_id=f"sig_passthrough_{signal_counter:04d}",
            pattern_type="pass_through",
            accounts=[mule_node],
            txn_ids=sorted(list(all_node_txns)),
            start_ts=earliest_start,
            end_ts=latest_end,
            total_amount=round(sum(amounts), 2),
            details={
                "mule_account": mule_node,
                "event_count": len(events),
                "median_delay_minutes": round(statistics.median(delays), 2),
                "average_pass_through_ratio": round(statistics.mean(ratios), 4),
                "part_of_chain": is_in_chain,
                "matched_pairs": [
                    {
                        "in_txn": ev["in_txn_id"],
                        "out_txn": ev["out_txn_id"],
                        "ratio": ev["ratio"],
                        "delay_min": ev["delay_minutes"],
                    }
                    for ev in events
                ],
            },
        ))
        signal_counter += 1

    return signals
