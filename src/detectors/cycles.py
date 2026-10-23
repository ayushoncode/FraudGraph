"""Time-bounded circular flow detector (cycles.py).

Detects laundering rings where funds travel in closed loops (A -> B -> C -> ... -> A)
within a defined time window with amount retention.
"""

from datetime import datetime
from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import pandas as pd

from src.detectors.base import DetectorConfig, Signal


def _canonical_cycle(cycle_nodes: List[str]) -> Tuple[str, ...]:
    """Return a unique rotation-invariant representation of a cycle.

    Interview Explanation:
    In a directed cycle, [A, B, C], [B, C, A], and [C, A, B] describe the exact same loop.
    Finding the minimum node lexicographically and rotating so it starts the tuple ensures
    each cycle is inspected and reported exactly once.
    """
    min_idx = min(range(len(cycle_nodes)), key=lambda i: cycle_nodes[i])
    return tuple(cycle_nodes[min_idx:] + cycle_nodes[:min_idx])


def find_time_bounded_cycles(
    multigraph: nx.MultiDiGraph,
    collapsed: nx.DiGraph,
    config: DetectorConfig | None = None,
) -> List[Signal]:
    """Find circular fund flows satisfying temporal and financial retention constraints.

    Step 1: Rapid candidate cycle discovery on the collapsed graph using nx.simple_cycles.
            Restricted to strongly connected components with length bounds.
    Step 2: Strict empirical verification on multigraph transactions:
            - Tests rotations to identify the true initiating transaction
            - Strictly increasing timestamps along the cycle
            - Completed within cycle_window_hours
            - Amount retention per hop between retention_min and retention_max

    Args:
        multigraph: Raw MultiDiGraph with individual timestamped transactions.
        collapsed: Simple DiGraph with aggregated edge pairs.
        config: Detector threshold configuration.

    Returns:
        List of verified Signal objects with pattern_type='cycle'.
    """
    cfg = config or DetectorConfig()
    signals: List[Signal] = []
    seen_canonical_cycles: Set[Tuple[str, ...]] = set()
    seen_txn_sets: Set[Tuple[str, ...]] = set()
    signal_counter = 1

    # Pre-filter: Optionally skip high-degree hub nodes (merchants, payroll)
    # Trade-off: Speeds up cycle search by cutting combinatorial explosion,
    # but could miss fraud rings intentionally routing through a compromised hub.
    search_graph = collapsed
    if cfg.skip_hubs:
        hubs = [
            n for n in collapsed.nodes()
            if (collapsed.in_degree(n) + collapsed.out_degree(n)) >= cfg.hub_degree_threshold
        ]
        if hubs:
            search_graph = collapsed.copy()
            search_graph.remove_nodes_from(hubs)

    # Search only inside strongly connected components of sufficient size
    sccs = [
        scc for scc in nx.strongly_connected_components(search_graph)
        if len(scc) >= cfg.cycle_min_len
    ]

    candidate_count = 0
    candidate_cycles: List[List[str]] = []

    for scc in sccs:
        sub_g = search_graph.subgraph(scc)
        # NetworkX 3.1+ supports length_bound in simple_cycles
        for raw_cycle in nx.simple_cycles(sub_g, length_bound=cfg.cycle_max_len):
            if len(raw_cycle) < cfg.cycle_min_len:
                continue

            canonical = _canonical_cycle(raw_cycle)
            if canonical in seen_canonical_cycles:
                continue
            seen_canonical_cycles.add(canonical)

            candidate_cycles.append(raw_cycle)
            candidate_count += 1
            if candidate_count >= cfg.cycle_max_candidates:
                break
        if candidate_count >= cfg.cycle_max_candidates:
            break

    # Step 2: Empirically verify candidate cycles against individual transactions
    for base_cycle_nodes in candidate_cycles:
        k = len(base_cycle_nodes)
        canonical_key = _canonical_cycle(base_cycle_nodes)

        # Check all k possible starting positions (rotations) of this cycle
        # This handles arbitrary node ordering returned by simple_cycles
        for rot_idx in range(k):
            cycle_nodes = base_cycle_nodes[rot_idx:] + base_cycle_nodes[:rot_idx]
            first_src = cycle_nodes[0]
            first_dst = cycle_nodes[1]

            if not multigraph.has_edge(first_src, first_dst):
                continue

            first_hop_txns = list(multigraph.get_edge_data(first_src, first_dst).values())
            first_hop_txns.sort(key=lambda t: pd.to_datetime(t["timestamp"]))

            for t0 in first_hop_txns:
                current_time = pd.to_datetime(t0["timestamp"])
                current_amount = float(t0["amount"])
                matched_txns = [t0]
                amounts = [current_amount]
                retentions = [1.0]
                cycle_valid = True

                # Advance along remaining hops: 1 -> 2, 2 -> 3, ..., (k-1) -> 0
                for hop_idx in range(1, k):
                    src = cycle_nodes[hop_idx]
                    dst = cycle_nodes[(hop_idx + 1) % k]

                    if not multigraph.has_edge(src, dst):
                        cycle_valid = False
                        break

                    # Get candidate transactions occurring after current_time
                    candidates = []
                    for edge_dict in multigraph.get_edge_data(src, dst).values():
                        ts = pd.to_datetime(edge_dict["timestamp"])
                        if ts > current_time:
                            span_hours = (ts - pd.to_datetime(t0["timestamp"])).total_seconds() / 3600.0
                            if span_hours <= cfg.cycle_window_hours:
                                candidates.append((ts, edge_dict))

                    if not candidates:
                        cycle_valid = False
                        break

                    candidates.sort(key=lambda item: item[0])

                    # Find earliest candidate satisfying amount retention
                    hop_found = False
                    for ts, data in candidates:
                        amt = float(data["amount"])
                        retention = amt / current_amount if current_amount > 0 else 0.0
                        if cfg.cycle_retention_min <= retention <= cfg.cycle_retention_max:
                            matched_txns.append(data)
                            amounts.append(amt)
                            retentions.append(round(retention, 4))
                            current_time = ts
                            current_amount = amt
                            hop_found = True
                            break

                    if not hop_found:
                        cycle_valid = False
                        break

                if cycle_valid and len(matched_txns) == k:
                    # Deduplicate by unique set of transaction IDs
                    sorted_txns = tuple(sorted(t["txn_id"] for t in matched_txns))
                    if sorted_txns in seen_txn_sets:
                        continue
                    seen_txn_sets.add(sorted_txns)

                    start_ts_str = str(matched_txns[0]["timestamp"])
                    end_ts_str = str(matched_txns[-1]["timestamp"])
                    total_span_hours = (
                        pd.to_datetime(end_ts_str) - pd.to_datetime(start_ts_str)
                    ).total_seconds() / 3600.0

                    sig = Signal(
                        signal_id=f"sig_cycle_{signal_counter:04d}",
                        pattern_type="cycle",
                        accounts=list(canonical_key),
                        txn_ids=[t["txn_id"] for t in matched_txns],
                        start_ts=start_ts_str,
                        end_ts=end_ts_str,
                        total_amount=round(sum(amounts), 2),
                        details={
                            "length": k,
                            "span_hours": round(total_span_hours, 2),
                            "amounts_per_hop": [round(a, 2) for a in amounts],
                            "retention_per_hop": retentions,
                        },
                    )
                    signals.append(sig)
                    signal_counter += 1

    return signals
