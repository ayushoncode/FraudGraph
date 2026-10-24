"""Suspicious dense community detector using Louvain modularity (communities.py).

Partitions the financial transaction graph into closely connected communities,
evaluates internal vs. external financial volume (insularity), and emits alerts
for insular cliques with high internal velocity.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import pandas as pd

from src.detectors.base import DetectorConfig, Signal


def explain_not_flagged(group_metrics: Dict[str, Any], config: DetectorConfig) -> str:
    """Return a concise reason string explaining why a group was not alerted.

    Used by investigators to understand why borderline or benign communities were skipped.
    """
    # 1. Community insularity & sizing rules
    if "insularity" in group_metrics:
        size = group_metrics.get("size", 0)
        insularity = group_metrics.get("insularity", 0.0)

        if size < config.community_min_size:
            return f"size {size} below minimum size {config.community_min_size}"
        if size > config.community_max_size:
            return f"size {size} exceeds maximum size {config.community_max_size} (oversized community)"
        if insularity < config.insularity_min:
            return f"insularity {insularity:.2f} below threshold {config.insularity_min:.2f}"

    # 2. Shared hardware rules
    if "internal_txn_count" in group_metrics:
        size = group_metrics.get("member_count", group_metrics.get("size", 0))
        txns = group_metrics.get("internal_txn_count", 0)

        if size < config.shared_device_min_accounts:
            return f"account count {size} below minimum {config.shared_device_min_accounts}"
        if txns == 0:
            return "shared device but 0 internal transactions"
        if txns < config.device_min_internal_txns:
            return f"internal transactions ({txns}) below threshold {config.device_min_internal_txns}"

    return "within benign baseline thresholds"


def find_suspicious_communities(
    collapsed: nx.DiGraph,
    accounts_df: pd.DataFrame,
    config: DetectorConfig | None = None,
) -> Tuple[pd.DataFrame, List[Signal]]:
    """Partition graph with Louvain and identify insular, suspicious account communities.

    Interview Explanation:
    Why convert directed DiGraph to an undirected weighted Graph?
    In modularity optimization, we seek tightly bound clusters of mutual economic affinity.
    Whether money flows from A -> B or B -> A, both indicate an economic relationship connecting
    the accounts. Summing amounts across both directions yields the total bilateral volume,
    which serves as the ideal edge weight for Louvain clustering.

    Args:
        collapsed: Aggregated directed graph with 'total_amount' and 'txn_ids' on edges.
        accounts_df: DataFrame with account attributes (created_at, account_type).
        config: Detector threshold configuration.

    Returns:
        Tuple of:
          - pd.DataFrame: Comprehensive table of ALL communities with metrics.
          - List[Signal]: Alerts for communities passing insularity and size criteria.
    """
    cfg = config or DetectorConfig()
    signals: List[Signal] = []
    community_records: List[Dict[str, Any]] = []
    signal_counter = 1

    # Map accounts for quick attribute lookups
    account_lookup = accounts_df.set_index("account_id").to_dict(orient="index")

    # Step a: Convert collapsed DiGraph to undirected Graph with summed bilateral weights
    undirected_G = nx.Graph()
    for node, attrs in collapsed.nodes(data=True):
        undirected_G.add_node(node, **attrs)

    for u, v, data in collapsed.edges(data=True):
        amt = float(data.get("total_amount", 0.0))
        if undirected_G.has_edge(u, v):
            undirected_G[u][v]["weight"] += amt
        else:
            undirected_G.add_edge(u, v, weight=amt)

    # Step b: Run deterministic Louvain community detection
    # Handles empty graph edge case gracefully
    if undirected_G.number_of_edges() == 0:
        return pd.DataFrame(), signals

    raw_communities = nx.community.louvain_communities(
        undirected_G,
        weight="weight",
        resolution=cfg.louvain_resolution,
        seed=cfg.louvain_seed,
    )

    # Step c: Evaluate each community
    for comm_idx, comm_set in enumerate(raw_communities, start=1):
        comm_nodes = sorted(list(comm_set))
        size = len(comm_nodes)
        comm_set_lookup = set(comm_nodes)

        # 1. Financial volumes: internal vs. external
        internal_amount = 0.0
        external_amount = 0.0
        internal_edges_count = 0
        reciprocal_edges_count = 0
        internal_txn_ids: List[str] = []
        internal_timestamps: List[pd.Timestamp] = []

        for u in comm_nodes:
            # Outgoing edges from u in collapsed graph
            for _, v, data in collapsed.out_edges(u, data=True):
                amt = float(data.get("total_amount", 0.0))
                if v in comm_set_lookup:
                    internal_amount += amt
                    internal_edges_count += 1
                    internal_txn_ids.extend(data.get("txn_ids", []))
                    if "first_ts" in data and pd.notna(data["first_ts"]):
                        internal_timestamps.append(pd.to_datetime(data["first_ts"]))
                    # Check reciprocity (does v -> u also exist?)
                    if collapsed.has_edge(v, u):
                        reciprocal_edges_count += 1
                else:
                    external_amount += amt

            # Incoming edges into u from outside community
            for w, _, data in collapsed.in_edges(u, data=True):
                if w not in comm_set_lookup:
                    amt = float(data.get("total_amount", 0.0))
                    external_amount += amt

        # Insularity ratio: proportion of total volume recirculated internally
        total_vol = internal_amount + external_amount
        insularity = (internal_amount / total_vol) if total_vol > 0 else 0.0

        # Internal directed density: actual / maximum possible directed edges
        max_possible_edges = size * (size - 1)
        density = (internal_edges_count / max_possible_edges) if max_possible_edges > 0 else 0.0

        # Reciprocity: share of internal directed edges with mutual flow
        reciprocity = (
            reciprocal_edges_count / internal_edges_count
            if internal_edges_count > 0 else 0.0
        )

        # 2. Account type mix
        type_counts = {"individual": 0, "merchant": 0, "business": 0}
        created_dates = []
        for n in comm_nodes:
            acc_meta = account_lookup.get(n, {})
            acc_type = acc_meta.get("account_type", "individual")
            type_counts[acc_type] = type_counts.get(acc_type, 0) + 1
            if "created_at" in acc_meta and pd.notna(acc_meta["created_at"]):
                created_dates.append(pd.to_datetime(acc_meta["created_at"]))

        # 3. Share of new accounts (created within new_account_days of group start)
        group_ref_date = min(internal_timestamps) if internal_timestamps else (
            min(created_dates) if created_dates else pd.to_datetime("2025-01-01")
        )
        new_account_cutoff = group_ref_date - timedelta(days=cfg.new_account_days)
        new_account_count = sum(1 for d in created_dates if d >= new_account_cutoff)
        share_new_accounts = new_account_count / size if size > 0 else 0.0

        # Build metric dictionary
        metrics = {
            "community_id": f"comm_{comm_idx:03d}",
            "size": size,
            "insularity": round(insularity, 4),
            "internal_amount": round(internal_amount, 2),
            "external_amount": round(external_amount, 2),
            "internal_edge_density": round(density, 4),
            "reciprocity": round(reciprocity, 4),
            "share_of_new_accounts": round(share_new_accounts, 4),
            "account_type_mix": type_counts,
            "accounts": comm_nodes,
        }

        # Step d: Flagging decision
        is_size_valid = (cfg.community_min_size <= size <= cfg.community_max_size)
        is_insular = (insularity >= cfg.insularity_min)
        is_flagged = is_size_valid and is_insular

        metrics["is_flagged"] = int(is_flagged)
        metrics["reason_not_flagged"] = "" if is_flagged else explain_not_flagged(metrics, cfg)
        community_records.append(metrics)

        # Emit Signal for flagged communities
        if is_flagged:
            start_ts = min(internal_timestamps).strftime("%Y-%m-%d %H:%M:%S") if internal_timestamps else str(group_ref_date)
            end_ts = max(internal_timestamps).strftime("%Y-%m-%d %H:%M:%S") if internal_timestamps else str(group_ref_date)

            signals.append(Signal(
                signal_id=f"sig_community_{signal_counter:04d}",
                pattern_type="dense_community",
                accounts=comm_nodes,
                txn_ids=sorted(list(set(internal_txn_ids))),
                start_ts=start_ts,
                end_ts=end_ts,
                total_amount=round(internal_amount, 2),
                details={
                    "size": size,
                    "insularity": round(insularity, 4),
                    "internal_amount": round(internal_amount, 2),
                    "external_amount": round(external_amount, 2),
                    "internal_edge_density": round(density, 4),
                    "reciprocity": round(reciprocity, 4),
                    "share_of_new_accounts": round(share_new_accounts, 4),
                    "account_type_mix": type_counts,
                },
            ))
            signal_counter += 1

    communities_df = pd.DataFrame(community_records)
    return communities_df, signals
