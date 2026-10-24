"""Shared device and IP cluster detector (device_clusters.py).

Detects multi-account clusters sharing the same device_id or ip_address that actively
transfer money internally. Co-located accounts with no internal transactions (such as
families sharing a tablet) are explicitly filtered out to minimize false positives.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import pandas as pd

from src.detectors.base import DetectorConfig, Signal
from src.detectors.communities import explain_not_flagged
from src.graph.queries import accounts_sharing_device


def find_shared_device_clusters(
    multigraph: nx.MultiDiGraph,
    accounts_df: pd.DataFrame,
    config: DetectorConfig | None = None,
    prior_signals: Optional[List[Signal]] = None,
) -> Tuple[pd.DataFrame, List[Signal]]:
    """Identify suspicious account clusters sharing hardware (device_id) or IP footprint.

    Interview Explanation:
    Why require internal transactions?
    In consumer banking, family members frequently share a home tablet or PC (legitimate lookalike).
    A shared device is only alarming when the co-located accounts are actively funneling funds
    among themselves. Requiring >= device_min_internal_txns avoids persecuting innocent families.

    Args:
        multigraph: Directed multigraph with transaction timestamps.
        accounts_df: DataFrame with account attributes (device_id, ip_address, created_at).
        config: Detector threshold configuration.
        prior_signals: Optional list of previously emitted signals to check for overlap.

    Returns:
        Tuple of:
          - pd.DataFrame: Tabular log of ALL co-located hardware/IP groups with metrics.
          - List[Signal]: Alerts for clusters passing the internal activity threshold.
    """
    cfg = config or DetectorConfig()
    signals: List[Signal] = []
    cluster_records: List[Dict[str, Any]] = []
    signal_counter = 1
    prior_signals = prior_signals or []

    # Map accounts for quick attribute lookups
    account_lookup = accounts_df.set_index("account_id").to_dict(orient="index")

    # Step a: Group accounts by device_id and ip_address
    device_groups = accounts_sharing_device(multigraph)

    # Compute IP sharing groups (> 1 account)
    ip_map: Dict[str, List[str]] = {}
    for node, attrs in multigraph.nodes(data=True):
        ip = attrs.get("ip_address")
        if ip:
            ip_map.setdefault(str(ip), []).append(node)
    ip_groups = {ip: sorted(accs) for ip, accs in ip_map.items() if len(accs) > 1}

    # Combine into unified candidate clusters
    all_clusters: List[Dict[str, Any]] = []
    for dev_id, members in device_groups.items():
        all_clusters.append({
            "cluster_type": "device",
            "cluster_identifier": dev_id,
            "members": members,
        })
    for ip_addr, members in ip_groups.items():
        all_clusters.append({
            "cluster_type": "ip",
            "cluster_identifier": ip_addr,
            "members": members,
        })

    # Step b: Evaluate each group
    for c_idx, cluster in enumerate(all_clusters, start=1):
        c_type = cluster["cluster_type"]
        c_id = cluster["cluster_identifier"]
        members = cluster["members"]
        member_set = set(members)
        size = len(members)

        # 1. Count internal transactions and volume among members
        internal_txns: List[Dict[str, Any]] = []
        for u in members:
            for _, v, key, data in multigraph.out_edges(u, keys=True, data=True):
                if v in member_set:
                    internal_txns.append({
                        "txn_id": data["txn_id"],
                        "src": u,
                        "dst": v,
                        "amount": float(data["amount"]),
                        "timestamp": pd.to_datetime(data["timestamp"]),
                    })

        internal_txn_count = len(internal_txns)
        internal_amount = round(sum(t["amount"] for t in internal_txns), 2)

        # 2. Account type mix and creation dates
        type_counts = {"individual": 0, "merchant": 0, "business": 0}
        created_dates = []
        for n in members:
            meta = account_lookup.get(n, {})
            acc_type = meta.get("account_type", "individual")
            type_counts[acc_type] = type_counts.get(acc_type, 0) + 1
            if "created_at" in meta and pd.notna(meta["created_at"]):
                created_dates.append(pd.to_datetime(meta["created_at"]))

        # Share of new accounts
        ref_date = min(t["timestamp"] for t in internal_txns) if internal_txns else (
            min(created_dates) if created_dates else pd.to_datetime("2025-01-01")
        )
        cutoff = ref_date - timedelta(days=cfg.new_account_days)
        new_count = sum(1 for d in created_dates if d >= cutoff)
        share_new = new_count / size if size > 0 else 0.0

        # 3. Check overlap with previously flagged signals (cycles, mules)
        overlapping_signals = []
        for ps in prior_signals:
            if member_set.intersection(set(ps.accounts)):
                overlapping_signals.append(ps.signal_id)

        # Build metric dictionary
        metrics = {
            "cluster_index": f"cluster_{c_idx:03d}",
            "cluster_type": c_type,
            "cluster_identifier": c_id,
            "member_count": size,
            "internal_txn_count": internal_txn_count,
            "internal_amount": internal_amount,
            "share_of_new_accounts": round(share_new, 4),
            "account_type_mix": type_counts,
            "overlapping_signal_ids": overlapping_signals,
            "members": members,
        }

        # Step c: Flagging decision
        is_size_valid = (size >= cfg.shared_device_min_accounts)
        has_internal_traffic = (internal_txn_count >= cfg.device_min_internal_txns)
        is_flagged = is_size_valid and has_internal_traffic

        metrics["is_flagged"] = int(is_flagged)
        metrics["reason_not_flagged"] = "" if is_flagged else explain_not_flagged(metrics, cfg)
        cluster_records.append(metrics)

        # Emit Signal for flagged clusters
        if is_flagged:
            internal_txns.sort(key=lambda t: t["timestamp"])
            start_ts = str(internal_txns[0]["timestamp"])
            end_ts = str(internal_txns[-1]["timestamp"])

            signals.append(Signal(
                signal_id=f"sig_device_{signal_counter:04d}",
                pattern_type="shared_device_cluster",
                accounts=members,
                txn_ids=[t["txn_id"] for t in internal_txns],
                start_ts=start_ts,
                end_ts=end_ts,
                total_amount=internal_amount,
                details={
                    "cluster_type": c_type,
                    "cluster_identifier": c_id,
                    "member_count": size,
                    "internal_txn_count": internal_txn_count,
                    "internal_amount": internal_amount,
                    "share_of_new_accounts": round(share_new, 4),
                    "account_type_mix": type_counts,
                    "overlapping_signal_ids": overlapping_signals,
                },
            ))
            signal_counter += 1

    clusters_df = pd.DataFrame(cluster_records)
    return clusters_df, signals
