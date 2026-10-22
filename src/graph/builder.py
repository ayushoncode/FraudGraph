"""Graph builder module for FraudGraph.

Constructs NetworkX directed graphs from raw tabular account and transaction datasets.
Converts transactions into directed edges between accounts to enable graph algorithms.

NOTE: This module strictly operates on observable data (accounts.csv and transactions.csv).
It NEVER accesses ground_truth.csv.
"""

from pathlib import Path
from typing import Tuple, Union

import networkx as nx
import pandas as pd


def load_data(
    data_dir_or_path: Union[str, Path]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate accounts.csv and transactions.csv.

    Args:
        data_dir_or_path: Directory path containing 'accounts.csv' and 'transactions.csv'.

    Returns:
        Tuple of (accounts_df, transactions_df) with parsed datetimes.

    Raises:
        FileNotFoundError: If either required CSV file is missing.
        ValueError: If transactions reference unknown accounts not present in accounts.csv.
    """
    base_dir = Path(data_dir_or_path)
    accounts_file = base_dir / "accounts.csv"
    txns_file = base_dir / "transactions.csv"

    if not accounts_file.exists():
        raise FileNotFoundError(f"Accounts file not found at: {accounts_file.resolve()}")
    if not txns_file.exists():
        raise FileNotFoundError(f"Transactions file not found at: {txns_file.resolve()}")

    # 1. Load CSVs with pandas
    accounts_df = pd.read_csv(accounts_file)
    transactions_df = pd.read_csv(txns_file)

    # 2. Parse timestamps into pandas datetimes
    accounts_df["created_at"] = pd.to_datetime(accounts_df["created_at"])
    transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])

    # 3. Validate referential integrity: every src and dst must exist in accounts
    known_accounts = set(accounts_df["account_id"])
    src_accounts = set(transactions_df["src_account"])
    dst_accounts = set(transactions_df["dst_account"])

    unknown_srcs = src_accounts - known_accounts
    unknown_dsts = dst_accounts - known_accounts
    unknown_all = unknown_srcs.union(unknown_dsts)

    if unknown_all:
        sample_unknown = sorted(list(unknown_all))[:5]
        raise ValueError(
            f"Validation failed: transactions reference {len(unknown_all)} unknown account(s) "
            f"not found in accounts.csv. Sample unknown IDs: {sample_unknown}"
        )

    return accounts_df, transactions_df


def build_multigraph(
    accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
) -> nx.MultiDiGraph:
    """Build a directed multigraph where each transaction is an edge.

    Interview Explanation:
    In financial networks, accounts frequently transfer money multiple times to the same
    recipient at different timestamps and amounts. A MultiDiGraph preserves every individual
    transaction event and its temporal sequence.

    Args:
        accounts_df: DataFrame with columns [account_id, account_type, created_at, device_id, ip_address]
        transactions_df: DataFrame with columns [txn_id, src_account, dst_account, amount, timestamp]

    Returns:
        nx.MultiDiGraph: Graph with account nodes and individual transaction edges.
    """
    G = nx.MultiDiGraph()

    # 1. Add nodes with account attributes
    for _, row in accounts_df.iterrows():
        G.add_node(
            row["account_id"],
            account_type=row["account_type"],
            created_at=row["created_at"],
            device_id=str(row["device_id"]),
            ip_address=str(row["ip_address"]),
        )

    # 2. Add edges with transaction attributes
    for _, row in transactions_df.iterrows():
        G.add_edge(
            row["src_account"],
            row["dst_account"],
            key=row["txn_id"],
            txn_id=row["txn_id"],
            amount=float(row["amount"]),
            timestamp=row["timestamp"],
        )

    return G


def build_collapsed_graph(multigraph: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse multigraph into a simple DiGraph with aggregated edge metrics.

    Interview Explanation:
    While multigraphs preserve temporal resolution, many graph algorithms (such as PageRank,
    simple cycle detection, or community clustering) operate significantly faster and more
    naturally on simple weighted graphs. This function collapses multiple payments between
    the same pair (A -> B) into a single edge while keeping exact lists of txn_ids for evidence.

    Args:
        multigraph: The source nx.MultiDiGraph.

    Returns:
        nx.DiGraph: Simple directed graph with aggregated edge weights and lineage metadata.
    """
    collapsed = nx.DiGraph()

    # 1. Preserve all node attributes
    for node, attrs in multigraph.nodes(data=True):
        collapsed.add_node(node, **attrs)

    # 2. Group edges by ordered pair (u, v)
    pair_buckets = {}
    for u, v, _, data in multigraph.edges(keys=True, data=True):
        pair = (u, v)
        if pair not in pair_buckets:
            pair_buckets[pair] = {
                "amounts": [],
                "timestamps": [],
                "txn_ids": [],
            }
        pair_buckets[pair]["amounts"].append(data["amount"])
        pair_buckets[pair]["timestamps"].append(data["timestamp"])
        pair_buckets[pair]["txn_ids"].append(data["txn_id"])

    # 3. Add single collapsed edge with aggregated summary metrics
    for (u, v), data in pair_buckets.items():
        total_amount = round(sum(data["amounts"]), 2)
        txn_count = len(data["amounts"])
        first_ts = min(data["timestamps"])
        last_ts = max(data["timestamps"])

        collapsed.add_edge(
            u,
            v,
            total_amount=total_amount,
            txn_count=txn_count,
            first_ts=first_ts,
            last_ts=last_ts,
            txn_ids=data["txn_ids"],
        )

    return collapsed
