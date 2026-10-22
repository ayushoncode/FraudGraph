"""Graph inspection and query utilities for FraudGraph.

Provides helper routines to inspect accounts, traverse neighborhoods,
filter by time windows, and detect device-sharing clusters.
"""

from typing import Any, Dict, List, Set, Union
import networkx as nx
import pandas as pd


def get_account_info(
    G: Union[nx.MultiDiGraph, nx.DiGraph], account_id: str
) -> Dict[str, Any]:
    """Retrieve metadata and degree metrics for a specific account.

    Args:
        G: NetworkX graph.
        account_id: Unique account identifier.

    Returns:
        Dict containing account attributes, in-degree, and out-degree.

    Raises:
        KeyError: If account_id is not in the graph.
    """
    if account_id not in G:
        raise KeyError(f"Account '{account_id}' does not exist in graph.")

    node_data = dict(G.nodes[account_id])
    node_data["account_id"] = account_id
    node_data["in_degree"] = G.in_degree(account_id)
    node_data["out_degree"] = G.out_degree(account_id)
    return node_data


def get_transactions(
    G: Union[nx.MultiDiGraph, nx.DiGraph],
    account_id: str,
    direction: str = "both",
) -> List[Dict[str, Any]]:
    """Retrieve all transactions associated with an account, sorted chronologically.

    Args:
        G: MultiDiGraph or DiGraph.
        account_id: Account identifier.
        direction: Flow direction to query ('in', 'out', or 'both').

    Returns:
        List of transaction dictionaries sorted ascending by timestamp.

    Raises:
        KeyError: If account_id is not in the graph.
        ValueError: If direction is not one of ('in', 'out', 'both').
    """
    if account_id not in G:
        raise KeyError(f"Account '{account_id}' does not exist in graph.")

    valid_directions = ("in", "out", "both")
    if direction not in valid_directions:
        raise ValueError(f"Direction must be one of {valid_directions}, got '{direction}'.")

    results: List[Dict[str, Any]] = []

    # Case 1: nx.MultiDiGraph (preserves individual transactions)
    if isinstance(G, nx.MultiDiGraph):
        if direction in ("out", "both"):
            for u, v, _, data in G.out_edges(account_id, keys=True, data=True):
                results.append({
                    "txn_id": data.get("txn_id"),
                    "src_account": u,
                    "dst_account": v,
                    "amount": data.get("amount", 0.0),
                    "timestamp": data.get("timestamp"),
                    "direction": "outgoing",
                })
        if direction in ("in", "both"):
            for u, v, _, data in G.in_edges(account_id, keys=True, data=True):
                results.append({
                    "txn_id": data.get("txn_id"),
                    "src_account": u,
                    "dst_account": v,
                    "amount": data.get("amount", 0.0),
                    "timestamp": data.get("timestamp"),
                    "direction": "incoming",
                })

    # Case 2: nx.DiGraph (collapsed aggregated edges)
    else:
        if direction in ("out", "both"):
            for u, v, data in G.out_edges(account_id, data=True):
                results.append({
                    "src_account": u,
                    "dst_account": v,
                    "amount": data.get("total_amount", 0.0),
                    "timestamp": data.get("first_ts"),
                    "txn_count": data.get("txn_count", 1),
                    "txn_ids": data.get("txn_ids", []),
                    "direction": "outgoing",
                })
        if direction in ("in", "both"):
            for u, v, data in G.in_edges(account_id, data=True):
                results.append({
                    "src_account": u,
                    "dst_account": v,
                    "amount": data.get("total_amount", 0.0),
                    "timestamp": data.get("first_ts"),
                    "txn_count": data.get("txn_count", 1),
                    "txn_ids": data.get("txn_ids", []),
                    "direction": "incoming",
                })

    # Sort chronologically
    results.sort(key=lambda t: t["timestamp"])
    return results


def get_neighbors(
    G: Union[nx.MultiDiGraph, nx.DiGraph], account_id: str
) -> Dict[str, List[str]]:
    """Retrieve distinct inbound and outbound account neighbors.

    Args:
        G: NetworkX directed graph.
        account_id: Account identifier.

    Returns:
        Dict with keys 'in_neighbors' and 'out_neighbors', sorted alphabetically.

    Raises:
        KeyError: If account_id is not in the graph.
    """
    if account_id not in G:
        raise KeyError(f"Account '{account_id}' does not exist in graph.")

    return {
        "in_neighbors": sorted(list(set(G.predecessors(account_id)))),
        "out_neighbors": sorted(list(set(G.successors(account_id)))),
    }


def filter_by_time(
    G: Union[nx.MultiDiGraph, nx.DiGraph],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
) -> Union[nx.MultiDiGraph, nx.DiGraph]:
    """Create a new graph containing only transactions within the specified time range.

    Args:
        G: Source graph (MultiDiGraph or DiGraph).
        start: Start datetime (inclusive).
        end: End datetime (inclusive).

    Returns:
        New graph with identical nodes but filtered edges.
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    filtered = type(G)()

    # Copy all nodes and attributes
    for node, attrs in G.nodes(data=True):
        filtered.add_node(node, **attrs)

    if isinstance(G, nx.MultiDiGraph):
        for u, v, key, data in G.edges(keys=True, data=True):
            edge_ts = pd.to_datetime(data["timestamp"])
            if start_ts <= edge_ts <= end_ts:
                filtered.add_edge(u, v, key=key, **data)
    else:
        # Collapsed DiGraph: check if edge timeframe overlaps with window
        for u, v, data in G.edges(data=True):
            first_ts = pd.to_datetime(data.get("first_ts"))
            last_ts = pd.to_datetime(data.get("last_ts", first_ts))
            if first_ts <= end_ts and last_ts >= start_ts:
                filtered.add_edge(u, v, **data)

    return filtered


def get_subgraph(
    G: Union[nx.MultiDiGraph, nx.DiGraph],
    account_ids: Union[List[str], Set[str]],
) -> Union[nx.MultiDiGraph, nx.DiGraph]:
    """Extract the induced subgraph for a given set of accounts.

    Interview Explanation:
    Used when presenting evidence to investigators. Given an alerted ring of 5 accounts,
    we carve out the exact induced subgraph containing all flows among them.

    Args:
        G: Source graph.
        account_ids: Collection of account IDs to include.

    Returns:
        Isolated copy of the induced subgraph.
    """
    valid_accounts = [acc for acc in account_ids if acc in G]
    return G.subgraph(valid_accounts).copy()


def accounts_sharing_device(
    G: Union[nx.MultiDiGraph, nx.DiGraph]
) -> Dict[str, List[str]]:
    """Group accounts by shared device_id, returning only groups with >1 account.

    Interview Explanation:
    Financial fraud rings frequently operate multiple accounts from the same device,
    though family members legitimately share hardware too. This query clusters
    accounts by hardware footprint for downstream investigation.

    Args:
        G: NetworkX graph.

    Returns:
        Dict mapping device_id to list of account IDs (groups of 2 or more).
    """
    device_map: Dict[str, List[str]] = {}
    for node, attrs in G.nodes(data=True):
        dev_id = attrs.get("device_id")
        if dev_id:
            device_map.setdefault(str(dev_id), []).append(node)

    # Filter to only shared devices (> 1 account)
    return {
        dev: sorted(accounts)
        for dev, accounts in device_map.items()
        if len(accounts) > 1
    }
