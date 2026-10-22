"""Graph statistics and structural summary metrics for FraudGraph.

Computes macro network topology metrics (density, components, degree distributions)
and volume statistics to give investigators an overview of the network scale.
"""

from typing import Any, Dict, List, Tuple, Union
import networkx as nx


def graph_summary(G: Union[nx.MultiDiGraph, nx.DiGraph]) -> Dict[str, Any]:
    """Calculate key structural and financial metrics for the transaction graph.

    Args:
        G: NetworkX directed graph (MultiDiGraph or DiGraph).

    Returns:
        Dict with node/edge counts, density, components, top degree nodes, and total volume.
    """
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()

    # Density: ratio of actual edges to all possible directed edges N * (N - 1)
    density = nx.density(G)

    # Weakly connected components (subgraphs where every node is reachable ignoring direction)
    num_components = nx.number_weakly_connected_components(G)

    # Top 5 accounts by in-degree (aggregators / destinations)
    in_degrees: List[Tuple[str, int]] = sorted(
        G.in_degree(), key=lambda item: item[1], reverse=True
    )
    top_5_in_degree = in_degrees[:5]

    # Top 5 accounts by out-degree (dispersers / sources)
    out_degrees: List[Tuple[str, int]] = sorted(
        G.out_degree(), key=lambda item: item[1], reverse=True
    )
    top_5_out_degree = out_degrees[:5]

    # Total financial amount transferred
    total_amount = 0.0
    if isinstance(G, nx.MultiDiGraph):
        for _, _, data in G.edges(data=True):
            total_amount += data.get("amount", 0.0)
    else:
        for _, _, data in G.edges(data=True):
            total_amount += data.get("total_amount", data.get("amount", 0.0))

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": round(density, 6),
        "num_weakly_connected_components": num_components,
        "top_5_in_degree": top_5_in_degree,
        "top_5_out_degree": top_5_out_degree,
        "total_amount_moved": round(total_amount, 2),
    }


def print_graph_summary(summary: Dict[str, Any], title: str = "TRANSACTION GRAPH SUMMARY") -> None:
    """Format and print graph metrics in a clean, interview-ready summary."""
    print("\n" + "=" * 65)
    print(f"           {title}")
    print("=" * 65)
    print(f"Total Accounts (Nodes)      : {summary['node_count']:,}")
    print(f"Total Transfers (Edges)     : {summary['edge_count']:,}")
    print(f"Graph Density               : {summary['density']:.6f}")
    print(f"Weakly Connected Components : {summary['num_weakly_connected_components']:,}")
    print(f"Total Financial Volume      : ${summary['total_amount_moved']:,.2f}")
    print("-" * 65)
    print("TOP 5 ACCOUNTS BY IN-DEGREE (Most Inbound Senders):")
    for acc, deg in summary["top_5_in_degree"]:
        print(f"  • {acc:<14} : {deg:,} inbound transfers")
    print("-" * 65)
    print("TOP 5 ACCOUNTS BY OUT-DEGREE (Most Outbound Recipients):")
    for acc, deg in summary["top_5_out_degree"]:
        print(f"  • {acc:<14} : {deg:,} outbound transfers")
    print("=" * 65 + "\n")
