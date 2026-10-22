"""Command line interface for graph inspection and summary."""

import argparse
from pathlib import Path
import sys

from src.graph.builder import build_collapsed_graph, build_multigraph, load_data
from src.graph.stats import graph_summary, print_graph_summary


def main(argv=None) -> int:
    """Main CLI entrypoint for graph inspection."""
    parser = argparse.ArgumentParser(
        description="Build transaction graph from CSV datasets and print topological summary."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Path to directory containing accounts.csv and transactions.csv (default: data).",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    try:
        accounts_df, txns_df = load_data(data_dir)
    except Exception as exc:
        print(f"Error loading graph data from {data_dir.resolve()}: {exc}", file=sys.stderr)
        return 1

    # 1. Build MultiDiGraph (every transaction is an edge)
    multigraph = build_multigraph(accounts_df, txns_df)
    multi_summary = graph_summary(multigraph)
    print_graph_summary(multi_summary, title="TRANSACTION MULTIGRAPH SUMMARY")

    # 2. Build Collapsed DiGraph (aggregated account-to-account flows)
    collapsed = build_collapsed_graph(multigraph)
    collapsed_summary = graph_summary(collapsed)
    print_graph_summary(collapsed_summary, title="COLLAPSED AGGREGATED GRAPH SUMMARY")

    return 0


if __name__ == "__main__":
    sys.exit(main())
