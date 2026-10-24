"""Command line interface for the FraudGraph detector pipeline."""

import argparse
import json
from pathlib import Path
import sys

from src.detectors.base import DetectorConfig
from src.detectors.runner import print_detector_summary, run_all_detectors
from src.graph.builder import build_collapsed_graph, build_multigraph, load_data


def main(argv=None) -> int:
    """CLI entrypoint for running fraud detectors."""
    parser = argparse.ArgumentParser(
        description="Run fraud pattern detectors on transaction graphs and export signals."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Path to directory containing accounts.csv and transactions.csv (default: data).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/signals.json",
        help="Output path for exported JSON signals (default: data/signals.json).",
    )

    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load graph without touching ground truth
        accounts_df, txns_df = load_data(data_dir)
    except Exception as exc:
        print(f"Error loading graph data: {exc}", file=sys.stderr)
        return 1

    # 1. Build graphs
    multigraph = build_multigraph(accounts_df, txns_df)
    collapsed = build_collapsed_graph(multigraph)

    # 2. Run detection pipeline
    config = DetectorConfig()
    signals, runtimes, diagnostics = run_all_detectors(multigraph, collapsed, accounts_df, config)

    # 3. Export to JSON
    signals_dict = [sig.to_dict() for sig in signals]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(signals_dict, f, indent=2)

    # 4. Print summary
    print_detector_summary(signals, runtimes, diagnostics, out_file)
    print(f"Successfully saved {len(signals)} signals to {out_file.resolve()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
