"""Command line interface and summary reporter for the synthetic network generator."""

import argparse
from pathlib import Path
import sys

from src.generator.config import GeneratorConfig
from src.generator.synthetic_network import SyntheticFinancialNetwork


def print_summary(
    generator: SyntheticFinancialNetwork,
    accounts_df,
    transactions_df,
    ground_truth_df,
    out_dir: Path,
) -> None:
    """Print an interview-ready, structured summary of generated network data."""
    total_accounts = len(accounts_df)
    account_type_counts = accounts_df["account_type"].value_counts().to_dict()
    total_txns = len(transactions_df)
    min_date = transactions_df["timestamp"].min()
    max_date = transactions_df["timestamp"].max()

    print("\n" + "=" * 65)
    print("           FRAUDGRAPH SYNTHETIC NETWORK SUMMARY")
    print("=" * 65)
    print(f"Random Seed      : {generator.config.seed}")
    print(f"Difficulty Level : {generator.config.difficulty.upper()}")
    print(f"Timeline         : {min_date} --> {max_date}")
    print(f"Output Directory : {out_dir.resolve()}")
    print("-" * 65)
    print(f"ACCOUNTS ({total_accounts:,} total):")
    for acc_type, count in account_type_counts.items():
        print(f"  • {acc_type.capitalize():<12}: {count:,}")

    print("-" * 65)
    print(f"TRANSACTIONS ({total_txns:,} total):")
    avg_amount = transactions_df["amount"].mean()
    total_volume = transactions_df["amount"].sum()
    print(f"  • Total Volume : ${total_volume:,.2f}")
    print(f"  • Average Size : ${avg_amount:,.2f}")

    print("-" * 65)
    print("INJECTED FRAUD RINGS (Known Ground Truth):")
    fraud_accounts = ground_truth_df[ground_truth_df["is_fraud"] == 1]
    print(f"  • Total Fraud Accounts : {len(fraud_accounts)} ({len(fraud_accounts)/total_accounts*100:.1f}%)")
    for p_type, count in generator.fraud_ring_counts.items():
        print(f"  • Pattern '{p_type:<22}': {count} rings")

    print("-" * 65)
    print("LEGITIMATE LOOKALIKE GROUPS (Must NOT Alert):")
    lookalike_accounts = ground_truth_df[ground_truth_df["lookalike_id"].notna()]
    print(f"  • Total Lookalike Accounts : {len(lookalike_accounts)}")
    for l_type, count in generator.lookalike_counts.items():
        print(f"  • Lookalike '{l_type:<20}': {count} groups/entities")

    print("=" * 65 + "\n")


def main(argv=None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial network datasets with ground truth."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Complexity of patterns: easy, medium, hard (default: medium).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data",
        help="Output directory path (default: data).",
    )
    parser.add_argument(
        "--target-transactions",
        type=int,
        default=10000,
        help="Approximate target number of transactions (default: 10000).",
    )

    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig(
        seed=args.seed,
        difficulty=args.difficulty,
        target_transactions=args.target_transactions,
    )

    generator = SyntheticFinancialNetwork(config)
    accounts_df, transactions_df, ground_truth_df = generator.generate()

    # Save to CSV files
    accounts_csv = out_path / "accounts.csv"
    transactions_csv = out_path / "transactions.csv"
    ground_truth_csv = out_path / "ground_truth.csv"

    accounts_df.to_csv(accounts_csv, index=False)
    transactions_df.to_csv(transactions_csv, index=False)
    ground_truth_df.to_csv(ground_truth_csv, index=False)

    print_summary(generator, accounts_df, transactions_df, ground_truth_df, out_path)
    print(f"Successfully saved datasets to {out_path.resolve()}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
