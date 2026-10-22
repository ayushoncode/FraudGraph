"""Tests for the synthetic financial network generator."""

import pandas as pd
import pytest

from src.generator.config import GeneratorConfig
from src.generator.synthetic_network import SyntheticFinancialNetwork


def test_generator_deterministic_reproducibility():
    """Verify that using the same random seed produces bit-for-bit identical outputs."""
    config1 = GeneratorConfig(seed=42, difficulty="medium", target_transactions=3000)
    config2 = GeneratorConfig(seed=42, difficulty="medium", target_transactions=3000)

    gen1 = SyntheticFinancialNetwork(config1)
    acc1, txn1, gt1 = gen1.generate()

    gen2 = SyntheticFinancialNetwork(config2)
    acc2, txn2, gt2 = gen2.generate()

    pd.testing.assert_frame_equal(acc1, acc2)
    pd.testing.assert_frame_equal(txn1, txn2)
    pd.testing.assert_frame_equal(gt1, gt2)


def test_every_ring_in_ground_truth_exists_in_transactions():
    """Verify every fraud ring registered in ground_truth has real transactions."""
    config = GeneratorConfig(seed=123, difficulty="medium", target_transactions=2000)
    gen = SyntheticFinancialNetwork(config)
    _, txns, ground_truth = gen.generate()

    fraud_rings = ground_truth[ground_truth["is_fraud"] == 1]["ring_id"].dropna().unique()
    assert len(fraud_rings) > 0, "Expected at least one fraud ring to be injected"

    all_participating_accounts = set(txns["src_account"]).union(set(txns["dst_account"]))

    for ring_id in fraud_rings:
        ring_accounts = ground_truth[ground_truth["ring_id"] == ring_id]["account_id"].tolist()
        assert len(ring_accounts) >= 3, f"Ring {ring_id} should have at least 3 participants"

        # Every account in the ring must be active in transactions
        for acc in ring_accounts:
            assert acc in all_participating_accounts, (
                f"Account {acc} in ring {ring_id} has no transactions in the dataset"
            )


def test_no_fraud_ring_labeled_as_lookalike():
    """Verify strict mutual exclusivity: no fraud account is marked as a legitimate lookalike."""
    config = GeneratorConfig(seed=999, difficulty="medium", target_transactions=2000)
    gen = SyntheticFinancialNetwork(config)
    _, _, ground_truth = gen.generate()

    # An account cannot have is_fraud == 1 and also have a lookalike_id
    conflicting = ground_truth[
        (ground_truth["is_fraud"] == 1) & (ground_truth["lookalike_id"].notna())
    ]
    assert len(conflicting) == 0, (
        f"Found {len(conflicting)} accounts with conflicting fraud & lookalike labels: {conflicting}"
    )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generator_all_difficulties(difficulty: str):
    """Verify generation works cleanly across easy, medium, and hard modes."""
    config = GeneratorConfig(seed=42, difficulty=difficulty, target_transactions=2500)
    gen = SyntheticFinancialNetwork(config)
    accounts, txns, ground_truth = gen.generate()

    # Column integrity checks
    expected_acc_cols = {"account_id", "account_type", "created_at", "device_id", "ip_address"}
    expected_txn_cols = {"txn_id", "src_account", "dst_account", "amount", "timestamp"}
    expected_gt_cols = {"account_id", "is_fraud", "ring_id", "pattern_type", "lookalike_id", "lookalike_type"}

    assert expected_acc_cols.issubset(set(accounts.columns))
    assert expected_txn_cols.issubset(set(txns.columns))
    assert expected_gt_cols.issubset(set(ground_truth.columns))

    # Amounts must be positive
    assert (txns["amount"] > 0).all()

    # Timestamps must be sorted chronologically
    timestamps = pd.to_datetime(txns["timestamp"])
    assert timestamps.is_monotonic_increasing


def test_shared_devices_legitimate_vs_fraud():
    """Verify that both legitimate families and fraud clusters share devices, but are labeled differently."""
    config = GeneratorConfig(seed=777, difficulty="medium", target_transactions=2000)
    gen = SyntheticFinancialNetwork(config)
    accounts, _, ground_truth = gen.generate()

    merged = accounts.merge(ground_truth, on="account_id")

    # Family members sharing device
    families = merged[merged["lookalike_type"] == "family_shared_device"]
    assert len(families) > 0
    # Confirm family accounts actually share devices
    family_dev_counts = families.groupby("device_id")["account_id"].count()
    assert (family_dev_counts > 1).any(), "Family lookalikes must share a device_id"

    # Fraud cluster sharing device
    fraud_clusters = merged[merged["pattern_type"] == "shared_device_cluster"]
    assert len(fraud_clusters) > 0
    cluster_dev_counts = fraud_clusters.groupby("device_id")["account_id"].count()
    assert (cluster_dev_counts > 1).any(), "Fraud cluster must share a device_id"
