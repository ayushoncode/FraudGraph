"""Unit tests for FraudGraph pattern detectors (cycles, mules, fan, structuring, communities, devices)."""

from datetime import datetime, timedelta
import networkx as nx
import pandas as pd
import pytest

from src.detectors.base import DetectorConfig
from src.detectors.communities import (
    explain_not_flagged,
    find_suspicious_communities,
)
from src.detectors.cycles import find_time_bounded_cycles
from src.detectors.device_clusters import find_shared_device_clusters
from src.detectors.fan import compute_fan_features
from src.detectors.passthrough import find_pass_through
from src.detectors.runner import run_all_detectors
from src.graph.builder import build_collapsed_graph, build_multigraph


def _create_toy_graph(accounts_list, txns_list):
    """Helper to quickly assemble MultiDiGraph, DiGraph, and accounts_df for testing."""
    accounts_df = pd.DataFrame(accounts_list)
    txns_df = pd.DataFrame(txns_list)
    accounts_df["created_at"] = pd.to_datetime(accounts_df["created_at"])
    txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])

    mg = build_multigraph(accounts_df, txns_df)
    cg = build_collapsed_graph(mg)
    return mg, cg, accounts_df


# ---------------------------------------------------------------------------
# CYCLE DETECTOR TESTS
# ---------------------------------------------------------------------------


def test_cycle_found_inside_48h():
    """Test 1: A -> B -> C -> A with increasing timestamps inside 48h -> cycle found."""
    accounts = [
        {"account_id": "A", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d1", "ip_address": "1.1.1.1"},
        {"account_id": "B", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d2", "ip_address": "1.1.1.2"},
        {"account_id": "C", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d3", "ip_address": "1.1.1.3"},
    ]
    txns = [
        {"txn_id": "tx1", "src_account": "A", "dst_account": "B", "amount": 10000.0, "timestamp": "2025-01-01 10:00:00"},
        {"txn_id": "tx2", "src_account": "B", "dst_account": "C", "amount": 9800.0,  "timestamp": "2025-01-01 18:00:00"},
        {"txn_id": "tx3", "src_account": "C", "dst_account": "A", "amount": 9600.0,  "timestamp": "2025-01-02 12:00:00"},
    ]
    mg, cg, _ = _create_toy_graph(accounts, txns)
    signals = find_time_bounded_cycles(mg, cg)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.pattern_type == "cycle"
    assert set(sig.accounts) == {"A", "B", "C"}
    assert sig.txn_ids == ["tx1", "tx2", "tx3"]
    assert sig.details["length"] == 3
    assert sig.details["span_hours"] == 26.0


def test_cycle_spread_over_10_days_not_found():
    """Test 2: The same cycle spread over 10 days -> NOT found (exceeds 48h window)."""
    accounts = [
        {"account_id": "A", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d1", "ip_address": "1.1.1.1"},
        {"account_id": "B", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d2", "ip_address": "1.1.1.2"},
        {"account_id": "C", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d3", "ip_address": "1.1.1.3"},
    ]
    txns = [
        {"txn_id": "tx1", "src_account": "A", "dst_account": "B", "amount": 10000.0, "timestamp": "2025-01-01 10:00:00"},
        {"txn_id": "tx2", "src_account": "B", "dst_account": "C", "amount": 9800.0,  "timestamp": "2025-01-05 10:00:00"},
        {"txn_id": "tx3", "src_account": "C", "dst_account": "A", "amount": 9600.0,  "timestamp": "2025-01-11 10:00:00"},
    ]
    mg, cg, _ = _create_toy_graph(accounts, txns)
    signals = find_time_bounded_cycles(mg, cg)

    assert len(signals) == 0


def test_cycle_closing_payment_before_first_not_found():
    """Test 3: The same edges but the closing payment happened BEFORE the first -> NOT found."""
    accounts = [
        {"account_id": "A", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d1", "ip_address": "1.1.1.1"},
        {"account_id": "B", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d2", "ip_address": "1.1.1.2"},
        {"account_id": "C", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d3", "ip_address": "1.1.1.3"},
    ]
    txns = [
        {"txn_id": "tx1", "src_account": "A", "dst_account": "B", "amount": 10000.0, "timestamp": "2025-01-10 10:00:00"},
        {"txn_id": "tx2", "src_account": "B", "dst_account": "C", "amount": 9800.0,  "timestamp": "2025-01-10 14:00:00"},
        {"txn_id": "tx3", "src_account": "C", "dst_account": "A", "amount": 9600.0,  "timestamp": "2025-01-01 09:00:00"},
    ]
    mg, cg, _ = _create_toy_graph(accounts, txns)
    signals = find_time_bounded_cycles(mg, cg)

    assert len(signals) == 0


def test_two_account_back_and_forth_not_found():
    """Test 4: A 2-account back-and-forth -> NOT found (min_len=3)."""
    accounts = [
        {"account_id": "A", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d1", "ip_address": "1.1.1.1"},
        {"account_id": "B", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d2", "ip_address": "1.1.1.2"},
    ]
    txns = [
        {"txn_id": "tx1", "src_account": "A", "dst_account": "B", "amount": 5000.0, "timestamp": "2025-01-01 10:00:00"},
        {"txn_id": "tx2", "src_account": "B", "dst_account": "A", "amount": 4900.0, "timestamp": "2025-01-01 12:00:00"},
    ]
    mg, cg, _ = _create_toy_graph(accounts, txns)
    signals = find_time_bounded_cycles(mg, cg)

    assert len(signals) == 0


def test_cycle_rotation_reported_only_once():
    """Test 5: A rotation of the same cycle is reported only once."""
    accounts = [
        {"account_id": "A", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d1", "ip_address": "1.1.1.1"},
        {"account_id": "B", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d2", "ip_address": "1.1.1.2"},
        {"account_id": "C", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d3", "ip_address": "1.1.1.3"},
    ]
    txns = [
        {"txn_id": "tx1", "src_account": "A", "dst_account": "B", "amount": 10000.0, "timestamp": "2025-01-01 10:00:00"},
        {"txn_id": "tx2", "src_account": "B", "dst_account": "C", "amount": 9900.0,  "timestamp": "2025-01-01 12:00:00"},
        {"txn_id": "tx3", "src_account": "C", "dst_account": "A", "amount": 9800.0,  "timestamp": "2025-01-01 14:00:00"},
    ]
    mg, cg, _ = _create_toy_graph(accounts, txns)
    signals = find_time_bounded_cycles(mg, cg)

    assert len(signals) == 1


# ---------------------------------------------------------------------------
# MULE & FAN TESTS
# ---------------------------------------------------------------------------


def test_mule_account_pass_through_found():
    """Test 6: A mule account (in 100,000, out 95,000 after 5 min) -> pass-through found."""
    accounts = [
        {"account_id": "SRC",  "account_type": "individual", "created_at": "2025-01-01", "device_id": "d1", "ip_address": "1.1.1.1"},
        {"account_id": "MULE", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d2", "ip_address": "1.1.1.2"},
        {"account_id": "DST",  "account_type": "individual", "created_at": "2025-01-01", "device_id": "d3", "ip_address": "1.1.1.3"},
    ]
    txns = [
        {"txn_id": "in_1",  "src_account": "SRC",  "dst_account": "MULE", "amount": 100000.0, "timestamp": "2025-01-01 10:00:00"},
        {"txn_id": "out_1", "src_account": "MULE", "dst_account": "DST",  "amount": 95000.0,  "timestamp": "2025-01-01 10:05:00"},
    ]
    mg, _, _ = _create_toy_graph(accounts, txns)
    signals = find_pass_through(mg)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.pattern_type == "pass_through"
    assert sig.accounts == ["MULE"]
    assert sig.txn_ids == ["in_1", "out_1"]
    assert sig.details["average_pass_through_ratio"] == 0.95
    assert sig.details["median_delay_minutes"] == 5.0


def test_payroll_business_not_pass_through():
    """Test 7: A payroll business (out to 50 people, no inflow match) -> NOT pass-through."""
    accounts = [{"account_id": "BIZ", "account_type": "business", "created_at": "2024-01-01", "device_id": "corp1", "ip_address": "1.1.1.1"}]
    for i in range(1, 51):
        accounts.append({
            "account_id": f"EMP_{i}", "account_type": "individual", "created_at": "2024-01-01", "device_id": f"dev_{i}", "ip_address": f"10.0.0.{i}"
        })

    txns = []
    for i in range(1, 51):
        txns.append({
            "txn_id": f"pay_{i}",
            "src_account": "BIZ",
            "dst_account": f"EMP_{i}",
            "amount": 3500.0,
            "timestamp": "2025-01-01 09:00:00",
        })

    mg, _, _ = _create_toy_graph(accounts, txns)
    signals = find_pass_through(mg)

    assert len(signals) == 0


def test_merchant_high_fan_in_no_fan_in_fan_out_signal():
    """Test 8: A merchant with 200 distinct payers -> high fan-in feature but no fan_in_fan_out signal."""
    accounts = [{"account_id": "MERCHANT", "account_type": "merchant", "created_at": "2024-01-01", "device_id": "pos1", "ip_address": "1.1.1.1"}]
    for i in range(1, 201):
        accounts.append({
            "account_id": f"CUST_{i}", "account_type": "individual", "created_at": "2024-01-01", "device_id": f"dev_{i}", "ip_address": f"10.0.0.{i}"
        })

    txns = []
    for i in range(1, 201):
        txns.append({
            "txn_id": f"buy_{i}",
            "src_account": f"CUST_{i}",
            "dst_account": "MERCHANT",
            "amount": 45.0,
            "timestamp": "2025-01-01 12:00:00",
        })

    mg, _, _ = _create_toy_graph(accounts, txns)
    features_df, signals = compute_fan_features(mg)

    merchant_features = features_df[features_df["account_id"] == "MERCHANT"]
    assert len(merchant_features) > 0
    assert merchant_features["fan_in"].max() == 200
    assert merchant_features["fan_out"].max() == 0
    assert len(signals) == 0


def test_pipeline_determinism_identical_signals_twice():
    """Test 9: The same input gives identical signals twice (strict determinism)."""
    accounts = [
        {"account_id": "A", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d1", "ip_address": "1.1.1.1"},
        {"account_id": "B", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d2", "ip_address": "1.1.1.2"},
        {"account_id": "C", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d3", "ip_address": "1.1.1.3"},
    ]
    txns = [
        {"txn_id": "tx1", "src_account": "A", "dst_account": "B", "amount": 10000.0, "timestamp": "2025-01-01 10:00:00"},
        {"txn_id": "tx2", "src_account": "B", "dst_account": "C", "amount": 9800.0,  "timestamp": "2025-01-01 18:00:00"},
        {"txn_id": "tx3", "src_account": "C", "dst_account": "A", "amount": 9600.0,  "timestamp": "2025-01-02 12:00:00"},
    ]
    mg1, cg1, acc1 = _create_toy_graph(accounts, txns)
    mg2, cg2, acc2 = _create_toy_graph(accounts, txns)

    signals1, _, _ = run_all_detectors(mg1, cg1, acc1)
    signals2, _, _ = run_all_detectors(mg2, cg2, acc2)

    assert len(signals1) == len(signals2)
    for s1, s2 in zip(signals1, signals2):
        assert s1.to_dict() == s2.to_dict()


# ---------------------------------------------------------------------------
# COMMUNITY & LOUVAIN TESTS
# ---------------------------------------------------------------------------


def test_louvain_separates_two_dense_groups():
    """Test 10: Two dense groups joined by one weak edge -> Louvain separates them cleanly."""
    accounts = []
    # Group 1: A1, A2, A3, A4
    for i in range(1, 5):
        accounts.append({"account_id": f"A{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": f"dev_A{i}", "ip_address": "1.1.1.1"})
    # Group 2: B1, B2, B3, B4
    for i in range(1, 5):
        accounts.append({"account_id": f"B{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": f"dev_B{i}", "ip_address": "2.2.2.2"})

    txns = []
    t_idx = 1
    # Dense internal edges inside A
    for i in range(1, 5):
        for j in range(1, 5):
            if i != j:
                txns.append({"txn_id": f"t_{t_idx}", "src_account": f"A{i}", "dst_account": f"A{j}", "amount": 5000.0, "timestamp": "2025-01-02 10:00:00"})
                t_idx += 1

    # Dense internal edges inside B
    for i in range(1, 5):
        for j in range(1, 5):
            if i != j:
                txns.append({"txn_id": f"t_{t_idx}", "src_account": f"B{i}", "dst_account": f"B{j}", "amount": 5000.0, "timestamp": "2025-01-02 10:00:00"})
                t_idx += 1

    # Weak bridge edge: A4 -> B1
    txns.append({"txn_id": f"t_{t_idx}", "src_account": "A4", "dst_account": "B1", "amount": 10.0, "timestamp": "2025-01-02 12:00:00"})

    _, cg, acc_df = _create_toy_graph(accounts, txns)
    communities_df, signals = find_suspicious_communities(cg, acc_df)

    assert len(communities_df) >= 2
    a_comms = [c for c in communities_df["accounts"] if "A1" in c]
    b_comms = [c for c in communities_df["accounts"] if "B1" in c]
    assert len(a_comms) == 1 and len(b_comms) == 1
    assert "B1" not in a_comms[0]


def test_community_determinism():
    """Test 11: Same seed gives identical communities twice."""
    accounts = [{"account_id": f"acc_{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": f"d_{i}", "ip_address": "1.1.1.1"} for i in range(6)]
    txns = [
        {"txn_id": f"t_{i}", "src_account": f"acc_{i}", "dst_account": f"acc_{(i+1)%6}", "amount": 1000.0, "timestamp": "2025-01-02 10:00:00"}
        for i in range(6)
    ]
    _, cg, acc_df = _create_toy_graph(accounts, txns)

    cfg = DetectorConfig(louvain_seed=42)
    df1, sigs1 = find_suspicious_communities(cg, acc_df, cfg)
    df2, sigs2 = find_suspicious_communities(cg, acc_df, cfg)

    assert len(df1) == len(df2)
    assert len(sigs1) == len(sigs2)
    assert df1["accounts"].tolist() == df2["accounts"].tolist()


def test_community_insularity_0_9_flagged():
    """Test 12: A community with insularity 0.9 and size 5 -> flagged."""
    accounts = [{"account_id": f"acc_{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": f"d_{i}", "ip_address": "1.1.1.1"} for i in range(5)]
    # External cluster
    for i in range(5):
        accounts.append({"account_id": f"out_{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": f"d_out_{i}", "ip_address": "2.2.2.2"})

    txns = []
    t_idx = 1
    # Dense internal volume for community A: 10 directed pairs of $9,000 = $90,000
    for i in range(5):
        for j in range(5):
            if i != j:
                txns.append({"txn_id": f"t_{t_idx}", "src_account": f"acc_{i}", "dst_account": f"acc_{j}", "amount": 4500.0, "timestamp": "2025-01-02 10:00:00"})
                t_idx += 1

    # Dense external community B to keep out_i together
    for i in range(5):
        for j in range(5):
            if i != j:
                txns.append({"txn_id": f"t_{t_idx}", "src_account": f"out_{i}", "dst_account": f"out_{j}", "amount": 10000.0, "timestamp": "2025-01-02 10:00:00"})
                t_idx += 1

    # External boundary transfers from A to B: 5 edges of $2,000 = $10,000
    for i in range(5):
        txns.append({"txn_id": f"t_{t_idx}", "src_account": f"acc_{i}", "dst_account": f"out_{i}", "amount": 2000.0, "timestamp": "2025-01-02 11:00:00"})
        t_idx += 1

    # Insularity = 90,000 / (90,000 + 10,000) = 0.90
    _, cg, acc_df = _create_toy_graph(accounts, txns)
    communities_df, signals = find_suspicious_communities(cg, acc_df)

    flagged = [s for s in signals if set(s.accounts) == {f"acc_{i}" for i in range(5)}]
    assert len(flagged) == 1
    sig = flagged[0]
    assert sig.details["insularity"] == 0.90
    assert sig.details["size"] == 5


def test_community_insularity_0_3_not_flagged():
    """Test 13: A community with insularity 0.3 -> NOT flagged, with a reason string."""
    accounts = [{"account_id": f"acc_{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": f"d_{i}", "ip_address": "1.1.1.1"} for i in range(5)]
    for i in range(5):
        accounts.append({"account_id": f"out_{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": f"d_out_{i}", "ip_address": "2.2.2.2"})

    txns = []
    t_idx = 1
    # Internal volume: 10 pairs of $3,000 = $30,000
    for i in range(5):
        for j in range(5):
            if i != j:
                txns.append({"txn_id": f"t_{t_idx}", "src_account": f"acc_{i}", "dst_account": f"acc_{j}", "amount": 1500.0, "timestamp": "2025-01-02 10:00:00"})
                t_idx += 1

    # Dense outside community to hold out_i nodes together
    for i in range(5):
        for j in range(5):
            if i != j:
                txns.append({"txn_id": f"t_{t_idx}", "src_account": f"out_{i}", "dst_account": f"out_{j}", "amount": 50000.0, "timestamp": "2025-01-02 10:00:00"})
                t_idx += 1

    # External boundary volume: 5 edges of $14,000 = $70,000
    for i in range(5):
        txns.append({"txn_id": f"t_{t_idx}", "src_account": f"acc_{i}", "dst_account": f"out_{i}", "amount": 14000.0, "timestamp": "2025-01-02 11:00:00"})
        t_idx += 1

    # Insularity = 30,000 / (30,000 + 70,000) = 0.30
    _, cg, acc_df = _create_toy_graph(accounts, txns)
    communities_df, signals = find_suspicious_communities(cg, acc_df)

    # Check reason_not_flagged function directly and through DataFrame
    reason = explain_not_flagged({"size": 5, "insularity": 0.30}, DetectorConfig())
    assert "insularity 0.30 below threshold 0.70" in reason

    target_comm = communities_df[communities_df["accounts"].apply(lambda accs: set(accs) == {f"acc_{i}" for i in range(5)})]
    if len(target_comm) > 0:
        row = target_comm.iloc[0]
        assert row["is_flagged"] == 0
        assert "insularity" in row["reason_not_flagged"]


def test_oversized_community_size_100_not_flagged():
    """Test 14: A community of size 100 -> not flagged as suspicious, listed as oversized."""
    accounts = [{"account_id": f"acc_{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": f"d_{i}", "ip_address": "1.1.1.1"} for i in range(100)]
    # Star graph topology keeps all 100 nodes in exactly 1 community in Louvain
    txns = [
        {"txn_id": f"t_{i}", "src_account": "acc_0", "dst_account": f"acc_{i}", "amount": 1000.0, "timestamp": "2025-01-02 10:00:00"}
        for i in range(1, 100)
    ]
    _, cg, acc_df = _create_toy_graph(accounts, txns)

    cfg = DetectorConfig(community_max_size=30)
    communities_df, signals = find_suspicious_communities(cg, acc_df, cfg)

    assert len(signals) == 0
    oversized = communities_df[communities_df["size"] == 100]
    assert len(oversized) == 1
    assert "oversized" in oversized.iloc[0]["reason_not_flagged"]


# ---------------------------------------------------------------------------
# SHARED HARDWARE & DEVICE CLUSTER TESTS
# ---------------------------------------------------------------------------


def test_shared_device_zero_internal_txns_not_flagged():
    """Test 15: 4 accounts on one device, no transactions among them -> NOT flagged."""
    accounts = [{"account_id": f"acc_{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": "shared_pad", "ip_address": "1.1.1.1"} for i in range(4)]
    accounts.append({"account_id": "STORE", "account_type": "merchant", "created_at": "2024-01-01", "device_id": "pos_1", "ip_address": "2.2.2.2"})

    txns = [
        {"txn_id": f"buy_{i}", "src_account": f"acc_{i}", "dst_account": "STORE", "amount": 50.0, "timestamp": "2025-01-02 10:00:00"}
        for i in range(4)
    ]
    mg, _, acc_df = _create_toy_graph(accounts, txns)
    clusters_df, signals = find_shared_device_clusters(mg, acc_df)

    assert len(signals) == 0
    dev_cluster = clusters_df[clusters_df["cluster_identifier"] == "shared_pad"]
    assert len(dev_cluster) == 1
    assert dev_cluster.iloc[0]["is_flagged"] == 0
    assert "0 internal transactions" in dev_cluster.iloc[0]["reason_not_flagged"]


def test_shared_device_new_accounts_with_internal_txns_flagged():
    """Test 16: 4 new accounts on one device with 6 internal transactions -> flagged."""
    accounts = [{"account_id": f"bot_{i}", "account_type": "individual", "created_at": "2025-01-01", "device_id": "fraud_device", "ip_address": "100.1.1.1"} for i in range(4)]
    txns = [
        {"txn_id": f"bot_tx_{i}", "src_account": f"bot_{i}", "dst_account": f"bot_{(i+1)%4}", "amount": 5000.0, "timestamp": "2025-01-02 10:00:00"}
        for i in range(4)
    ]
    txns.append({"txn_id": "bot_tx_4", "src_account": "bot_0", "dst_account": "bot_2", "amount": 3000.0, "timestamp": "2025-01-02 11:00:00"})
    txns.append({"txn_id": "bot_tx_5", "src_account": "bot_1", "dst_account": "bot_3", "amount": 3000.0, "timestamp": "2025-01-02 12:00:00"})

    mg, _, acc_df = _create_toy_graph(accounts, txns)
    clusters_df, signals = find_shared_device_clusters(mg, acc_df)

    flagged = [s for s in signals if s.details.get("cluster_identifier") == "fraud_device"]
    assert len(flagged) == 1
    sig = flagged[0]
    assert sig.pattern_type == "shared_device_cluster"
    assert sig.details["internal_txn_count"] == 6
    assert sig.details["member_count"] == 4


def test_family_shared_device_one_internal_txn_not_flagged():
    """Test 17: Family of 3 sharing a device, old accounts, only 1 internal transaction -> NOT flagged."""
    accounts = [{"account_id": f"fam_{i}", "account_type": "individual", "created_at": "2024-01-01", "device_id": "family_tablet", "ip_address": "192.168.1.1"} for i in range(3)]
    txns = [
        {"txn_id": "allowance_1", "src_account": "fam_0", "dst_account": "fam_1", "amount": 40.0, "timestamp": "2025-01-02 10:00:00"}
    ]

    mg, _, acc_df = _create_toy_graph(accounts, txns)
    clusters_df, signals = find_shared_device_clusters(mg, acc_df)

    assert len(signals) == 0
    fam_row = clusters_df[clusters_df["cluster_identifier"] == "family_tablet"].iloc[0]
    assert fam_row["is_flagged"] == 0
    assert "below threshold 3" in fam_row["reason_not_flagged"]


def test_merchant_many_customers_never_forms_insular_community():
    """Test 18: A merchant with many customers -> never forms a flagged insular community."""
    accounts = [{"account_id": "SHOP", "account_type": "merchant", "created_at": "2024-01-01", "device_id": "pos_main", "ip_address": "50.1.1.1"}]
    for i in range(1, 31):
        accounts.append({"account_id": f"BUYER_{i}", "account_type": "individual", "created_at": "2024-01-01", "device_id": f"phone_{i}", "ip_address": f"10.0.0.{i}"})

    txns = [
        {"txn_id": f"sale_{i}", "src_account": f"BUYER_{i}", "dst_account": "SHOP", "amount": 25.0, "timestamp": "2025-01-02 12:00:00"}
        for i in range(1, 31)
    ]
    _, cg, acc_df = _create_toy_graph(accounts, txns)
    communities_df, signals = find_suspicious_communities(cg, acc_df)

    flagged = [s for s in signals if "SHOP" in s.accounts]
    assert len(flagged) == 0
