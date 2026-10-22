"""Unit tests for the graph module (builder, queries, and stats)."""

from datetime import datetime, timedelta
import networkx as nx
import pandas as pd
import pytest

from src.graph.builder import build_collapsed_graph, build_multigraph, load_data
from src.graph.queries import (
    accounts_sharing_device,
    filter_by_time,
    get_account_info,
    get_neighbors,
    get_subgraph,
    get_transactions,
)
from src.graph.stats import graph_summary


@pytest.fixture
def sample_data():
    """Create a minimal, deterministic set of accounts and transactions for testing."""
    accounts_df = pd.DataFrame([
        {
            "account_id": "acc_001",
            "account_type": "individual",
            "created_at": pd.to_datetime("2025-01-01 10:00:00"),
            "device_id": "dev_A",
            "ip_address": "192.168.1.1",
        },
        {
            "account_id": "acc_002",
            "account_type": "individual",
            "created_at": pd.to_datetime("2025-01-02 11:00:00"),
            "device_id": "dev_A",  # shares device with acc_001
            "ip_address": "192.168.1.2",
        },
        {
            "account_id": "acc_003",
            "account_type": "merchant",
            "created_at": pd.to_datetime("2024-12-01 09:00:00"),
            "device_id": "pos_01",
            "ip_address": "192.168.1.3",
        },
        {
            "account_id": "acc_004",
            "account_type": "business",
            "created_at": pd.to_datetime("2024-11-01 08:00:00"),
            "device_id": "corp_01",
            "ip_address": "192.168.1.4",
        },
    ])

    transactions_df = pd.DataFrame([
        {
            "txn_id": "txn_001",
            "src_account": "acc_001",
            "dst_account": "acc_002",
            "amount": 100.0,
            "timestamp": pd.to_datetime("2025-01-10 12:00:00"),
        },
        {
            "txn_id": "txn_002",
            "src_account": "acc_001",
            "dst_account": "acc_002",
            "amount": 150.0,
            "timestamp": pd.to_datetime("2025-01-11 12:00:00"),
        },
        {
            "txn_id": "txn_003",
            "src_account": "acc_002",
            "dst_account": "acc_003",
            "amount": 75.0,
            "timestamp": pd.to_datetime("2025-01-15 14:00:00"),
        },
        {
            "txn_id": "txn_004",
            "src_account": "acc_004",
            "dst_account": "acc_001",
            "amount": 3000.0,
            "timestamp": pd.to_datetime("2025-01-05 09:00:00"),
        },
    ])

    return accounts_df, transactions_df


def test_node_and_edge_counts(sample_data):
    """Test 1: Node count equals number of accounts; edge count equals number of transactions."""
    accounts_df, txns_df = sample_data
    multigraph = build_multigraph(accounts_df, txns_df)

    assert multigraph.number_of_nodes() == len(accounts_df)
    assert multigraph.number_of_edges() == len(txns_df)

    # Node attributes check
    node_data = multigraph.nodes["acc_001"]
    assert node_data["account_type"] == "individual"
    assert node_data["device_id"] == "dev_A"


def test_collapsed_graph_amounts_equal_multigraph(sample_data):
    """Test 2: Collapsed graph total_amount per pair equals the sum from the multigraph."""
    accounts_df, txns_df = sample_data
    multigraph = build_multigraph(accounts_df, txns_df)
    collapsed = build_collapsed_graph(multigraph)

    # acc_001 -> acc_002 has two transactions (100.0 + 150.0 = 250.0)
    assert collapsed.has_edge("acc_001", "acc_002")
    edge_data = collapsed["acc_001"]["acc_002"]
    assert edge_data["total_amount"] == 250.0
    assert edge_data["txn_count"] == 2
    assert edge_data["txn_ids"] == ["txn_001", "txn_002"]
    assert edge_data["first_ts"] == pd.to_datetime("2025-01-10 12:00:00")
    assert edge_data["last_ts"] == pd.to_datetime("2025-01-11 12:00:00")

    # Global total sum matches
    multi_sum = sum(d["amount"] for _, _, d in multigraph.edges(data=True))
    collapsed_sum = sum(d["total_amount"] for _, _, d in collapsed.edges(data=True))
    assert multi_sum == collapsed_sum == 3325.0


def test_filter_by_time_window(sample_data):
    """Test 3: filter_by_time returns only edges strictly inside the window."""
    accounts_df, txns_df = sample_data
    multigraph = build_multigraph(accounts_df, txns_df)

    # Filter for window 2025-01-09 to 2025-01-12 (should only include txn_001 and txn_002)
    filtered = filter_by_time(
        multigraph,
        start="2025-01-09 00:00:00",
        end="2025-01-12 23:59:59",
    )

    assert filtered.number_of_nodes() == 4
    assert filtered.number_of_edges() == 2
    included_txns = [d["txn_id"] for _, _, d in filtered.edges(data=True)]
    assert sorted(included_txns) == ["txn_001", "txn_002"]


def test_get_neighbors_on_tiny_graph():
    """Test 4: get_neighbors is correct on a tiny hand-made graph."""
    # Graph: A -> B, C -> B, B -> D
    G = nx.DiGraph()
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_edge("A", "B")
    G.add_edge("C", "B")
    G.add_edge("B", "D")

    neighbors_B = get_neighbors(G, "B")
    assert neighbors_B["in_neighbors"] == ["A", "C"]
    assert neighbors_B["out_neighbors"] == ["D"]

    neighbors_A = get_neighbors(G, "A")
    assert neighbors_A["in_neighbors"] == []
    assert neighbors_A["out_neighbors"] == ["B"]


def test_builder_fails_on_unknown_account():
    """Test 5: Builder fails clearly on a transaction that references an unknown account."""
    accounts_df = pd.DataFrame([
        {"account_id": "acc_001", "account_type": "individual", "created_at": "2025-01-01", "device_id": "d1", "ip_address": "1.1.1.1"}
    ])
    # txn references unknown dst_account 'acc_999'
    bad_txns_df = pd.DataFrame([
        {"txn_id": "txn_001", "src_account": "acc_001", "dst_account": "acc_999", "amount": 50.0, "timestamp": "2025-01-02"}
    ])

    # Test load_data validation via temporary csvs
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pd.io.common.Path(tmp_dir)
        accounts_df.to_csv(tmp_path / "accounts.csv", index=False)
        bad_txns_df.to_csv(tmp_path / "transactions.csv", index=False)

        with pytest.raises(ValueError) as exc_info:
            load_data(tmp_path)

        assert "unknown account" in str(exc_info.value).lower()
        assert "acc_999" in str(exc_info.value)


def test_device_sharing_and_queries(sample_data):
    """Test helper queries: accounts_sharing_device, get_account_info, get_subgraph."""
    accounts_df, txns_df = sample_data
    multigraph = build_multigraph(accounts_df, txns_df)

    # Device sharing: dev_A is shared by acc_001 and acc_002
    sharing = accounts_sharing_device(multigraph)
    assert "dev_A" in sharing
    assert sharing["dev_A"] == ["acc_001", "acc_002"]
    assert "pos_01" not in sharing  # only 1 account, should not be included

    # Account info
    info = get_account_info(multigraph, "acc_001")
    assert info["account_id"] == "acc_001"
    assert info["in_degree"] == 1  # received from acc_004
    assert info["out_degree"] == 2  # sent 2 txns to acc_002

    # Get transactions sorted
    txns = get_transactions(multigraph, "acc_001", direction="both")
    assert len(txns) == 3
    # Check chronologically sorted
    assert txns[0]["timestamp"] <= txns[1]["timestamp"] <= txns[2]["timestamp"]

    # Subgraph
    sub = get_subgraph(multigraph, ["acc_001", "acc_002"])
    assert sub.number_of_nodes() == 2
    assert sub.number_of_edges() == 2  # only the edges between acc_001 and acc_002


def test_real_dataset_loading():
    """Verify loading the actual generated data/ directory if it exists."""
    import os
    if os.path.exists("data/accounts.csv") and os.path.exists("data/transactions.csv"):
        accounts_df, txns_df = load_data("data")
        G = build_multigraph(accounts_df, txns_df)
        summary = graph_summary(G)

        assert summary["node_count"] == len(accounts_df)
        assert summary["edge_count"] == len(txns_df)
        assert summary["total_amount_moved"] > 0
        assert summary["density"] > 0
