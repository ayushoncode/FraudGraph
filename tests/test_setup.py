"""Smoke test to verify that the environment and core dependencies are correctly installed."""

import networkx as nx
import pandas as pd
import sklearn
import pytest


def test_environment_dependencies():
    """Verify core libraries can be imported and initialized."""
    # Test networkx graph creation
    graph = nx.DiGraph()
    graph.add_edge("acc_A", "acc_B", amount=100.0)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1

    # Test pandas DataFrame creation
    df = pd.DataFrame([{"sender": "acc_A", "receiver": "acc_B", "amount": 100.0}])
    assert len(df) == 1
    assert "sender" in df.columns

    # Test scikit-learn import
    assert hasattr(sklearn, "__version__")
