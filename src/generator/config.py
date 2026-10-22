"""Configuration dataclass for the synthetic financial network generator.

This module defines all configurable parameters for account population,
transaction volume, simulation timeline, and pattern counts.
Using a dataclass allows clean parameter passing and makes interview
explanations simple: all assumptions and levers are in one place.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class GeneratorConfig:
    """Settings controlling synthetic network generation.

    Attributes:
        seed: Random seed for deterministic reproducibility.
        difficulty: Complexity of injected patterns ('easy', 'medium', 'hard').
        simulation_days: Timeline window for transactions in days.
        start_date: Starting datetime of the simulation.
        num_individuals: Number of regular individual consumer accounts.
        num_merchants: Number of merchant/retailer accounts.
        num_businesses: Number of corporate/employer accounts.
        target_transactions: Approximate target for total transactions (5,000 - 20,000).
        threshold_amount: Financial threshold reference for structuring (e.g., 50,000).
        num_simple_cycles: Number of simple 3-hop cycles to inject.
        num_layered_cycles: Number of long 5-8 hop cycles to inject.
        num_structuring_rings: Number of structuring (smurfing) rings to inject.
        num_mule_chains: Number of rapid pass-through mule chains to inject.
        num_shared_device_clusters: Number of shared-device fraud rings to inject.
        num_roommate_splits: Number of legitimate roommate rent cycles to inject.
        num_family_networks: Number of legitimate family shared-device groups.
        num_bill_splits: Number of legitimate dining/bill split groups.
    """

    seed: int = 42
    difficulty: str = "medium"  # 'easy', 'medium', 'hard'
    simulation_days: int = 90
    start_date: datetime = datetime(2025, 1, 1, 0, 0, 0)

    # Base entity sizing
    num_individuals: int = 400
    num_merchants: int = 30
    num_businesses: int = 15
    target_transactions: int = 10000
    threshold_amount: float = 50000.0

    # Injected fraud pattern frequencies
    num_simple_cycles: int = 3
    num_layered_cycles: int = 2
    num_structuring_rings: int = 3
    num_mule_chains: int = 3
    num_shared_device_clusters: int = 2

    # Injected legitimate lookalike frequencies
    num_roommate_splits: int = 4
    num_family_networks: int = 5
    num_bill_splits: int = 5

    def __post_init__(self) -> None:
        """Validate configuration settings."""
        valid_difficulties = ("easy", "medium", "hard")
        if self.difficulty not in valid_difficulties:
            raise ValueError(
                f"Invalid difficulty '{self.difficulty}'. Must be one of {valid_difficulties}."
            )
        if self.target_transactions < 1000:
            raise ValueError("target_transactions must be at least 1,000.")
        if self.simulation_days < 1:
            raise ValueError("simulation_days must be at least 1.")
