"""Base data structures and configuration for FraudGraph pattern detectors.

This module defines:
1. Signal: The standard container for all detector outputs, storing the involved accounts,
   exact transaction IDs (for Phase 6 evidence generation), time bounds, and metrics.
2. DetectorConfig: A centralized configuration dataclass holding all detection thresholds.
   Never hardcode thresholds inside detector logic.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class Signal:
    """Represents a suspicious pattern alert emitted by a detector.

    Attributes:
        signal_id: Unique identifier for the alert (e.g., 'sig_cycle_0001').
        pattern_type: Pattern category ('cycle', 'pass_through', 'pass_through_chain',
                      'fan_in_fan_out', 'structuring').
        accounts: List of account IDs directly involved in the detected pattern.
        txn_ids: List of transaction IDs forming the exact empirical evidence trail.
                 Required for explainability and investigator subgraphs.
        start_ts: Earliest timestamp of the suspicious activity (ISO string).
        end_ts: Latest timestamp of the suspicious activity (ISO string).
        total_amount: Total financial volume involved in the pattern.
        details: Dictionary containing the specific mathematical and topological metrics
                 justifying the alert (retention ratios, durations, degrees, etc.).
    """

    signal_id: str
    pattern_type: str
    accounts: List[str]
    txn_ids: List[str]
    start_ts: str
    end_ts: str
    total_amount: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to a JSON-serializable dictionary."""
        return {
            "signal_id": self.signal_id,
            "pattern_type": self.pattern_type,
            "accounts": self.accounts,
            "txn_ids": self.txn_ids,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "total_amount": round(self.total_amount, 2),
            "details": self.details,
        }


@dataclass
class DetectorConfig:
    """Centralized thresholds and configuration parameters for all detectors.

    All detection levers are declared here with defaults and commentary.
    Never hardcode magic numbers inside individual detector functions.
    """

    # -------------------------------------------------------------------------
    # Circular Flow (Cycles) Detection
    # -------------------------------------------------------------------------
    cycle_min_len: int = 3
    """Minimum cycle length (number of hops). Set to 3 to ignore normal 2-party reciprocal transfers."""

    cycle_max_len: int = 8
    """Maximum cycle length to search for. Criminal laundering rarely exceeds 8 hops due to friction."""

    cycle_window_hours: float = 48.0
    """Maximum allowable time delta between the first and last hop in the cycle."""

    cycle_retention_min: float = 0.70
    """Minimum amount retention ratio between successive hops (amount_next >= 0.70 * amount_prev).
    Allows up to 30% loss to transaction fees, smurf cuts, or currency conversion."""

    cycle_retention_max: float = 1.05
    """Maximum amount ratio between successive hops (amount_next <= 1.05 * amount_prev).
    Allows minor account balance top-ups."""

    cycle_max_candidates: int = 5000
    """Safety ceiling on candidate cycles evaluated from simple_cycles to prevent
    combinatorial explosion in dense network components."""

    skip_hubs: bool = False
    """Trade-off: If True, accounts with degree >= hub_degree_threshold are excluded from cycle search.
    Trade-off rationale: Popular merchants and payroll accounts create massive numbers of benign
    combinatorial paths. Skipping them speeds up search dramatically, but risks missing fraud rings
    deliberately routing hops through compromised high-volume accounts."""

    hub_degree_threshold: int = 100
    """Total degree (in_degree + out_degree) above which an account is considered a hub."""

    # -------------------------------------------------------------------------
    # Pass-Through (Mule / Layering) Detection
    # -------------------------------------------------------------------------
    passthrough_minutes: float = 30.0
    """Maximum dwell time in minutes between an inflow and its matched outflow."""

    passthrough_min_ratio: float = 0.80
    """Minimum outflow / inflow ratio (outflow >= 80% of inflow)."""

    passthrough_max_ratio: float = 1.05
    """Maximum outflow / inflow ratio to prevent matching unrelated massive payments."""

    passthrough_min_amount: float = 500.0
    """Minimum transaction dollar amount to consider for pass-through analysis."""

    passthrough_min_events: int = 1
    """Minimum number of matched pass-through events required to trigger an alert."""

    # -------------------------------------------------------------------------
    # Fan-In / Fan-Out (Dispersion & Aggregation) Detection
    # -------------------------------------------------------------------------
    fan_window_hours: float = 24.0
    """Duration of the sliding window for aggregating fan-in and fan-out counts."""

    fan_in_threshold: int = 8
    """Minimum distinct senders within the time window to flag significant aggregation."""

    fan_out_threshold: int = 8
    """Minimum distinct receivers within the time window to flag significant dispersal."""

    fan_min_amount: float = 1000.0
    """Minimum financial volume within the window for fan-in/fan-out evaluation."""

    # -------------------------------------------------------------------------
    # Structuring (Smurfing) Detection
    # -------------------------------------------------------------------------
    reporting_threshold: float = 50000.0
    """Statutory reporting threshold for financial transactions (e.g., $50,000)."""

    structuring_window_hours: float = 72.0
    """Time window in hours during which sub-threshold transfers are consolidated."""

    structuring_min_transfers: int = 3
    """Minimum number of distinct sub-threshold transfers into the target account."""

    structuring_lower_bound_ratio: float = 0.70
    """Lower bound factor of reporting threshold (e.g. 0.70 * 50,000 = $35,000).
    Transfers between $35k and $50k are considered 'just below' the threshold."""

    # -------------------------------------------------------------------------
    # Community & Dense Subgraph Detection (Louvain)
    # -------------------------------------------------------------------------
    louvain_resolution: float = 1.0
    """Resolution parameter for Louvain modularity optimization (1.0 = standard modularity)."""

    louvain_seed: int = 42
    """Fixed random seed for deterministic Louvain community partitions."""

    community_min_size: int = 3
    """Minimum number of accounts to evaluate as an insular community."""

    community_max_size: int = 30
    """Maximum number of accounts for a community alert. Communities larger than 30 are
    considered too large/diffuse for single-case investigation and are marked as oversized."""

    insularity_min: float = 0.70
    """Minimum insularity ratio: internal_amount / (internal_amount + external_amount).
    Rings with >= 70% of their money recirculating internally are flagged as insular."""

    new_account_days: int = 30
    """Window in days to classify an account as 'new' relative to group activity."""

    # -------------------------------------------------------------------------
    # Shared Device & Hardware Cluster Detection
    # -------------------------------------------------------------------------
    shared_device_min_accounts: int = 3
    """Minimum accounts sharing the same device_id or ip_address to evaluate as a cluster."""

    device_min_internal_txns: int = 3
    """Minimum internal transactions between co-located accounts required to alert.
    Prevents flagging families sharing a tablet who rarely or never transfer money to each other."""

