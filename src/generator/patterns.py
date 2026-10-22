"""Pattern injection logic for synthetic fraud rings and legitimate lookalikes.

Every pattern here is deterministic (governed by an isolated random.Random instance)
and produces explicit ground-truth labels for evaluation.
"""

from datetime import datetime, timedelta
import random
from typing import Any, Dict, List, Tuple


def generate_simple_cycle(
    rng: random.Random,
    ring_idx: int,
    accounts: List[str],
    start_time: datetime,
    difficulty: str,
    base_amount: float = 18500.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Inject a simple circular flow: A -> B -> C -> A within 48 hours.

    Interview Explanation:
    Circular flows are an evasion tactic where funds travel in a closed loop to create
    the illusion of independent commerce or to wash money through intermediaries.
    """
    ring_id = f"ring_simple_cycle_{ring_idx:03d}"
    txns: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    # Simple cycle uses 3 accounts
    a, b, c = accounts[:3]
    for acc in [a, b, c]:
        ground_truth.append({
            "account_id": acc,
            "is_fraud": 1,
            "ring_id": ring_id,
            "pattern_type": "simple_cycle",
            "lookalike_id": None,
            "lookalike_type": None,
        })

    # Timing between hops (must complete within 48h)
    if difficulty == "easy":
        hop_delta = timedelta(hours=6)
        amount = base_amount
    elif difficulty == "medium":
        hop_delta = timedelta(hours=rng.randint(8, 14))
        amount = round(base_amount + rng.uniform(-150.0, 150.0), 2)
    else:  # hard
        hop_delta = timedelta(hours=rng.randint(10, 15))
        amount = round(base_amount + rng.uniform(-300.0, 300.0), 2)

    t1 = start_time
    t2 = t1 + hop_delta
    t3 = t2 + hop_delta

    txns.append({
        "src_account": a,
        "dst_account": b,
        "amount": amount,
        "timestamp": t1.strftime("%Y-%m-%d %H:%M:%S"),
        "ring_id": ring_id,
    })
    txns.append({
        "src_account": b,
        "dst_account": c,
        "amount": amount if difficulty == "easy" else round(amount * 0.99, 2),
        "timestamp": t2.strftime("%Y-%m-%d %H:%M:%S"),
        "ring_id": ring_id,
    })
    txns.append({
        "src_account": c,
        "dst_account": a,
        "amount": amount if difficulty == "easy" else round(amount * 0.98, 2),
        "timestamp": t3.strftime("%Y-%m-%d %H:%M:%S"),
        "ring_id": ring_id,
    })

    return txns, ground_truth


def generate_layered_cycle(
    rng: random.Random,
    ring_idx: int,
    accounts: List[str],
    start_time: datetime,
    difficulty: str,
    base_amount: float = 45000.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Inject a long cycle (5-8 accounts) with time delays and a fee skimmed at each hop.

    Interview Explanation:
    Layering inserts multiple hops into the path to increase distance from the origin.
    At each step, facilitators typically skim a small fee (1% - 3%), so the amount
    slowly decreases along the cycle.
    """
    ring_id = f"ring_layered_cycle_{ring_idx:03d}"
    txns: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    # Select length based on difficulty: 5 hops for easy, up to 7-8 for hard
    if difficulty == "easy":
        hop_count = 5
        fee_rate = 0.02
    elif difficulty == "medium":
        hop_count = 6
        fee_rate = 0.025
    else:
        hop_count = min(len(accounts), 7)
        fee_rate = 0.03

    cycle_accounts = accounts[:hop_count]
    for acc in cycle_accounts:
        ground_truth.append({
            "account_id": acc,
            "is_fraud": 1,
            "ring_id": ring_id,
            "pattern_type": "layered_cycle",
            "lookalike_id": None,
            "lookalike_type": None,
        })

    current_time = start_time
    current_amount = base_amount

    for i in range(hop_count):
        src = cycle_accounts[i]
        dst = cycle_accounts[(i + 1) % hop_count]

        # Delay between hops: 12 to 24 hours
        hop_hours = 12 if difficulty == "easy" else rng.randint(10, 24)
        current_time += timedelta(hours=hop_hours)

        txns.append({
            "src_account": src,
            "dst_account": dst,
            "amount": round(current_amount, 2),
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ring_id": ring_id,
        })
        # Skim fee for next hop
        current_amount *= (1.0 - fee_rate)

    return txns, ground_truth


def generate_structuring(
    rng: random.Random,
    ring_idx: int,
    accounts: List[str],
    start_time: datetime,
    difficulty: str,
    threshold: float = 50000.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Inject structuring (smurfing): large amount split into transfers just under threshold.

    Interview Explanation:
    Structuring is the practice of breaking up large financial transactions into multiple
    smaller ones specifically to avoid statutory reporting thresholds (e.g. $50,000).
    A source disperses sub-threshold amounts to smurfs, who then consolidate into a sink.
    """
    ring_id = f"ring_structuring_{ring_idx:03d}"
    txns: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    # 1 source, 1 sink, and 3-5 smurfs
    num_smurfs = 3 if difficulty == "easy" else (4 if difficulty == "medium" else 5)
    needed = 2 + num_smurfs
    used_accounts = accounts[:needed]
    source = used_accounts[0]
    sink = used_accounts[1]
    smurfs = used_accounts[2:]

    for acc in [source, sink] + smurfs:
        ground_truth.append({
            "account_id": acc,
            "is_fraud": 1,
            "ring_id": ring_id,
            "pattern_type": "structuring",
            "lookalike_id": None,
            "lookalike_type": None,
        })

    # Step 1: Disperse funds to smurfs just below threshold (e.g. 45,000 to 49,500)
    current_time = start_time
    smurf_amounts: List[float] = []

    for smurf in smurfs:
        if difficulty == "easy":
            amount = threshold - 1200.0  # e.g., 48,800
        else:
            amount = round(threshold - rng.uniform(800.0, 4500.0), 2)
        smurf_amounts.append(amount)

        current_time += timedelta(hours=rng.randint(1, 4))
        txns.append({
            "src_account": source,
            "dst_account": smurf,
            "amount": amount,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ring_id": ring_id,
        })

    # Step 2: Smurfs forward their funds to the final sink account shortly after
    for smurf, amt in zip(smurfs, smurf_amounts):
        # Consolidation delay
        fwd_delay = timedelta(hours=rng.randint(2, 8))
        fwd_time = current_time + fwd_delay
        # In practice, smurfs might keep a small $100 cut or forward clean
        fwd_amount = amt if difficulty == "easy" else round(amt * 0.985, 2)
        txns.append({
            "src_account": smurf,
            "dst_account": sink,
            "amount": fwd_amount,
            "timestamp": fwd_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ring_id": ring_id,
        })

    return txns, ground_truth


def generate_mule_chain(
    rng: random.Random,
    ring_idx: int,
    accounts: List[str],
    start_time: datetime,
    difficulty: str,
    base_amount: float = 32000.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Inject a rapid pass-through mule chain: funds received and pushed out within minutes.

    Interview Explanation:
    Mules act as temporary buffers. The key signature is high flow-through ratio (near 100%)
    and extremely short dwell time (minutes to a few hours), leaving minimal balance.
    """
    ring_id = f"ring_mule_{ring_idx:03d}"
    txns: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    # Origin -> Mule 1 -> (Mule 2 if hard) -> Final Receiver
    is_multi_hop = (difficulty == "hard")
    chain = accounts[:4] if is_multi_hop else accounts[:3]

    for acc in chain:
        ground_truth.append({
            "account_id": acc,
            "is_fraud": 1,
            "ring_id": ring_id,
            "pattern_type": "mule_chain",
            "lookalike_id": None,
            "lookalike_type": None,
        })

    current_time = start_time
    current_amount = base_amount

    for i in range(len(chain) - 1):
        src = chain[i]
        dst = chain[i + 1]

        txns.append({
            "src_account": src,
            "dst_account": dst,
            "amount": round(current_amount, 2),
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ring_id": ring_id,
        })

        # Mules forward funds quickly: 15 to 45 minutes
        minutes_delay = 15 if difficulty == "easy" else rng.randint(15, 60)
        current_time += timedelta(minutes=minutes_delay)
        # Mule retains small commission (1% to 2%)
        current_amount *= 0.98

    return txns, ground_truth


def generate_shared_device_cluster(
    rng: random.Random,
    ring_idx: int,
    accounts: List[str],
    start_time: datetime,
    difficulty: str,
    base_amount: float = 24000.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Inject fraudulent cluster: accounts sharing device or IP moving money among themselves.

    Interview Explanation:
    Synthetic identity rings or botnets often register multiple accounts from identical
    hardware/IP footprints and churn funds rapidly between them.
    """
    ring_id = f"ring_shared_device_{ring_idx:03d}"
    txns: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    cluster_size = 3 if difficulty == "easy" else 4
    cluster_accounts = accounts[:cluster_size]

    for acc in cluster_accounts:
        ground_truth.append({
            "account_id": acc,
            "is_fraud": 1,
            "ring_id": ring_id,
            "pattern_type": "shared_device_cluster",
            "lookalike_id": None,
            "lookalike_type": None,
        })

    current_time = start_time
    # High-velocity interconnected transactions within the cluster
    for i in range(cluster_size):
        src = cluster_accounts[i]
        dst = cluster_accounts[(i + 1) % cluster_size]
        amt = round(base_amount + rng.uniform(-500.0, 500.0), 2)
        current_time += timedelta(hours=rng.randint(2, 6))

        txns.append({
            "src_account": src,
            "dst_account": dst,
            "amount": amt,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ring_id": ring_id,
        })

    # In medium/hard, add an additional internal cross-transfer
    if difficulty in ("medium", "hard"):
        src = cluster_accounts[0]
        dst = cluster_accounts[cluster_size - 1]
        current_time += timedelta(hours=3)
        txns.append({
            "src_account": src,
            "dst_account": dst,
            "amount": round(base_amount * 0.5, 2),
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ring_id": ring_id,
        })

    return txns, ground_truth


# ---------------------------------------------------------------------------
# LEGITIMATE LOOKALIKES
# ---------------------------------------------------------------------------


def generate_roommate_split(
    rng: random.Random,
    group_idx: int,
    accounts: List[str],
    start_time: datetime,
    simulation_days: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Legitimate lookalike: roommates splitting rent and utility bills.

    Often mimics cycles or fan-in: roommate A pays the landlord, roommates B & C
    send their portion to A each month.
    """
    lookalike_id = f"lookalike_roommates_{group_idx:03d}"
    txns: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    a, b, c = accounts[:3]
    for acc in [a, b, c]:
        ground_truth.append({
            "account_id": acc,
            "is_fraud": 0,
            "ring_id": None,
            "pattern_type": None,
            "lookalike_id": lookalike_id,
            "lookalike_type": "roommate_rent_split",
        })

    rent_share = 950.0  # realistic monthly rent portion
    num_months = max(1, simulation_days // 30)

    for m in range(num_months):
        month_time = start_time + timedelta(days=m * 30 + 1)
        # B pays A
        t_b = month_time + timedelta(hours=rng.randint(8, 14))
        txns.append({
            "src_account": b,
            "dst_account": a,
            "amount": rent_share,
            "timestamp": t_b.strftime("%Y-%m-%d %H:%M:%S"),
            "ring_id": None,
        })
        # C pays A
        t_c = month_time + timedelta(hours=rng.randint(9, 18))
        txns.append({
            "src_account": c,
            "dst_account": a,
            "amount": rent_share,
            "timestamp": t_c.strftime("%Y-%m-%d %H:%M:%S"),
            "ring_id": None,
        })
        # Occasional utility split back from A to B or C
        t_u = month_time + timedelta(days=15, hours=rng.randint(10, 16))
        txns.append({
            "src_account": a,
            "dst_account": c,
            "amount": 45.0,  # e.g., internet split
            "timestamp": t_u.strftime("%Y-%m-%d %H:%M:%S"),
            "ring_id": None,
        })

    return txns, ground_truth


def generate_family_network(
    rng: random.Random,
    group_idx: int,
    accounts: List[str],
    start_time: datetime,
    simulation_days: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Legitimate lookalike: family members sharing a device sending allowances & gifts.

    Crucial for interview discussions: shared device alone must NEVER trigger a fraud alert.
    """
    lookalike_id = f"lookalike_family_{group_idx:03d}"
    txns: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    family_size = min(len(accounts), 4)
    family_members = accounts[:family_size]

    for acc in family_members:
        ground_truth.append({
            "account_id": acc,
            "is_fraud": 0,
            "ring_id": None,
            "pattern_type": None,
            "lookalike_id": lookalike_id,
            "lookalike_type": "family_shared_device",
        })

    # Parents send allowances or pocket money to children bi-weekly
    parent = family_members[0]
    children = family_members[1:]

    biweeks = max(1, simulation_days // 14)
    for w in range(biweeks):
        base_dt = start_time + timedelta(days=w * 14 + rng.randint(0, 3))
        for child in children:
            t = base_dt + timedelta(hours=rng.randint(10, 20))
            amount = round(rng.uniform(30.0, 150.0), 2)
            txns.append({
                "src_account": parent,
                "dst_account": child,
                "amount": amount,
                "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                "ring_id": None,
            })

    return txns, ground_truth


def generate_bill_split(
    rng: random.Random,
    group_idx: int,
    accounts: List[str],
    start_time: datetime,
    simulation_days: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Legitimate lookalike: friend group splitting restaurant, travel, and outing bills."""
    lookalike_id = f"lookalike_bill_split_{group_idx:03d}"
    txns: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    group_size = min(len(accounts), 4)
    friends = accounts[:group_size]

    for acc in friends:
        ground_truth.append({
            "account_id": acc,
            "is_fraud": 0,
            "ring_id": None,
            "pattern_type": None,
            "lookalike_id": lookalike_id,
            "lookalike_type": "friend_bill_split",
        })

    # Several outings across the simulation
    num_outings = simulation_days // 10
    for o in range(num_outings):
        out_date = start_time + timedelta(days=o * 10 + rng.randint(0, 4))
        # One friend pays, other friends send reimbursement back
        payer = friends[o % group_size]
        others = [f for f in friends if f != payer]

        for other in others:
            t = out_date + timedelta(hours=rng.randint(18, 23), minutes=rng.randint(0, 59))
            amt = round(rng.uniform(15.0, 65.0), 2)
            txns.append({
                "src_account": other,
                "dst_account": payer,
                "amount": amt,
                "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                "ring_id": None,
            })

    return txns, ground_truth
