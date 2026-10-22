"""Synthetic financial network generator orchestrator.

Generates realistic background activity (consumer spending, payroll, p2p),
legitimate lookalike structures (roommates, family, bill splits),
and injected fraud patterns (cycles, structuring, mules, device clusters).
"""

from datetime import datetime, timedelta
import random
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.generator.config import GeneratorConfig
from src.generator.patterns import (
    generate_simple_cycle,
    generate_layered_cycle,
    generate_structuring,
    generate_mule_chain,
    generate_shared_device_cluster,
    generate_roommate_split,
    generate_family_network,
    generate_bill_split,
)


class SyntheticFinancialNetwork:
    """Orchestrates generation of accounts, transactions, and ground truth."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self.rng = random.Random(self.config.seed)

        self.accounts: List[Dict[str, Any]] = []
        self.transactions: List[Dict[str, Any]] = []
        self.ground_truth: Dict[str, Dict[str, Any]] = {}

        # Tracking summary metadata
        self.fraud_ring_counts: Dict[str, int] = {}
        self.lookalike_counts: Dict[str, int] = {}

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute full synthetic network generation.

        Returns:
            Tuple of (accounts_df, transactions_df, ground_truth_df).
        """
        # Step 1: Create account population
        self._generate_accounts()

        # Step 2: Inject legitimate lookalikes (roommates, families, bill splits)
        self._inject_lookalikes()

        # Step 3: Inject fraud patterns (cycles, structuring, mules, clusters)
        self._inject_fraud_rings()

        # Step 4: Generate realistic background commerce and payroll
        self._generate_background_traffic()

        # Step 5: Format and sort transactions chronologically
        accounts_df = pd.DataFrame(self.accounts)
        
        # Sort all transactions strictly by timestamp
        self.transactions.sort(key=lambda t: t["timestamp"])
        for idx, txn in enumerate(self.transactions, start=1):
            txn["txn_id"] = f"txn_{idx:06d}"

        transactions_df = pd.DataFrame(self.transactions)[[
            "txn_id", "src_account", "dst_account", "amount", "timestamp"
        ]]

        # Ground truth table
        ground_truth_records = list(self.ground_truth.values())
        ground_truth_df = (
            pd.DataFrame(ground_truth_records)
            .sort_values(by="account_id")
            .reset_index(drop=True)
        )

        return accounts_df, transactions_df, ground_truth_df

    def _generate_accounts(self) -> None:
        """Create baseline account records with devices and IPs."""
        total_accounts = (
            self.config.num_individuals
            + self.config.num_merchants
            + self.config.num_businesses
        )
        
        acc_idx = 1
        # Helper to generate random IP
        def random_ip() -> str:
            return f"192.168.{self.rng.randint(1, 254)}.{self.rng.randint(1, 254)}"

        # 1. Individuals
        for _ in range(self.config.num_individuals):
            acc_id = f"acc_{acc_idx:05d}"
            acc_idx += 1
            # Normal created date within 180 days before simulation
            created_days_prior = self.rng.randint(10, 180)
            created_at = self.config.start_date - timedelta(days=created_days_prior)

            self.accounts.append({
                "account_id": acc_id,
                "account_type": "individual",
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "device_id": f"dev_{self.rng.randint(1, 5000):06d}",
                "ip_address": random_ip(),
            })
            self._set_default_ground_truth(acc_id)

        # 2. Merchants
        for _ in range(self.config.num_merchants):
            acc_id = f"acc_{acc_idx:05d}"
            acc_idx += 1
            created_at = self.config.start_date - timedelta(days=self.rng.randint(60, 365))
            self.accounts.append({
                "account_id": acc_id,
                "account_type": "merchant",
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "device_id": f"pos_{self.rng.randint(1, 1000):04d}",
                "ip_address": random_ip(),
            })
            self._set_default_ground_truth(acc_id)

        # 3. Businesses
        for _ in range(self.config.num_businesses):
            acc_id = f"acc_{acc_idx:05d}"
            acc_idx += 1
            created_at = self.config.start_date - timedelta(days=self.rng.randint(90, 700))
            self.accounts.append({
                "account_id": acc_id,
                "account_type": "business",
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "device_id": f"corp_{self.rng.randint(1, 500):04d}",
                "ip_address": random_ip(),
            })
            self._set_default_ground_truth(acc_id)

    def _set_default_ground_truth(self, account_id: str) -> None:
        """Register initial clean ground truth for an account."""
        self.ground_truth[account_id] = {
            "account_id": account_id,
            "is_fraud": 0,
            "ring_id": None,
            "pattern_type": None,
            "lookalike_id": None,
            "lookalike_type": None,
        }

    def _get_available_individuals(self, count: int) -> List[str]:
        """Reserve individual accounts that have not been assigned to any special group."""
        available = [
            acc["account_id"]
            for acc in self.accounts
            if acc["account_type"] == "individual"
            and self.ground_truth[acc["account_id"]]["is_fraud"] == 0
            and self.ground_truth[acc["account_id"]]["lookalike_id"] is None
        ]
        if len(available) < count:
            raise ValueError(
                f"Requested {count} clean accounts, but only {len(available)} available. "
                "Increase num_individuals in GeneratorConfig."
            )
        chosen = available[:count]
        return chosen

    def _inject_lookalikes(self) -> None:
        """Inject legitimate lookalikes: roommates, families, bill splits."""
        # 1. Roommates
        for i in range(1, self.config.num_roommate_splits + 1):
            cohort = self._get_available_individuals(3)
            txns, gt = generate_roommate_split(
                self.rng, i, cohort, self.config.start_date, self.config.simulation_days
            )
            self.transactions.extend(txns)
            for item in gt:
                self.ground_truth[item["account_id"]].update(item)
            self.lookalike_counts["roommate_rent_split"] = (
                self.lookalike_counts.get("roommate_rent_split", 0) + 1
            )

        # 2. Families sharing a device
        for i in range(1, self.config.num_family_networks + 1):
            cohort = self._get_available_individuals(4)
            # Enforce shared device & IP for family accounts
            shared_dev = f"dev_family_{i:03d}"
            shared_ip = f"192.168.10.{i}"
            for acc_id in cohort:
                for acc in self.accounts:
                    if acc["account_id"] == acc_id:
                        acc["device_id"] = shared_dev
                        acc["ip_address"] = shared_ip

            txns, gt = generate_family_network(
                self.rng, i, cohort, self.config.start_date, self.config.simulation_days
            )
            self.transactions.extend(txns)
            for item in gt:
                self.ground_truth[item["account_id"]].update(item)
            self.lookalike_counts["family_shared_device"] = (
                self.lookalike_counts.get("family_shared_device", 0) + 1
            )

        # 3. Bill splits
        for i in range(1, self.config.num_bill_splits + 1):
            cohort = self._get_available_individuals(4)
            txns, gt = generate_bill_split(
                self.rng, i, cohort, self.config.start_date, self.config.simulation_days
            )
            self.transactions.extend(txns)
            for item in gt:
                self.ground_truth[item["account_id"]].update(item)
            self.lookalike_counts["friend_bill_split"] = (
                self.lookalike_counts.get("friend_bill_split", 0) + 1
            )

        # 4. Merchant fan-in (label top merchants as high fan-in lookalikes)
        merchants = [a["account_id"] for a in self.accounts if a["account_type"] == "merchant"]
        top_merchants = merchants[:3]
        for idx, m_id in enumerate(top_merchants, start=1):
            self.ground_truth[m_id].update({
                "lookalike_id": f"lookalike_merchant_{idx:03d}",
                "lookalike_type": "merchant_fan_in",
            })
        self.lookalike_counts["merchant_fan_in"] = len(top_merchants)

        # 5. Payroll fan-out (label top businesses as payroll lookalikes)
        businesses = [a["account_id"] for a in self.accounts if a["account_type"] == "business"]
        top_businesses = businesses[:3]
        for idx, b_id in enumerate(top_businesses, start=1):
            self.ground_truth[b_id].update({
                "lookalike_id": f"lookalike_payroll_{idx:03d}",
                "lookalike_type": "payroll_fan_out",
            })
        self.lookalike_counts["payroll_fan_out"] = len(top_businesses)

    def _inject_fraud_rings(self) -> None:
        """Inject synthetic fraud rings with strict ground truth tracking."""
        # 1. Simple cycles (A -> B -> C -> A within 48h)
        for i in range(1, self.config.num_simple_cycles + 1):
            cohort = self._get_available_individuals(3)
            # Pick a random starting day inside simulation
            start_day = self.rng.randint(5, self.config.simulation_days - 10)
            start_dt = self.config.start_date + timedelta(days=start_day, hours=self.rng.randint(6, 18))
            txns, gt = generate_simple_cycle(
                self.rng, i, cohort, start_dt, self.config.difficulty
            )
            self.transactions.extend(txns)
            for item in gt:
                self.ground_truth[item["account_id"]].update(item)
            self.fraud_ring_counts["simple_cycle"] = (
                self.fraud_ring_counts.get("simple_cycle", 0) + 1
            )

        # 2. Layered cycles (5-8 hops with fee skimming)
        for i in range(1, self.config.num_layered_cycles + 1):
            cohort = self._get_available_individuals(7)
            start_day = self.rng.randint(5, self.config.simulation_days - 20)
            start_dt = self.config.start_date + timedelta(days=start_day, hours=self.rng.randint(6, 18))
            txns, gt = generate_layered_cycle(
                self.rng, i, cohort, start_dt, self.config.difficulty
            )
            self.transactions.extend(txns)
            for item in gt:
                self.ground_truth[item["account_id"]].update(item)
            self.fraud_ring_counts["layered_cycle"] = (
                self.fraud_ring_counts.get("layered_cycle", 0) + 1
            )

        # 3. Structuring (smurfing transfers below threshold)
        for i in range(1, self.config.num_structuring_rings + 1):
            cohort = self._get_available_individuals(7)
            start_day = self.rng.randint(5, self.config.simulation_days - 15)
            start_dt = self.config.start_date + timedelta(days=start_day, hours=self.rng.randint(6, 18))
            txns, gt = generate_structuring(
                self.rng, i, cohort, start_dt, self.config.difficulty, self.config.threshold_amount
            )
            self.transactions.extend(txns)
            for item in gt:
                self.ground_truth[item["account_id"]].update(item)
            self.fraud_ring_counts["structuring"] = (
                self.fraud_ring_counts.get("structuring", 0) + 1
            )

        # 4. Mule chains (rapid pass-through)
        for i in range(1, self.config.num_mule_chains + 1):
            cohort = self._get_available_individuals(4)
            start_day = self.rng.randint(5, self.config.simulation_days - 10)
            start_dt = self.config.start_date + timedelta(days=start_day, hours=self.rng.randint(6, 18))
            txns, gt = generate_mule_chain(
                self.rng, i, cohort, start_dt, self.config.difficulty
            )
            self.transactions.extend(txns)
            for item in gt:
                self.ground_truth[item["account_id"]].update(item)
            self.fraud_ring_counts["mule_chain"] = (
                self.fraud_ring_counts.get("mule_chain", 0) + 1
            )

        # 5. Shared device clusters
        for i in range(1, self.config.num_shared_device_clusters + 1):
            cohort = self._get_available_individuals(4)
            # In shared device fraud clusters, new accounts are created close together
            shared_dev = f"dev_fraud_bot_{i:03d}"
            shared_ip = f"203.0.113.{10 + i}"
            base_creation = self.config.start_date + timedelta(days=self.rng.randint(10, 40))
            for idx, acc_id in enumerate(cohort):
                for acc in self.accounts:
                    if acc["account_id"] == acc_id:
                        acc["device_id"] = shared_dev
                        acc["ip_address"] = shared_ip
                        # Created within hours of each other
                        acc["created_at"] = (
                            base_creation + timedelta(hours=idx * 2)
                        ).strftime("%Y-%m-%d %H:%M:%S")

            start_dt = base_creation + timedelta(days=1, hours=4)
            txns, gt = generate_shared_device_cluster(
                self.rng, i, cohort, start_dt, self.config.difficulty
            )
            self.transactions.extend(txns)
            for item in gt:
                self.ground_truth[item["account_id"]].update(item)
            self.fraud_ring_counts["shared_device_cluster"] = (
                self.fraud_ring_counts.get("shared_device_cluster", 0) + 1
            )

        # If difficulty is hard, inject subtle normal camouflage transactions for fraud accounts
        if self.config.difficulty == "hard":
            self._inject_camouflage_transactions()

    def _inject_camouflage_transactions(self) -> None:
        """Add low-value merchant transactions from fraud accounts to simulate evasion."""
        merchants = [a["account_id"] for a in self.accounts if a["account_type"] == "merchant"]
        fraud_accounts = [
            acc_id for acc_id, record in self.ground_truth.items() if record["is_fraud"] == 1
        ]
        for f_acc in fraud_accounts:
            # 2 to 4 small merchant transactions
            for _ in range(self.rng.randint(2, 4)):
                m_target = self.rng.choice(merchants)
                rand_day = self.rng.randint(1, self.config.simulation_days - 1)
                t = self.config.start_date + timedelta(
                    days=rand_day,
                    hours=self.rng.randint(8, 21),
                    minutes=self.rng.randint(0, 59),
                )
                amt = round(self.rng.uniform(4.50, 45.00), 2)
                self.transactions.append({
                    "src_account": f_acc,
                    "dst_account": m_target,
                    "amount": amt,
                    "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "ring_id": None,
                })

    def _generate_background_traffic(self) -> None:
        """Generate normal daily consumer purchases, payroll, and p2p payments."""
        individuals = [a["account_id"] for a in self.accounts if a["account_type"] == "individual"]
        merchants = [a["account_id"] for a in self.accounts if a["account_type"] == "merchant"]
        businesses = [a["account_id"] for a in self.accounts if a["account_type"] == "business"]

        # 1. Bi-monthly Payroll fan-out from businesses to employees
        employees_per_biz = len(individuals) // len(businesses)
        for b_idx, biz_id in enumerate(businesses):
            emp_slice = individuals[b_idx * employees_per_biz : (b_idx + 1) * employees_per_biz]
            for month in range(max(1, self.config.simulation_days // 30)):
                # Pay on 1st and 15th
                for pay_day in [1, 15]:
                    d = month * 30 + pay_day
                    if d >= self.config.simulation_days:
                        continue
                    pay_time = self.config.start_date + timedelta(days=d, hours=9)
                    for emp in emp_slice:
                        salary = round(self.rng.uniform(2200.0, 5500.0), 2)
                        self.transactions.append({
                            "src_account": biz_id,
                            "dst_account": emp,
                            "amount": salary,
                            "timestamp": pay_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "ring_id": None,
                        })

        # 2. Daily consumer spending (Individuals -> Merchants)
        # Calculate how many ambient transactions needed to reach target_transactions
        current_count = len(self.transactions)
        remaining = max(1000, self.config.target_transactions - current_count)

        # Distribute consumer spending across simulation days
        txns_per_day = remaining // self.config.simulation_days
        for day in range(self.config.simulation_days):
            day_dt = self.config.start_date + timedelta(days=day)
            daily_quota = txns_per_day + self.rng.randint(-10, 10)
            for _ in range(max(1, daily_quota)):
                src = self.rng.choice(individuals)
                dst = self.rng.choice(merchants)
                
                # Realistic consumer amount ($3 to $280)
                amount = round(self.rng.expovariate(1.0 / 45.0) + 3.0, 2)
                amount = min(amount, 450.0)

                # Daytime business hours preferred (07:00 - 22:00)
                hour = self.rng.choices(
                    population=list(range(24)),
                    weights=[1, 1, 1, 1, 1, 2, 4, 7, 8, 8, 8, 9, 9, 8, 8, 9, 9, 8, 7, 5, 4, 3, 2, 1],
                    k=1,
                )[0]
                minute = self.rng.randint(0, 59)
                second = self.rng.randint(0, 59)
                t = day_dt + timedelta(hours=hour, minutes=minute, seconds=second)

                self.transactions.append({
                    "src_account": src,
                    "dst_account": dst,
                    "amount": round(amount, 2),
                    "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "ring_id": None,
                })

        # 3. Occasional legitimate peer-to-peer transfers between clean individuals
        p2p_count = int(self.config.simulation_days * 5)
        for _ in range(p2p_count):
            src, dst = self.rng.sample(individuals, 2)
            rand_day = self.rng.randint(0, self.config.simulation_days - 1)
            t = self.config.start_date + timedelta(
                days=rand_day,
                hours=self.rng.randint(10, 22),
                minutes=self.rng.randint(0, 59),
            )
            amt = round(self.rng.uniform(10.0, 150.0), 2)
            self.transactions.append({
                "src_account": src,
                "dst_account": dst,
                "amount": amt,
                "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                "ring_id": None,
            })
