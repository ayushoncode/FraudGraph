# FraudGraph: Explainable Financial Network Investigation Tool

FraudGraph is an explainable graph analytics tool designed to investigate suspicious transactional relationships across financial accounts. It constructs directed transaction graphs, identifies known suspicious flow topologies (such as circular fund movements, pass-through accounts, and fan-in/fan-out patterns), computes deterministic risk scores, and provides human investigators with exact, verifiable evidence behind every generated alert.

---

## Core Principles

1. **Synthetic Data Only**: All datasets and scenarios are strictly synthetic. Graph patterns never "prove" real fraud; alerts indicate suspicious topologies that are **"worth investigating"**.
2. **Deterministic & Explainable**: Detection logic is rule-based and transparent. Every risk score point is tied to an explicit, human-readable rationale.
3. **Reproducibility**: All synthetic generation and graph processes use fixed random seeds.
4. **Agent Role Boundaries**: Any future LLM agent integration is restricted to summarizing and explaining the structured evidence for investigators—never making autonomous decisions or generating risk scores.

---

## Topology Detection Patterns

- **Circular Flows (Cycles)**: Funds routed through multiple intermediary accounts returning to the source or originating ring.
- **Pass-Through Accounts (Mules)**: Accounts receiving inflows and rapidly routing similar amounts out with minimal retention and short dwell time.
- **Fan-In / Fan-Out (Smurfing / Layering)**:
  - *Fan-In*: Many accounts sending funds into a single aggregating account.
  - *Fan-Out*: A single account dispersing funds outward to many recipient accounts.
- **Bipartite & Dense Clusters**: Closely linked sub-networks and tightly coupled account communities.

---

## Project Structure

```text
fraudgraph/
├── data/              # Generated synthetic datasets (gitignored)
├── src/
│   ├── generator/     # Synthetic transaction & account generator
│   ├── graph/         # Directed graph construction (NetworkX)
│   ├── detectors/     # Topology detectors (cycles, pass-through, fan-in/out, communities)
│   ├── scoring/       # Interpretable risk rules & scoring logic
│   ├── evidence/      # Alert evidence formatting & subgraphs for investigators
│   └── evaluation/    # Benchmarks, baseline comparisons, and metrics
├── tests/             # Unit and integration test suite (pytest)
├── requirements.txt   # Core Python dependencies
├── .gitignore         # Ignored files (data artifacts, venv, caches)
└── README.md          # Project documentation
```

---

## Current Tech Stack

- **Python**: 3.11+
- **Graph Processing**: [NetworkX](https://networkx.org/)
- **Data Manipulation**: [pandas](https://pandas.pydata.org/)
- **Machine Learning Baselines & Metrics**: [scikit-learn](https://scikit-learn.org/)
- **Testing**: [pytest](https://docs.pytest.org/)

*(Future milestones will incorporate FastAPI, Pydantic, PostgreSQL, Next.js + TypeScript, and Docker Compose).*

---

## Getting Started

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment using Python 3.11+
python3.11 -m venv .venv

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Tests

```bash
pytest
```
