# FRESCO — Multi-Agent Inventory & Waste Reduction System

CAP 6942 Capstone Project — Sanjana Vijayabhaskar & [Teammate Name]

## What this is

A six-agent system that automates the weekly restaurant ordering decision
(what to order, how much, from whom) while balancing demand uncertainty,
perishability, supplier constraints, and budget. See `docs/` (or your
submitted Statement of Work / Requirements Statement / Project Plan) for
the full design rationale.

## Project status

**Week 1 complete** (environment & data setup): representative SKUs,
synthetic demand data, and a supplier catalog are in place under `data/`,
and the six-agent skeleton is wired end-to-end with placeholder logic in
each agent, confirmed by the smoke test in `tests/`.

## Structure

```
fresco-project/
├── data/                     # Week 1 deliverable
│   ├── items.csv             # Item master — 54 SKUs, 9 categories
│   ├── demand.csv            # Daily sales history (Demand Agent input)
│   ├── inventory_params.csv  # Safety stock / reorder point (Inventory Agent input)
│   ├── suppliers.csv         # Cost / lead time / MOQ (Supplier Agent input)
│   ├── DATA_DICTIONARY.md    # Full column definitions + methodology
│   └── generate.py           # Script that generated the above (rerun to regenerate)
├── agents/
│   ├── base_agent.py          # Shared interface every agent implements
│   ├── data_loader.py         # Loads + validates the four data files
│   ├── demand_agent.py        # FR1 — Week 2
│   ├── inventory_agent.py     # FR2 — Week 3
│   ├── supplier_agent.py      # FR3 — Week 4
│   ├── optimization_agent.py  # FR4 — Week 5
│   ├── critic_agent.py        # FR5 — Week 6
│   └── orchestrator_agent.py  # FR6 — Week 7 (wires everything together)
├── tests/
│   └── test_week1_setup.py   # Confirms env + data + pipeline wiring works
├── notebooks/                 # For exploratory analysis, evaluation, plots
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Verify Week 1 is working

```bash
python -m pytest tests/ -v
```

Or run the end-to-end pipeline directly:

```bash
python -m agents.orchestrator_agent
```

This runs all six agents on one sample item (`item_id=1000`) using each
agent's Week 1 placeholder logic, and prints the full pipeline output —
confirming the wiring is correct before any agent's real logic is built.

## Each agent's placeholder, and what replaces it

Every agent currently has intentionally simple placeholder logic (a
moving average, a basic reorder-point rule, etc.) so the *pipeline*
works end to end starting Week 1. Each placeholder gets replaced with
real logic during its own sprint, per the Project Plan's Work Breakdown
Structure — see the `TODO (Week N)` comment at the top of each agent's
`run()` method.

## Team split (see Requirements Statement, Section 1.4)

- **Sanjana** — Demand, Inventory, Supplier Agents (input/forecasting side)
- **[Teammate]** — Optimization, Critic, Orchestrator Agents (decision side)
