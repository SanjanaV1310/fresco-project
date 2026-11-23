# FRESCO — Multi-Agent Inventory & Waste Reduction System

CAP 6942 Capstone Project — Sanjana Vijayabhaskar & Akila Kumar

A six-agent system that automates the restaurant ordering decision (what to order, how much,
from whom) while balancing demand uncertainty, perishability, supplier constraints and budget.
Everything runs offline on synthetic data (NFR4).

## Quick start

```bash
pip install -r requirements.txt
python demo.py                          # ~10 s: the Final Demo walkthrough (5 scenes)
python -m webapp.server                 # class demo website -> http://localhost:8000 (no internet needed)
python -m pytest -q                     # ~15 s: 104 tests
python scripts/evaluate_system.py       # ~70 s: full evaluation -> reports/system_evaluation.*
python scripts/evaluate_demand_agent.py # Week 2 forecast backtest -> reports/demand_agent_evaluation.*
```

## Status: all nine work packages done

| # | Work package | Where | Status |
|---|---|---|---|
| 1 | Environment & data | `data/`, `agents/data_loader.py` | done |
| 2 | Demand Forecasting Agent (FR1) | `agents/demand_agent.py`, `forecasting.py` | done |
| 3 | Inventory Agent (FR2) | `agents/inventory_agent.py` | done |
| 4 | Supplier Agent (FR3) | `agents/supplier_agent.py` | done |
| 5 | Optimization Agent (FR4) | `agents/optimization_agent.py` | done |
| 6 | Critic Agent (FR5) | `agents/critic_agent.py` | done |
| 7 | Orchestrator + replan loop (FR6) | `agents/orchestrator_agent.py` | done |
| 8 | Simulation / evaluation (FR7) | `simulation/`, `scripts/evaluate_system.py` | done |
| 9 | Demo | `demo.py` (final report/slides still to write) | demo done |

## How it works

```
Supplier ─┐
Inventory ─┼─> Demand (horizon = lead time + cycle) ─> Optimization ─> Critic ─┬─ approve
           │                       ^                                          └─ object: send the Critic's
           └── per item, up to 3 tries <──── tighter min/max units ◄────────────  min/max bound back
then, for the whole basket:  budget step -> Critic basket check -> FINAL ORDER PLAN
new information (supplier terms / stock count / budget) -> REPLAN only the affected steps
```

Agents never call each other; every hand-off goes through the Orchestrator (tested).

- **Demand** — recent level x weekday pattern x promo lift, 80% ranges from the item's own backtest errors, `as_of` replay with no peeking at the future.
- **Inventory** — FIFO batch ledger with shelf life counted from delivery, in-transit orders, expiry flags, configurable safety stock, physical-count corrections.
- **Supplier** — price, lead time, MOQ, availability per item; logged updates; quotes.
- **Optimization** — order covers days `lead time .. lead time + cycle - 1`. Target service level from the cost of being short vs. long (perishables near the forecast, durables with a cushion). Nets stock day by day including expiry, then applies: shelf-life cap, safety-stock floor, supplier shortfall gross-up, MOQ economics (order the minimum only if margin beats expected waste), Critic bounds. Basket budget: shrink cushions first, then drop the least profitable per dollar.
- **Critic** — re-projects the order under low / forecast / high demand *independently of the Optimizer*. Objects to: invalid quantity, expiry at forecast demand, waste if demand is low, understock that ordering more would actually fix. Says exactly what to change. Unavoidable trade-offs (shelf life, budget, MOQ) are approved **with a warning**, not argued forever.
- **Orchestrator** — per-item loop (max 3 rounds, stops on no progress/contradiction, then `needs_human_review`), weekly `plan_week`, and `replan` for supplier updates, stock corrections and budget changes.
- **Simulation** — day-by-day stockroom replay with short deliveries, FIFO sales, expiry waste; common random numbers; unit-conservation check; two baselines; dev/holdout split; paired bootstrap.

## Results (holdout = last 15 of 32 replayed weeks; details in `reports/system_evaluation.md`)

| Scenario | Policy | Demand served | Waste | Margin after waste (15 wks) |
|---|---|---|---|---|
| Weekly (SOW) | SOW reorder-point rule | 63.6% | 40.5% | -$42,570 |
| | Order-up-to rule | 73.0% | 3.2% | $110,925 |
| | **FRESCO** | 73.2% | 3.0% | $112,102 |
| Twice-weekly | SOW reorder-point rule | 70.3% | 19.1% | $49,771 |
| | Order-up-to rule | 93.1% | 5.2% | $136,508 |
| | **FRESCO** | 91.9% | 2.5% | $144,518 |

- Acceptance criterion met: large improvement over the SOW baseline on waste and stockouts.
- Against the stronger rule: a **small, not clearly reliable** edge with weekly ordering (+1.1%); a **clear** one with twice-weekly ordering (+5.9% margin, -53% waste cost).
- Main insight: short-shelf-life items can't be served by one weekly delivery. Weekly ordering serves only ~73% of demand for any policy.

## Limitations (see the report for the full list)
Synthetic data; one year of history (holiday bump not replayed); baselines get full-year statistics (a small advantage to them); FRESCO knows planned promotions; simple supplier model (one supplier per item, fixed lead times/prices); twice-weekly ordering extends the SOW's weekly cadence; the Critic rarely objects in normal operation (3 of 54 items in the demo's twice-weekly plan, 169 of 3,456 in the evaluation) and its objections are small adjustments.

## Structure

```
fresco-project/
├── demo.py                   # Final demo: python demo.py
├── webapp/                   # Local demo website (stdlib only): python -m webapp.server
├── data/                     # items, demand, inventory_params, suppliers, DATA_DICTIONARY.md, generate.py
├── agents/                   # the six agents + forecasting.py, inventory_math.py (shared projection), data_loader.py
├── simulation/               # environment.py, policies.py (FRESCO + baselines), metrics.py
├── scripts/                  # evaluate_demand_agent.py, evaluate_system.py
├── reports/                  # demand + system evaluation (.md, .png, data)
├── tests/                    # 104 tests: per agent, orchestrator/replan, simulator, week-1 smoke
├── notebooks/
├── requirements.txt
└── README.md
```

## Requirement traceability

| Requirement | Evidence |
|---|---|
| FR1 forecast | `demand_agent.py`; `tests/test_demand_agent.py`; `reports/demand_agent_evaluation.md` |
| FR2 inventory, safety stock, expiry flags | `inventory_agent.py`; `tests/test_inventory_supplier.py` |
| FR3 supplier terms | `supplier_agent.py`; `tests/test_inventory_supplier.py` |
| FR4 order quantity under budget/MOQ/lead time/safety stock | `optimization_agent.py`; `tests/test_optimization_critic.py` |
| FR5 approve or specific objection | `critic_agent.py`; clear-bad and clear-good order tests |
| FR6 coordination + replan | `orchestrator_agent.py`; `tests/test_orchestrator.py`; demo scenes 3-4 |
| FR7 simulate and report waste/stockout/cost | `simulation/`; `tests/test_simulation.py` |
| NFR1 explainability | every agent returns `rationale`; `explanation` + `trace` per item |
| NFR2 modularity | no agent imports another (tested); each tested alone |
| NFR3 runtime | full 54-item week plans in ~1 s; demo ~10 s |
| NFR4 synthetic data | `data/` |
