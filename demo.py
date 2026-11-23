"""FRESCO end-to-end demo (Final Demo, Dec 8).  Run:  python demo.py   (about 20 seconds)

Scenes
  1. One item through all six agents, with each agent explaining itself
  2. The full weekly order plan for 54 items
  3. The Critic objects -> the Orchestrator replans -> the Critic approves
  4. New information arrives mid-week -> only the affected steps re-run
  5. How the plan performs vs. the baselines when replayed over 32 real weeks
"""
from __future__ import annotations

import argparse
import textwrap
import time
from datetime import timedelta

import pandas as pd

from agents.data_loader import load_items
from agents.demand_agent import DemandForecastingAgent
from agents.inventory_agent import InventoryAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.supplier_agent import SupplierAgent
from simulation.environment import Simulator
from simulation.metrics import summarize, weekly_table
from simulation.policies import FrescoPolicy, OrderUpToBaseline, ReorderPointBaseline

AS_OF = "2026-09-20"          # a Sunday; the plan is for Monday 2026-09-21
NAMES = load_items().set_index("item_id")["item_name"].to_dict()


def wrap(s, indent=6):
    return textwrap.fill(s, width=100, initial_indent=" " * indent, subsequent_indent=" " * indent)


def banner(n, title):
    print("\n" + "=" * 100 + f"\n SCENE {n}: {title}\n" + "=" * 100)


def stocked_ledger(demand_agent) -> InventoryAgent:
    """Shelves as they might look on Sunday night: about two days of average sales on hand,
    delivered the day before (so perishables are already part-way through their life)."""
    inv = InventoryAgent()
    t = demand_agent._cutoff_index(AS_OF)
    for r, i in enumerate(demand_agent.item_ids):
        two_days = int(round(demand_agent.Y[r, t - 28:t].mean() * 2))
        inv.receive(int(i), two_days, pd.Timestamp(AS_OF).date())
    return inv


def main():
    t0 = time.time()
    demand = DemandForecastingAgent()
    ledger = stocked_ledger(demand)
    orch = OrchestratorAgent(demand_agent=demand, inventory_agent=ledger, supplier_agent=SupplierAgent())
    print("FRESCO - Multi-Agent Inventory & Waste Reduction System")
    print(f"Planning the week starting Monday {pd.Timestamp(AS_OF).date() + timedelta(days=1)} "
          f"from history up to {AS_OF}. 54 items, one supplier each.")

    # ------------------------------------------------------------------ 1
    banner(1, "One item through the six agents (item 1004, a fresh-produce item)")
    item = 1004
    r = orch.plan_item(item, as_of=AS_OF)
    print(f"  {NAMES[item]}")
    for step in r["trace"]:
        label = {"supplier": "SUPPLIER", "inventory": "INVENTORY", "demand": "DEMAND",
                 "optimization": f"OPTIMIZATION (round {step.get('round')})",
                 "critic": f"CRITIC (round {step.get('round')})"}[step["agent"]]
        print(f"\n   [{label}]\n" + wrap(step["summary"]))
    print(f"\n   ORCHESTRATOR -> final order: {r['final_order_units']} units, status: {r['status']}")

    # ------------------------------------------------------------------ 2
    banner(2, "The whole week: one coordinated order plan")
    week = orch.plan_week(as_of=AS_OF)
    s = week["summary"]
    print(f"  {s['items_ordered']} of {s['items_planned']} items ordered, {s['total_units']:,} units, "
          f"${s['total_cost']:,.0f} total.")
    print(f"  Critic: {s['approved_first_pass']} approved first time ({s['with_warnings']} with warnings), "
          f"{s['approved_after_revision']} after a revision, {s['needs_human_review']} need a human.")
    rows = sorted(week["items"].values(), key=lambda it: -it["optimization_result"]["order_cost"])[:8]
    print("\n   Largest orders:")
    print(f"   {'item':<16}{'order':>7}{'cost':>10}   {'status':<24}arrives in")
    for it in rows:
        o = it["optimization_result"]
        print(f"   {NAMES[it['item_id']]:<16}{it['final_order_units']:>7}{'$' + format(o['order_cost'], ',.0f'):>10}"
              f"   {it['status']:<24}{o['lead_time_days']} days")
    short = [it for it in week["items"].values() if it["optimization_result"]["shelf_capped"]]
    print(f"\n   {len(short)} short-shelf-life items are capped by shelf life: ordering a full week would "
          f"only create waste, so late-week demand goes unserved. (Scene 5 shows what that costs and what fixes it.)")
    ex = short[0]
    print(wrap(f"Example, {NAMES[ex['item_id']]}: " + ex["critic_result"]["rationale"], 3))

    # ------------------------------------------------------------------ 3
    banner(3, "The Critic objects, the Orchestrator replans")
    print("  Twice-weekly plan (orders for the 3 days to Thursday) - looking for an item where the "
          "first proposal was challenged:")
    tw = orch.plan_week(as_of=AS_OF, cycle_days=3)
    hit = max((it for it in tw["items"].values() if it["revisions"]), default=None,
              key=lambda it: it["revisions"][0]["units"] - it["final_order_units"])
    if hit:
        print(f"\n   {NAMES[hit['item_id']]}: {hit['rounds']} rounds, final status {hit['status']}")
        for step in hit["trace"]:
            if step["agent"] in ("optimization", "critic"):
                print(f"\n   [{step['agent'].upper()} round {step['round']}]\n" + wrap(step["summary"]))
        print(f"\n   Order went {hit['revisions'][0]['units']} -> {hit['final_order_units']} units.")
    else:
        print("   (no objection this week; the unit tests inject a bad proposal to cover this path)")
    n_rev = sum(1 for it in tw["items"].values() if it["revisions"])
    print(f"\n   {n_rev} of {len(tw['items'])} items needed a revision; "
          f"{sum(it['needs_human_review'] for it in tw['items'].values())} escalated to a human.")

    # ------------------------------------------------------------------ 4
    banner(4, "New information arrives - replan only what it affects")
    big = max(week["items"].values(), key=lambda it: it["optimization_result"]["order_cost"])["item_id"]
    old = orch.supplier_agent.run(big)
    steps = [
        {"type": "supplier_update", "item_id": big,
         "changes": {"moq_units": max(old["moq_units"] * 7, 300), "unit_cost": round(old["unit_cost"] * 1.15, 2)},
         "reason": "supplier raised minimum order and price"},
        {"type": "stock_correction", "item_id": 1004, "count": 0},
        {"type": "budget_change", "budget": round(week["summary"]["total_cost"] * 0.75)},
    ]
    cur = week
    for u in steps:
        cur = orch.replan(cur, u)
        lg = cur["replan_log"][-1]
        print(f"\n   NEWS: {lg['trigger']} - {lg['detail']}")
        print(f"   Re-ran: {' -> '.join(lg['steps_rerun'])}")
        for c in lg["orders_changed"][:5]:
            print(f"     {NAMES[c['item_id']]}: {c['before']} -> {c['after']} units")
        if len(lg["orders_changed"]) > 5:
            print(f"     ... and {len(lg['orders_changed']) - 5} more")
        print(f"   Basket cost ${lg['cost_before']:,.0f} -> ${lg['cost_after']:,.0f}")
    print(f"\n   Final basket respects budget: {cur['basket']['approved']} ({cur['basket']['rationale']})")
    ex = cur["items"][big]
    print(wrap("Explanation for the item whose supplier changed: " + ex["explanation"], 3))

    # ------------------------------------------------------------------ 5
    banner(5, "Does it work? Replay 32 real weeks against the baselines")
    sim = Simulator(seed=0)
    hold = slice(17, sim.max_weeks())
    for label, rd in (("WEEKLY ordering (the SOW scenario)", (0,)), ("TWICE-WEEKLY ordering (Mon + Thu)", (0, 3))):
        print(f"\n   {label}, last 15 weeks:")
        print(f"   {'policy':<28}{'served':>8}{'waste':>8}{'margin after waste':>22}")
        for name, pol in (("SOW reorder-point rule", ReorderPointBaseline()),
                          ("Order-up-to rule", OrderUpToBaseline()), ("FRESCO", FrescoPolicy())):
            m = summarize(weekly_table(sim.run(pol, review_days=rd)).iloc[hold])
            print(f"   {name:<28}{m['fill_rate']:>8.1%}{m['waste_rate']:>8.1%}{'$' + format(m['margin_after_waste'], ',.0f'):>22}")
    print("\n   Full evaluation, confidence intervals and limitations: reports/system_evaluation.md")
    print(f"\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    main()
