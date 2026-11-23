"""Manager/Orchestrator Agent -- WBS #7.

Implements FR6 (Requirements Statement, Section 2.1):
"The system shall coordinate the sequence of agent calls (Demand ->
Inventory/Supplier -> Optimization -> Critic), assemble the final
order plan, and support a replan loop when new information requires
revisiting earlier steps."

The agents never call each other. Every hand-off goes through here:

    Supplier ──┐                                   ┌─ approve ──────────────┐
    Inventory ─┼─> Demand(horizon = lead time + 7) ─> Optimization -> Critic ┤
               │                       ^                                     └─ object: send the
               │                       └──── tighter min/max units ◄───────────  Critic's bound back
               └─ (one item at a time, up to `max_rounds` tries)
    then, for the whole week:  budget step -> Critic basket check -> final order plan

REPLAN LOOP (two kinds)
  1. Critic objection -> Optimizer retry with the Critic's bound. Stops when the Critic
     approves, when the same objection comes back (no progress), when the bounds
     contradict, or after `max_rounds`; then the item is marked needs_human_review
     with the best order so far.
  2. New information (`replan`): supplier terms changed, a physical stock count disagrees
     with the ledger, or the budget changed. Only the steps the news invalidates re-run.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from agents.base_agent import BaseAgent
from agents.critic_agent import CriticAgent
from agents.demand_agent import DemandForecastingAgent
from agents.inventory_agent import InventoryAgent
from agents.optimization_agent import CYCLE_DAYS, OptimizationAgent
from agents.supplier_agent import SupplierAgent

MAX_ROUNDS = 3


class OrchestratorAgent(BaseAgent):
    name = "orchestrator_agent"

    def __init__(self, demand_agent=None, inventory_agent=None, supplier_agent=None,
                 optimization_agent=None, critic_agent=None, max_rounds: int = MAX_ROUNDS):
        self.demand_agent = demand_agent or DemandForecastingAgent()
        self.inventory_agent = inventory_agent or InventoryAgent()
        self.supplier_agent = supplier_agent or SupplierAgent()
        self.optimization_agent = optimization_agent or OptimizationAgent()
        self.critic_agent = critic_agent or CriticAgent()
        self.max_rounds = max_rounds

    # ------------------------------------------------------------- one item
    def run(self, item_id: int, current_stock: int | None = 0, as_of=None, **kwargs) -> dict:
        """Plan one item. `current_stock` is an ad-hoc stock count (pass None to use the
        inventory ledger instead)."""
        return self.plan_item(item_id, as_of=as_of, current_stock=current_stock, **kwargs)

    def plan_item(self, item_id: int, as_of=None, current_stock: int | None = None,
                  planned_promotions=None, reuse: dict | None = None,
                  cycle_days: int = CYCLE_DAYS) -> dict:
        """Demand -> (Inventory, Supplier) -> Optimization <-> Critic loop for one item.

        `cycle_days` is how many days the order must last (7 = weekly ordering).
        `as_of` is the last day of history the forecast may use; the plan is made for the
        next day (the "decision date"). `reuse` lets a replan skip steps whose inputs
        haven't changed, e.g. {"demand_forecast": ...}.
        """
        reuse = reuse or {}
        item_id = int(item_id)
        as_of_ts = pd.Timestamp(as_of).normalize() if as_of is not None else self.demand_agent.dates[-1]
        decision = (as_of_ts + timedelta(days=1)).date()
        trace: list[dict] = []

        supplier_info = self.supplier_agent.run(item_id=item_id)
        trace.append({"agent": "supplier", "summary": supplier_info["rationale"]})
        L = supplier_info["lead_time_days"]

        inventory_status = reuse.get("inventory_status") or self.inventory_agent.run(
            item_id=item_id, as_of=decision, current_stock=current_stock)
        trace.append({"agent": "inventory", "summary": inventory_status["rationale"]})

        demand_forecast = reuse.get("demand_forecast")
        if demand_forecast is None or demand_forecast["horizon_days"] != L + cycle_days:
            demand_forecast = self.demand_agent.run(
                item_id=item_id, as_of=as_of_ts, horizon_days=L + cycle_days,
                cover_window=(L, L + cycle_days), include_baseline=False,
                planned_promotions=planned_promotions)
        trace.append({"agent": "demand", "summary": demand_forecast["rationale"]})

        constraints: dict = {}
        revisions: list[dict] = []
        plan = verdict = None
        status = "needs_human_review"
        n_rounds = 0
        for rnd in range(1, self.max_rounds + 1):
            n_rounds = rnd
            plan = self.optimization_agent.run(demand_forecast=demand_forecast,
                                               inventory_status=inventory_status,
                                               supplier_info=supplier_info,
                                               constraints=constraints,
                                               cycle_days=cycle_days)
            trace.append({"agent": "optimization", "round": rnd,
                          "summary": plan["rationale"]})
            verdict = self.critic_agent.run(optimization_result=plan,
                                            demand_forecast=demand_forecast,
                                            shelf_life_days=inventory_status["shelf_life_days"],
                                            inventory_status=inventory_status,
                                            supplier_info=supplier_info)
            trace.append({"agent": "critic", "round": rnd, "summary": verdict["rationale"]})
            if verdict["approved"]:
                status = ("approved_after_revision" if rnd > 1 else
                          "approved_with_warnings" if verdict["warnings"] else "approved")
                break
            new_c = self._merge(constraints, verdict["suggested"])
            revisions.append({"round": rnd, "units": plan["recommended_order_units"],
                              "objections": [o["code"] for o in verdict["objections"]],
                              "bounds_sent": dict(new_c)})
            lo, hi = new_c.get("min_units"), new_c.get("max_units")
            if new_c == constraints or (lo is not None and hi is not None and lo > hi):
                break                                        # no progress / contradictory
            constraints = new_c
        needs_review = status == "needs_human_review"
        final_units = int(plan["recommended_order_units"])
        return {
            "item_id": item_id,
            "decision_date": decision.isoformat(),
            "cycle_days": cycle_days,
            "demand_forecast": demand_forecast,
            "inventory_status": inventory_status,
            "supplier_info": supplier_info,
            "optimization_result": plan,
            "critic_result": verdict,
            "status": status,
            "needs_human_review": needs_review,
            "rounds": n_rounds,
            "revisions": revisions,
            "final_order_units": final_units,
            "explanation": self._explain(item_id, plan, verdict, revisions, status),
            "trace": trace,
        }

    # ------------------------------------------------------------- the week
    def plan_week(self, as_of=None, budget: float | None = None, item_ids=None,
                  planned_promotions: dict | None = None, stock_overrides: dict | None = None,
                  cycle_days: int = CYCLE_DAYS) -> dict:
        """Plan every item for the coming week and assemble ONE order plan.

        budget              weekly purchasing budget in dollars (None = unlimited)
        planned_promotions  {item_id: [dates]} promotions the restaurant has scheduled
        stock_overrides     {item_id: units} ad-hoc counts replacing the ledger
        """
        ids = [int(i) for i in (item_ids if item_ids is not None else self.demand_agent.item_ids)]
        promos = planned_promotions or {}
        overrides = stock_overrides or {}
        items = {i: self.plan_item(i, as_of=as_of, current_stock=overrides.get(i),
                                   planned_promotions=promos.get(i), cycle_days=cycle_days)
                 for i in ids}
        ctx = {"as_of": None if as_of is None else str(pd.Timestamp(as_of).date()),
               "budget": budget, "item_ids": ids, "planned_promotions": promos,
               "stock_overrides": overrides, "cycle_days": cycle_days}
        return self._assemble(items, budget, ctx, replan_log=[])

    def _assemble(self, items: dict, budget, ctx: dict, replan_log: list) -> dict:
        """Budget step + basket check + summary, from per-item plans (plan_item output).

        The pre-budget plan/verdict/status are kept on each item, so a later replan
        (e.g. the budget changing) can start again from them.
        """
        for it in items.values():
            for key, src in (("optimization_pre_budget", "optimization_result"),
                             ("critic_pre_budget", "critic_result"),
                             ("status_pre_budget", "status")):
                it.setdefault(key, it[src])
            it["trace"] = [t for t in it["trace"]
                           if t["agent"] != "budget" and t.get("round") != "post-budget"]
            it["optimization_result"] = it["optimization_pre_budget"]
            it["critic_result"] = it["critic_pre_budget"]
            it["status"] = it["status_pre_budget"]
            it["needs_human_review"] = it["status"] == "needs_human_review"
        pre = [it["optimization_pre_budget"] for it in items.values()]
        plans, report = self.optimization_agent.allocate_budget(pre, budget)
        for p in plans:
            it = items[p["item_id"]]
            if p["recommended_order_units"] != it["optimization_pre_budget"]["recommended_order_units"]:
                p = self.optimization_agent.with_quantity(
                    p, p["recommended_order_units"], it["demand_forecast"], it["inventory_status"])
                verdict = self.critic_agent.run(
                    optimization_result=p, demand_forecast=it["demand_forecast"],
                    shelf_life_days=it["inventory_status"]["shelf_life_days"],
                    inventory_status=it["inventory_status"], supplier_info=it["supplier_info"])
                it["critic_result"] = verdict
                it["trace"].append({"agent": "budget", "summary": p["rationale"].split("BUDGET:")[-1].strip()})
                it["trace"].append({"agent": "critic", "round": "post-budget",
                                    "summary": verdict["rationale"]})
                it["status"] = ("approved_with_warnings" if verdict["approved"] else "needs_human_review")
                it["needs_human_review"] = not verdict["approved"]
            it["optimization_result"] = p
            it["final_order_units"] = int(p["recommended_order_units"])
            it["explanation"] = self._explain(it["item_id"], p, it["critic_result"],
                                              it["revisions"], it["status"])
        basket = self.critic_agent.review_basket([it["optimization_result"] for it in items.values()],
                                                 budget)
        basket["budget_step"] = report
        orders = {i: it["final_order_units"] for i, it in items.items() if it["final_order_units"] > 0}
        statuses = [it["status"] for it in items.values()]
        summary = {
            "items_planned": len(items),
            "items_ordered": len(orders),
            "total_units": int(sum(orders.values())),
            "total_cost": basket["total_cost"],
            "approved_first_pass": statuses.count("approved") + statuses.count("approved_with_warnings"),
            "approved_after_revision": statuses.count("approved_after_revision"),
            "needs_human_review": statuses.count("needs_human_review"),
            "with_warnings": sum(1 for it in items.values() if it["critic_result"]["warnings"]),
            "budget_ok": basket["approved"],
        }
        d = next(iter(items.values()))["decision_date"] if items else None
        return {"decision_date": d, "context": ctx, "items": items, "orders": orders,
                "basket": basket, "summary": summary, "replan_log": replan_log}

    # -------------------------------------------------- replan on new information
    def replan(self, week_plan: dict, update: dict) -> dict:
        """Revise a week plan after new information. `update` is one of:
          {"type": "supplier_update", "item_id": n, "changes": {"moq_units": 80, ...}, "reason": "..."}
          {"type": "stock_correction", "item_id": n, "count": units}
          {"type": "budget_change", "budget": dollars}
        Only the steps the news invalidates are re-run; the log says which.
        """
        ctx = dict(week_plan["context"])
        items = {i: dict(it, trace=list(it["trace"])) for i, it in week_plan["items"].items()}
        before = {i: it["final_order_units"] for i, it in items.items()}
        cost_before = week_plan["basket"]["total_cost"]
        kind = update["type"]
        as_of = ctx["as_of"]
        rerun: list[str]
        if kind == "supplier_update":
            i = int(update["item_id"])
            rec = self.supplier_agent.apply_update(i, reason=update.get("reason", "supplier update"),
                                                   **update["changes"])
            lead_changed = "lead_time_days" in rec["changes"]
            reuse = {"inventory_status": items[i]["inventory_status"]}
            if not lead_changed:
                reuse["demand_forecast"] = items[i]["demand_forecast"]
            items[i] = self.plan_item(i, as_of=as_of, planned_promotions=ctx["planned_promotions"].get(i),
                                      reuse=reuse, cycle_days=ctx["cycle_days"])
            rerun = (["supplier"] + (["demand"] if lead_changed else []) +
                     ["optimization", "critic", "budget"])
            what = f"item {i}: " + ("; ".join(f"{k} {v['old']} -> {v['new']}" for k, v in rec["changes"].items())
                                    or "no change")
        elif kind == "stock_correction":
            i = int(update["item_id"])
            decision = items[i]["decision_date"]
            adj = self.inventory_agent.set_count(i, int(update["count"]), decision)
            items[i] = self.plan_item(i, as_of=as_of, planned_promotions=ctx["planned_promotions"].get(i),
                                      reuse={"demand_forecast": items[i]["demand_forecast"]},
                                      cycle_days=ctx["cycle_days"])
            rerun = ["inventory", "optimization", "critic", "budget"]
            what = (f"item {i}: count {adj['counted_units']} vs ledger {adj['ledger_units']} "
                    f"({adj['adjustment']:+d})")
        elif kind == "budget_change":
            ctx["budget"] = update["budget"]
            rerun = ["budget", "critic (basket)"]
            what = f"budget -> ${update['budget']:,.0f}"
        else:
            raise ValueError(f"Unknown update type '{kind}'.")
        new = self._assemble(items, ctx["budget"], ctx, replan_log=list(week_plan["replan_log"]))
        changed = [{"item_id": i, "before": before[i], "after": it["final_order_units"]}
                   for i, it in new["items"].items() if it["final_order_units"] != before[i]]
        new["replan_log"].append({"trigger": kind, "detail": what, "steps_rerun": rerun,
                                  "orders_changed": changed, "cost_before": cost_before,
                                  "cost_after": new["basket"]["total_cost"]})
        return new

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _merge(current: dict, suggested: dict) -> dict:
        out = dict(current)
        if "max_units" in suggested:
            out["max_units"] = min(out.get("max_units", suggested["max_units"]), suggested["max_units"])
        if "min_units" in suggested:
            out["min_units"] = max(out.get("min_units", suggested["min_units"]), suggested["min_units"])
        return out

    @staticmethod
    def _explain(item_id, plan, verdict, revisions, status) -> str:
        s = plan["rationale"]
        if revisions:
            path = "; ".join(f"round {r['round']}: {r['units']} units -> Critic objected "
                             f"({', '.join(r['objections'])})" for r in revisions)
            s += f" REPLAN: {path}."
        s += f" CRITIC [{status}]: {verdict['rationale']}"
        return s


if __name__ == "__main__":
    import json
    r = OrchestratorAgent().run(item_id=1000, current_stock=10)
    print(json.dumps({k: r[k] for k in ("item_id", "status", "final_order_units", "explanation")},
                     indent=2))
