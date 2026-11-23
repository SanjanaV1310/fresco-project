"""Ordering policies the simulator can replay.

  FrescoPolicy            the six-agent system (Orchestrator.plan_week)
  ReorderPointBaseline    the SOW baseline: "simple reorder-point rules" -- the
                          Week 1 placeholder logic, applied weekly
  OrderUpToBaseline       a stronger non-agent baseline a careful manager might run

Both baselines are handed the full-year average and spread of daily demand from
inventory_params.csv. That includes the future, which a real manager wouldn't
have, so the baselines are given a small advantage -- not FRESCO.
"""
from __future__ import annotations

import math
from statistics import NormalDist

from agents.data_loader import load_inventory_params
from agents.orchestrator_agent import OrchestratorAgent
from agents.optimization_agent import OptimizationAgent

_ND = NormalDist()


def _position(ledger, item) -> int:
    """Units on the shelf plus units already on order."""
    return (sum(b["units"] for b in ledger.batches[item])
            + sum(o["units"] for o in ledger.in_transit[item]))


class FrescoPolicy:
    def __init__(self, budget: float | None = None, goodwill: float | None = None,
                 max_rounds: int = 3, name: str = "FRESCO"):
        self.budget, self.goodwill, self.max_rounds, self.name = budget, goodwill, max_rounds, name
        self.log: list[dict] = []
        self._orch = None

    def reset(self):
        self.log, self._orch = [], None

    def decide(self, ctx: dict) -> dict:
        if self._orch is None or self._orch.inventory_agent is not ctx["inventory"]:
            opt = OptimizationAgent(**({} if self.goodwill is None else {"goodwill": self.goodwill}))
            self._orch = OrchestratorAgent(demand_agent=ctx["demand_agent"],
                                           inventory_agent=ctx["inventory"],
                                           supplier_agent=ctx["supplier"],
                                           optimization_agent=opt, max_rounds=self.max_rounds)
        plan = self._orch.plan_week(as_of=ctx["as_of"], budget=self.budget,
                                    planned_promotions=ctx["planned_promotions"],
                                    cycle_days=ctx["cycle_days"])
        s = plan["summary"]
        self.log.append({"week": ctx["week"], "decision_date": ctx["decision_date"].isoformat(),
                         **s, "total_cost": plan["basket"]["total_cost"],
                         "revisions": sum(len(it["revisions"]) for it in plan["items"].values())})
        return plan["orders"]


class _StaticStats:
    def __init__(self):
        p = load_inventory_params().set_index("item_id")
        self.mean = p["mean_daily_demand"].to_dict()
        self.std = p["std_daily_demand"].to_dict()
        self.rop = p["reorder_point_units"].to_dict()
        self.shelf = p["shelf_life_days"].to_dict()


class ReorderPointBaseline(_StaticStats):
    """On each ordering day, if (on hand + on order) is at or below the reorder point, order
    up to the reorder point plus one week of average demand (at least the MOQ)."""
    name = "Reorder-point rule (SOW baseline)"

    def decide(self, ctx: dict) -> dict:
        led, sup = ctx["inventory"], ctx["supplier"]
        orders = {}
        for i in self.mean:
            pos = _position(led, i)
            if pos <= self.rop[i]:
                target = self.rop[i] + ctx["cycle_days"] * self.mean[i]
                qty = max(int(math.ceil(target - pos)), int(sup.terms[i]["moq_units"]))
                orders[i] = qty
        return orders


class OrderUpToBaseline(_StaticStats):
    """On each ordering day order up to  mean x (lead time + cycle) + z x std x sqrt(lead time + cycle),
    using full-year average demand. Capped at what the average day can sell before the
    shelf life runs out, so it is not naive about spoilage."""

    def __init__(self, z: float = 1.645, shelf_cap: bool = True, name: str | None = None):
        super().__init__()
        self.z, self.shelf_cap = z, shelf_cap
        self.name = name or f"Order-up-to rule (z={z:g})"

    def decide(self, ctx: dict) -> dict:
        led, sup = ctx["inventory"], ctx["supplier"]
        orders = {}
        for i in self.mean:
            terms = sup.terms[i]
            H = int(terms["lead_time_days"]) + ctx["cycle_days"]
            target = self.mean[i] * H + self.z * self.std[i] * math.sqrt(H)
            qty = target - _position(led, i)
            if self.shelf_cap and self.shelf[i] < H:
                qty = min(qty, self.mean[i] * self.shelf[i])
            if qty > 0:
                orders[i] = max(int(math.ceil(qty)), int(terms["moq_units"]))
        return orders
