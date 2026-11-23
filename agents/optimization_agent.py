"""Optimization Agent -- WBS #5.

Implements FR4 (Requirements Statement, Section 2.1):
"The system shall compute a recommended order quantity per SKU that satisfies
budget, MOQ, lead-time, and safety-stock constraints while minimizing expected
waste and stockout risk."

Per Risk R6 the model stays simple enough to explain in one paragraph. For one
item it answers: "how many units should I order today so they arrive after the
lead time and last through the next ordering cycle?"

  1. COVER WINDOW. An order placed today arrives on day L (the lead time) and must
     last until the next order arrives: days L .. L+cycle-1 (7 days for a weekly order).
  2. HOW MUCH SERVICE IS WORTH BUYING. Each unit short costs its margin (price -
     cost, plus a goodwill penalty); each unit left over costs its purchase price
     *if it expires before it can be sold*. Perishables (shelf life <= the ordering cycle)
     are wasted when left over; long-life items just sit on the shelf. The ratio of
     the two costs sets a target service level (the "newsvendor" rule), so fragile
     items are ordered near the forecast and durable items with a cushion.
  3. QUANTITY. Demand over the window is the forecast, inflated for the target
     service level using the forecast's own measured error. Stock already on hand
     (oldest first, minus whatever expires) and orders already on their way are
     netted off day by day, because stock that expires first can't cover later days.
  4. CONSTRAINTS, in this order: shelf-life cap (never order more than can be sold
     before the new batch expires) -> safety-stock floor (only for items durable
     enough to hold) -> supplier shortfall gross-up (ordered units / expected fill)
     -> minimum order quantity (order the MOQ only if the margin on the units we
     need beats the expected waste on the extras, else skip) -> bounds sent back by
     the Critic (min_units / max_units).
  5. BASKET BUDGET (`allocate_budget`). If the whole basket costs more than the
     weekly budget, safety cushions shrink first, then the least profitable items
     per dollar are dropped.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from agents.base_agent import BaseAgent
from agents.inventory_math import (CYCLE_DAYS, DEFAULT_WINDOW_ERROR, extend_forecast,
                                   project_fifo)

GOODWILL = 0.25             # extra cost of a stockout, as a fraction of the retail price
SL_MIN, SL_MAX = 0.50, 0.97  # target service-level bounds
DURABLE_DAYS = 14           # shelf life from which holding safety stock is sensible
_ND = NormalDist()


def waste_risk(shelf_life_days: int, cycle: int = CYCLE_DAYS) -> float:
    """Chance a leftover unit is eventually thrown away rather than sold:
    1 for anything that expires within the ordering cycle, decaying for longer-life items."""
    if shelf_life_days <= cycle:
        return 1.0
    return max(math.exp(-(shelf_life_days - cycle) / cycle), 0.02)


class OptimizationAgent(BaseAgent):
    name = "optimization_agent"

    def __init__(self, goodwill: float = GOODWILL, sl_min: float = SL_MIN,
                 sl_max: float = SL_MAX):
        self.goodwill, self.sl_min, self.sl_max = goodwill, sl_min, sl_max

    # ------------------------------------------------------------------ run
    def run(self, demand_forecast: dict, inventory_status: dict, supplier_info: dict,
            constraints: dict | None = None, cycle_days: int = CYCLE_DAYS, **kwargs) -> dict:
        """constraints (optional, from the Critic's objection): {"max_units": n, "min_units": n}
        cycle_days  days until the NEXT order can arrive (7 = one order a week; the order
                    placed now has to cover that many days)."""
        cycle = int(cycle_days)
        constraints = constraints or {}
        item_id = int(demand_forecast["item_id"])
        L = int(supplier_info["lead_time_days"])
        fill = float(supplier_info.get("expected_fill", supplier_info["availability_pct"]))
        moq = int(supplier_info["moq_units"])
        cost = float(supplier_info["unit_cost"])
        price = float(inventory_status["unit_price"])
        shelf = int(inventory_status["shelf_life_days"])
        ss = int(inventory_status.get("safety_stock_units", 0))
        notes: list[str] = []
        binding: list[str] = []

        if demand_forecast["forecast_start"] != inventory_status["as_of"]:
            raise ValueError(
                f"Item {item_id}: forecast starts {demand_forecast['forecast_start']} but the "
                f"inventory snapshot is for {inventory_status['as_of']}. Re-run both for the "
                f"same decision date.")

        need_days = L + cycle
        f_raw = [d["units"] for d in demand_forecast["daily_forecast"]]
        f = extend_forecast(f_raw, need_days)
        if len(f_raw) < need_days:
            notes.append(f"forecast only had {len(f_raw)} days, repeated its weekly pattern "
                         f"to reach {need_days}")
        win = slice(L, L + cycle)
        window_mean = float(f[win].sum())

        rel = (demand_forecast.get("error_scale") or {}).get("window_relative")
        low_conf = rel is None
        if low_conf:
            rel = DEFAULT_WINDOW_ERROR
            notes.append(f"no forecast track record, assumed +/-{rel:.0%} error")

        # --- 2. target service level from the cost of being short vs. long ----
        w = waste_risk(shelf, cycle)
        under = (price - cost) + self.goodwill * price
        over = cost * w
        cr = under / (under + over)
        sl = float(min(max(cr, self.sl_min), self.sl_max))
        z = _ND.inv_cdf(sl)
        k = 1.0 + z * rel

        # --- 3. net what we already have against the scaled demand -------------
        existing = [(b["units"], b["last_offset"]) for b in inventory_status["batches"]]
        arrivals = [(max(o["offset"], 0), o["units"] * fill)
                    for o in inventory_status["in_transit"]]
        proj_q = project_fifo(existing, arrivals, k * f, shelf)
        proj_m = project_fifo(existing, arrivals, f, shelf)
        need_q = float(proj_q["unmet"][win].sum())
        need_m = float(proj_m["unmet"][win].sum())
        q_target = need_q

        # --- 4a. shelf-life cap ------------------------------------------------
        cap_days = min(shelf, cycle)
        sellable = float(proj_m["unmet"][L:L + cap_days].sum())
        shelf_capped = shelf < cycle and q_target > sellable + 1e-9
        gap_units = max(need_m - sellable, 0.0) if shelf < cycle else 0.0
        if shelf_capped:
            q_target = sellable
            binding.append("shelf_life_cap")
            notes.append(f"shelf life is {shelf} day(s): a new batch can only be sold on its "
                         f"first {shelf} day(s) of the {cycle}-day window, so ordering for the rest "
                         f"would only create waste (~{gap_units:.0f} units of late-week demand "
                         f"go unserved; an extra delivery mid-cycle would fix this)")

        # --- 4b. safety-stock floor (only where holding stock is sensible) ------
        if shelf >= DURABLE_DAYS:
            q_ss = max(0.0, ss - proj_m["leftover"])
            if need_m + q_ss > q_target + 1e-9:
                q_target = need_m + q_ss
                binding.append("safety_stock")
                notes.append(f"raised to keep {ss} units of safety stock at the end of the cycle")
        elif ss > 0:
            notes.append(f"safety stock of {ss} not held: items that expire in {shelf} days "
                         f"would just be wasted")

        # --- 4c. supplier shortfall gross-up --------------------------------------
        gross = int(math.ceil(q_target / fill - 1e-9)) if q_target > 1e-9 else 0

        # --- 4d. minimum order quantity -------------------------------------------
        if 0 < gross < moq:
            sold_part = min(q_target, need_m) if need_m > 0 else q_target
            extra = moq * fill - q_target
            gain = (price - cost) * sold_part
            loss = cost * extra * w
            if gain >= loss:
                gross = moq
                binding.append("moq_raise")
                notes.append(f"need {q_target:.0f} but the minimum order is {moq}: margin on "
                             f"the units we need (${gain:.0f}) beats expected waste on the "
                             f"{extra:.0f} extra (${loss:.0f}), so order {moq}")
            else:
                gross = 0
                binding.append("moq_skip")
                notes.append(f"need {q_target:.0f} but the minimum order is {moq}: expected waste "
                             f"on the {extra:.0f} extra units (${loss:.0f}) exceeds the margin "
                             f"we would earn (${gain:.0f}), so skip this week")

        # --- 4e. bounds from the Critic ---------------------------------------------
        lo, hi = constraints.get("min_units"), constraints.get("max_units")
        if lo is not None and gross < lo:
            gross = int(lo)
            binding.append("min_units")
            notes.append(f"raised to the Critic's minimum of {lo}")
        if hi is not None and gross > hi:
            gross = int(hi)
            binding.append("max_units")
            notes.append(f"cut to the Critic's maximum of {hi}")
        if 0 < gross < moq:
            gross = 0
            binding.append("moq_skip")
            notes.append(f"after the Critic's bound the order fell under the {moq}-unit minimum, "
                         f"so no order")

        # --- base quantity (forecast only, no cushion): used by the budget step -----
        base = int(math.ceil(min(need_m, sellable if shelf < cycle else need_m) / fill - 1e-9))
        base = min(base, gross)
        if base > 0:
            base = max(base, moq)
        base = min(base, gross)

        # --- expected outcome of the final plan (mean-forecast scenario) -----------
        out = self._expected(existing, arrivals, gross * fill, L, f, shelf, proj_m, win)
        covers = (L + cycle - 1)
        rationale = self._rationale(item_id, gross, L, cycle, shelf, window_mean, sl, rel, need_m,
                                    inventory_status, out, notes, low_conf)
        return {
            "item_id": item_id,
            "recommended_order_units": int(gross),
            "base_units": int(base),
            "unit_cost": cost,
            "unit_price": price,
            "moq_units": moq,
            "order_cost": round(gross * cost, 2),
            "lead_time_days": L,
            "cycle_days": cycle,
            "expected_fill": fill,
            "shelf_life_days": shelf,
            "arrives_offset": L,
            "covers_through_offset": covers,
            "service_level_target": round(sl, 3),
            "window_demand": {"forecast": round(window_mean, 1),
                              "planned_for": round(float(k * f[win].sum()), 1),
                              "relative_error": round(float(rel), 3)},
            "net_need_units": round(need_q, 1),
            "binding_constraints": binding,
            "shelf_capped": bool(shelf_capped),
            "late_week_gap_units": round(gap_units, 1),
            "low_confidence": bool(low_conf),
            "expected": out,
            "constraints_in": {k_: v for k_, v in constraints.items() if v is not None},
            "rationale": rationale,
        }

    # -------------------------------------------------------- basket budget
    def allocate_budget(self, plans: list[dict], budget: float | None) -> tuple[list[dict], dict]:
        """Fit a whole basket of item plans inside a weekly budget.

        1. Shrink every item's safety cushion together (all the way to a
           forecast-only order if needed).
        2. If still over, drop the items that earn the least margin per dollar.
        Returns (new_plans, report). Input plans are not modified.
        """
        plans = [dict(p, binding_constraints=list(p["binding_constraints"])) for p in plans]
        total = sum(p["order_cost"] for p in plans)
        report = {"budget": budget, "cost_before": round(total, 2), "cost_after": round(total, 2),
                  "action": "none", "lambda": 1.0, "dropped": []}
        if budget is None or total <= budget + 1e-9:
            return plans, report

        def qty(p, lam):
            q = p["base_units"] + lam * (p["recommended_order_units"] - p["base_units"])
            q = int(math.ceil(q - 1e-9))
            if q > 0:
                q = max(q, p["moq_units"])
            return min(q, p["recommended_order_units"])

        def cost_at(lam):
            return sum(qty(p, lam) * p["unit_cost"] for p in plans)

        if cost_at(0.0) <= budget + 1e-9:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if cost_at(mid) <= budget + 1e-9 else (lo, mid)
            lam = lo
            report.update(action="trimmed_cushions", **{"lambda": round(lam, 3)})
            for p in plans:
                q = qty(p, lam)
                if q != p["recommended_order_units"]:
                    p["pre_budget_units"] = p["recommended_order_units"]
                    p["recommended_order_units"] = q
                    p["binding_constraints"].append("budget_trimmed")
        else:
            report.update(action="dropped_items", **{"lambda": 0.0})
            for p in plans:
                q = qty(p, 0.0)
                if q != p["recommended_order_units"]:
                    p["pre_budget_units"] = p["recommended_order_units"]
                    p["recommended_order_units"] = q
                    p["binding_constraints"].append("budget_trimmed")
            order = sorted((p for p in plans if p["recommended_order_units"] > 0),
                           key=lambda p: (p["unit_price"] - p["unit_cost"]) / p["unit_cost"])
            running = sum(p["recommended_order_units"] * p["unit_cost"] for p in plans)
            for p in order:
                if running <= budget + 1e-9:
                    break
                running -= p["recommended_order_units"] * p["unit_cost"]
                p.setdefault("pre_budget_units", p["recommended_order_units"])
                p["recommended_order_units"] = 0
                p["binding_constraints"].append("budget_dropped")
                report["dropped"].append(p["item_id"])
        for p in plans:
            p["order_cost"] = round(p["recommended_order_units"] * p["unit_cost"], 2)
            if "pre_budget_units" in p:
                p["rationale"] += (f" BUDGET: basket was over the ${budget:,.0f} budget, so this "
                                   f"order went from {p['pre_budget_units']} to "
                                   f"{p['recommended_order_units']} units.")
        report["cost_after"] = round(sum(p["order_cost"] for p in plans), 2)
        return plans, report

    def with_quantity(self, plan: dict, units: int, demand_forecast: dict,
                      inventory_status: dict) -> dict:
        """Return a copy of `plan` with a different order size (e.g. after the budget
        step) and its expected outcome recomputed to match."""
        L, shelf, fill = plan["lead_time_days"], plan["shelf_life_days"], plan["expected_fill"]
        cycle = int(plan["cycle_days"])
        f = extend_forecast([d["units"] for d in demand_forecast["daily_forecast"]], L + cycle)
        existing = [(b["units"], b["last_offset"]) for b in inventory_status["batches"]]
        arrivals = [(max(o["offset"], 0), o["units"] * fill) for o in inventory_status["in_transit"]]
        wo = project_fifo(existing, arrivals, f, shelf)
        new = dict(plan, binding_constraints=list(plan["binding_constraints"]))
        new["recommended_order_units"] = int(units)
        new["order_cost"] = round(units * plan["unit_cost"], 2)
        new["expected"] = self._expected(existing, arrivals, units * fill, L, f, shelf, wo,
                                         slice(L, L + cycle))
        return new

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _expected(existing, arrivals, delivered, L, f, shelf, proj_without, win) -> dict:
        arr = list(arrivals) + ([(L, delivered)] if delivered > 0 else [])
        p = project_fifo(existing, arr, f, shelf)
        demand_w = float(f[win].sum())
        unmet = float(p["unmet"][win].sum())
        return {
            "waste_units": round(float(p["expired"].sum() - proj_without["expired"].sum()), 1),
            "unmet_units": round(unmet, 1),
            "service_level": round(1 - unmet / demand_w, 3) if demand_w > 0 else 1.0,
            "leftover_units": round(p["leftover"], 1),
        }

    @staticmethod
    def _rationale(item_id, qty, L, cycle, shelf, window_mean, sl, rel, need_m, inv, out,
                   notes, low_conf) -> str:
        head = (f"Order {qty} units" if qty else "No order this week")
        s = (f"{head}. They arrive in {L} day(s) and must cover days {L}-{L + cycle - 1} from today: "
             f"forecast {window_mean:.0f} units (typical error +/-{rel:.0%}"
             f"{', assumed' if low_conf else ''}). On hand {inv['current_stock']}, "
             f"on order {inv['in_transit_units']}, still needed after netting the expiry of "
             f"older stock: {need_m:.0f}. Target service level {sl:.0%}.")
        if notes:
            s += " " + "; ".join(n[0].upper() + n[1:] if i == 0 else n
                                 for i, n in enumerate(notes)) + "."
        s += (f" Expected: {out['service_level']:.0%} of window demand served, "
              f"~{out['waste_units']:.0f} units wasted.")
        return s


if __name__ == "__main__":
    from agents.orchestrator_agent import OrchestratorAgent
    r = OrchestratorAgent().run(item_id=1000, current_stock=10)
    print(r["optimization_result"]["rationale"])
