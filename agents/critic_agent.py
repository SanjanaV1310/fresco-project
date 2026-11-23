"""Critic Agent -- WBS #6.

Implements FR5 (Requirements Statement, Section 2.1):
"The system shall review the Optimization Agent's proposed order and flag
orders that appear inconsistent with forecasted demand (e.g., a quantity
historical demand suggests won't be consumed before expiration), returning
either an approval or a specific objection."

The Critic does NOT reuse the Optimizer's numbers. It re-projects the proposed
order against the forecast under three demand scenarios (forecast, low = bottom
of the 80% range, high = top of it) and judges the result:

  INVALID          the quantity is not a sensible whole number            -> object
  EXPIRY           units would expire unsold even at forecast demand      -> object
  OVERSTOCK_LOW    >15% of the order would expire if demand comes in low  -> object
                   (only if cutting wouldn't leave us short at forecast)
  UNDERSTOCK       >20% of forecast demand unserved                       -> object
                   (a warning instead when shelf life, budget or MOQ makes it unavoidable)
                  and only when ordering more would actually serve more demand

An objection always says exactly what to change: `suggested` = {"max_units": n} or
{"min_units": n}, which the Orchestrator feeds back to the Optimizer (the replan
loop). When a problem can't be fixed without causing another (e.g. forced MOQ
waste, or a shelf-life-limited item that will run short late in the week) the
Critic approves WITH A WARNING and says why, instead of objecting forever.

Per Risk R10 this agent is tested against both clearly-bad and clearly-fine orders.
"""
from __future__ import annotations

import math

import numpy as np

from agents.base_agent import BaseAgent
from agents.inventory_math import (CYCLE_DAYS, DEFAULT_WINDOW_ERROR, extend_forecast,
                                   project_fifo)

_Z80 = 1.2816
EXPIRY_TOLERANCE = 0.10        # share of delivered units allowed to expire at forecast demand
LOW_DEMAND_WASTE_TOLERANCE = 0.15
UNDERSTOCK_TOLERANCE = 0.20
MIN_UNITS_TO_CARE = 3          # ignore "problems" smaller than this many units


class CriticAgent(BaseAgent):
    name = "critic_agent"

    def run(self, optimization_result: dict, demand_forecast: dict,
            shelf_life_days: int | None = None, inventory_status: dict | None = None,
            **kwargs) -> dict:
        plan = optimization_result
        item_id = int(plan["item_id"])
        Q = plan["recommended_order_units"]
        binding = set(plan.get("binding_constraints", []))
        objections: list[dict] = []
        warnings: list[dict] = []

        # ---- sanity ------------------------------------------------------
        if (not isinstance(Q, (int, np.integer))) or Q < 0 or (isinstance(Q, float) and math.isnan(Q)):
            objections.append(self._obj("INVALID", f"Quantity {Q!r} is not a whole number of "
                                        f"units >= 0.", {"max_units": 0}))
            return self._result(item_id, objections, warnings, None, plan)

        L = int(plan["lead_time_days"])
        cycle = int(plan.get("cycle_days", CYCLE_DAYS))
        fill = float(plan["expected_fill"])
        shelf = int(shelf_life_days or plan["shelf_life_days"])
        f = extend_forecast([d["units"] for d in demand_forecast["daily_forecast"]], L + cycle)
        win = slice(L, L + cycle)
        rel = (demand_forecast.get("error_scale") or {}).get("window_relative")
        low_conf = rel is None
        rel = DEFAULT_WINDOW_ERROR if low_conf else rel
        inv = inventory_status or {"batches": [], "in_transit": [], "expiring_soon": []}
        existing = [(b["units"], b["last_offset"]) for b in inv["batches"]]
        arrivals = [(max(o["offset"], 0), o["units"] * fill) for o in inv["in_transit"]]
        delivered = Q * fill

        def run_scenario(scale, qty):
            arr = arrivals + ([(L, qty)] if qty > 0 else [])
            return project_fifo(existing, arr, scale * f, shelf)

        lo_s, hi_s = max(1 - _Z80 * rel, 0.0), 1 + _Z80 * rel
        mean_wo, mean_w = run_scenario(1.0, 0), run_scenario(1.0, delivered)
        low_wo, low_w = run_scenario(lo_s, 0), run_scenario(lo_s, delivered)
        high_w = run_scenario(hi_s, delivered)
        waste_mean = float(mean_w["expired"].sum() - mean_wo["expired"].sum())
        waste_low = float(low_w["expired"].sum() - low_wo["expired"].sum())
        window_mean = float(f[win].sum())
        unmet_mean = float(mean_w["unmet"][win].sum())
        unmet_high = float(high_w["unmet"][win].sum())
        need_mean = float(mean_wo["unmet"][win].sum())    # what forecast demand needs from this order
        if shelf < cycle:
            need_mean = min(need_mean, float(mean_wo["unmet"][L:L + shelf].sum()))
        floor_units = int(math.ceil(need_mean / fill - 1e-9)) if need_mean > 0 else 0
        moq = int(plan.get("moq_units", 0))
        if 0 < floor_units < moq:
            floor_units = moq if "moq_skip" not in binding else 0
        checks = {"waste_at_forecast_units": round(waste_mean, 1),
                  "waste_if_demand_low_units": round(waste_low, 1),
                  "unmet_at_forecast_units": round(unmet_mean, 1),
                  "unmet_if_demand_high_units": round(unmet_high, 1),
                  "low_scenario": round(lo_s, 2), "high_scenario": round(hi_s, 2)}

        # ---- 1. expiry at forecast demand (the FR5 example) --------------------
        if delivered > 0 and waste_mean > EXPIRY_TOLERANCE * delivered and waste_mean >= MIN_UNITS_TO_CARE:
            cut = int(math.ceil(waste_mean / fill - 1e-9))
            suggestion = max(Q - cut, 0)
            msg = (f"Ordering {Q} units (~{delivered:.0f} arriving) but forecast demand only "
                   f"consumes ~{delivered - waste_mean:.0f} of them before they expire: "
                   f"~{waste_mean:.0f} units (${waste_mean * plan['unit_cost']:.0f}) would be "
                   f"wasted. Suggest at most {suggestion}.")
            if "moq_raise" in binding:
                warnings.append(self._warn("MOQ_FORCED_WASTE", msg + " Not objecting: the minimum "
                                           "order forces it and the Optimizer judged the margin "
                                           "worth it."))
            else:
                objections.append(self._obj("EXPIRY", msg, {"max_units": suggestion}))

        # ---- 2. waste if demand is low ---------------------------------------------
        elif (delivered > 0 and waste_low > LOW_DEMAND_WASTE_TOLERANCE * delivered
              and waste_low >= MIN_UNITS_TO_CARE and "moq_raise" not in binding):
            suggestion = max(Q - int(math.ceil(waste_low / fill - 1e-9)), floor_units)
            if suggestion < Q:
                objections.append(self._obj(
                    "OVERSTOCK_LOW",
                    f"If demand lands at the low end of its 80% range ({lo_s:.0%} of forecast), "
                    f"~{waste_low:.0f} of the {Q} units ({waste_low / delivered:.0%}) would "
                    f"expire unsold (${waste_low * plan['unit_cost']:.0f}). Suggest at most "
                    f"{suggestion}, which still covers forecast demand.", {"max_units": suggestion}))
            else:
                warnings.append(self._warn(
                    "WASTE_RISK_ACCEPTED",
                    f"~{waste_low:.0f} units could expire if demand is low, but cutting the order "
                    f"would leave forecast demand unserved, so the order stands."))

        # ---- 3. understock at forecast demand ----------------------------------------
        # Only worth objecting to if ordering MORE would actually help: ask what an unlimited
        # order would still leave unserved (that part is the shelf life's doing, not the plan's).
        if not any(o["code"] in ("EXPIRY", "OVERSTOCK_LOW") for o in objections) and window_mean > 0 \
                and unmet_mean > UNDERSTOCK_TOLERANCE * window_mean and unmet_mean >= MIN_UNITS_TO_CARE:
            unmet_floor = float(run_scenario(1.0, 1e7)["unmet"][win].sum())
            avoidable = unmet_mean - unmet_floor
            msg = (f"At forecast demand ~{unmet_mean:.0f} of {window_mean:.0f} units "
                   f"({unmet_mean / window_mean:.0%}) in the cover window would go unserved.")
            why = {"budget_trimmed": "the weekly budget is binding",
                   "budget_dropped": "the weekly budget is binding",
                   "moq_skip": "the minimum order isn't worth the expected waste",
                   "max_units": "an earlier objection capped the order"}
            reasons = [why[b] for b in sorted(binding & set(why))]
            if unmet_floor > UNDERSTOCK_TOLERANCE * window_mean:
                reasons.insert(0, f"the shelf life ({shelf} day(s)) is too short to hold stock for "
                                  f"the whole {cycle}-day window, so even an unlimited order would "
                                  f"leave ~{unmet_floor:.0f} units unserved")
            if reasons or avoidable < MIN_UNITS_TO_CARE:
                warnings.append(self._warn("UNDERSTOCK_ACCEPTED",
                                           f"{msg} Accepted because {'; '.join(reasons) or 'ordering more would not help'}."))
            else:
                add = int(math.ceil(avoidable / fill - 1e-9))
                objections.append(self._obj("UNDERSTOCK", f"{msg} Suggest at least {Q + add}.",
                                            {"min_units": Q + add}))

        # ---- informational warnings ----------------------------------------------------
        if (window_mean > 0 and unmet_high > 0.30 * high_w["sold"][win].sum() + 1e-9
                and unmet_high >= MIN_UNITS_TO_CARE and not any(w["code"] == "UNDERSTOCK_ACCEPTED" for w in warnings)):
            warnings.append(self._warn("STOCKOUT_RISK_HIGH_DEMAND",
                                       f"If demand lands at the high end (~{hi_s:.0%} of forecast) "
                                       f"~{unmet_high:.0f} units would go unserved."))
        if low_conf:
            warnings.append(self._warn("LOW_CONFIDENCE_FORECAST",
                                       "This item has no forecast track record, so error ranges are assumed."))
        if inv.get("expiring_soon"):
            u = sum(e["units"] for e in inv["expiring_soon"])
            warnings.append(self._warn("EXPIRING_STOCK",
                                       f"{u} units already on hand expire within 2 selling days."))
        return self._result(item_id, objections, warnings, checks, plan)

    # -------------------------------------------------------------- basket
    def review_basket(self, plans: list[dict], budget: float | None) -> dict:
        """Basket-level check: does the whole order respect the budget?"""
        cost = round(sum(p["order_cost"] for p in plans), 2)
        ok = budget is None or cost <= budget + 1e-6
        return {"approved": ok, "total_cost": cost, "budget": budget,
                "rationale": ("Basket within budget." if ok else
                              f"Basket costs ${cost:,.2f}, over the ${budget:,.2f} budget.")
                             if budget is not None else "No budget set."}

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _obj(code, message, suggested) -> dict:
        return {"code": code, "message": message, "suggested": suggested}

    @staticmethod
    def _warn(code, message) -> dict:
        return {"code": code, "message": message}

    @staticmethod
    def _result(item_id, objections, warnings, checks, plan) -> dict:
        approved = not objections
        if approved:
            verdict = "approve_with_warnings" if warnings else "approve"
            text = ("Approved." if not warnings else
                    "Approved with warning(s): " + " ".join(w["message"] for w in warnings))
        else:
            verdict = "object"
            text = "Objection: " + " ".join(o["message"] for o in objections)
        suggested: dict = {}
        for o in objections:
            for k, v in o["suggested"].items():
                suggested[k] = min(suggested.get(k, v), v) if k == "max_units" else max(suggested.get(k, v), v)
        return {
            "item_id": item_id,
            "approved": approved,
            "verdict": verdict,
            "objection": objections[0]["message"] if objections else None,
            "objections": objections,
            "warnings": warnings,
            "suggested": suggested,
            "checks": checks,
            "rationale": text,
        }
