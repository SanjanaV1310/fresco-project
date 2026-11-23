"""Supplier Agent -- WBS #4.

Implements FR3 (Requirements Statement, Section 2.1):
"The system shall retrieve and maintain supplier price, minimum order
quantity, availability, and lead time for each SKU, from at least one
supplier per item."

It is the single place the rest of the system asks "what are this item's
purchasing terms right now?". Terms start from the supplier catalog
(suppliers.csv) and can change during the week (a price rise, a new MOQ, a
longer lead time) through `apply_update`; every change is logged, so the
Orchestrator can replan and a human can see what changed and when.

Output contract used by the Optimization / Critic agents
  unit_cost, lead_time_days, moq_units, availability_pct (= expected_fill),
  supplier_name, risk_flags, rationale
"""
from __future__ import annotations

import math

import pandas as pd

from agents.base_agent import BaseAgent
from agents.data_loader import load_suppliers

_FIELDS = ("supplier_name", "unit_cost", "lead_time_days", "moq_units", "availability_pct")
LONG_LEAD_DAYS = 5
LOW_AVAILABILITY = 0.95


class SupplierAgent(BaseAgent):
    name = "supplier_agent"

    def __init__(self, suppliers: pd.DataFrame | None = None):
        df = load_suppliers() if suppliers is None else suppliers
        self._catalog = {int(r.item_id): {f: getattr(r, f) for f in _FIELDS}
                         for r in df.itertuples(index=False)}
        self.terms = {i: dict(t) for i, t in self._catalog.items()}
        self.change_log: list[dict] = []

    # ------------------------------------------------------------------ run
    def run(self, item_id: int, **kwargs) -> dict:
        t = self._get(item_id)
        flags = []
        if t["lead_time_days"] >= LONG_LEAD_DAYS:
            flags.append(f"long lead time ({t['lead_time_days']} days)")
        if t["availability_pct"] < LOW_AVAILABILITY:
            flags.append(f"unreliable supply ({t['availability_pct']:.1%} of ordered units "
                         f"arrive on average)")
        changed = [c for c in self.change_log if c["item_id"] == int(item_id)]
        rationale = (f"{t['supplier_name']}: ${t['unit_cost']:.2f}/unit, "
                     f"{t['lead_time_days']}-day lead time, minimum order {int(t['moq_units'])} "
                     f"units, {t['availability_pct']:.1%} of ordered units arrive on average."
                     + (f" Watch: {'; '.join(flags)}." if flags else "")
                     + (f" Terms changed {len(changed)} time(s) since the catalog "
                        f"(latest: {changed[-1]['reason']})." if changed else ""))
        return {
            "item_id": int(item_id),
            "supplier_name": t["supplier_name"],
            "unit_cost": float(t["unit_cost"]),
            "lead_time_days": int(t["lead_time_days"]),
            "moq_units": int(t["moq_units"]),
            "availability_pct": float(t["availability_pct"]),
            "expected_fill": float(t["availability_pct"]),
            "risk_flags": flags,
            "terms_changed": bool(changed),
            "rationale": rationale,
        }

    # -------------------------------------------------------------- updates
    def apply_update(self, item_id: int, reason: str = "supplier update", **changes) -> dict:
        """Change an item's terms (e.g. moq_units=80, lead_time_days=4)."""
        t = self._get(item_id)
        bad = set(changes) - set(_FIELDS)
        if bad:
            raise ValueError(f"Unknown supplier field(s) {sorted(bad)}; allowed: {list(_FIELDS)}")
        diff = {}
        for k, v in changes.items():
            self._validate(k, v)
            if t[k] != v:
                diff[k] = {"old": t[k], "new": v}
                t[k] = v
        rec = {"item_id": int(item_id), "reason": reason, "changes": diff}
        if diff:
            self.change_log.append(rec)
        return rec

    def reset(self) -> None:
        """Back to the catalog terms (used between simulation runs)."""
        self.terms = {i: dict(t) for i, t in self._catalog.items()}
        self.change_log = []

    def quote(self, item_id: int, units: int) -> dict:
        """What would ordering `units` cost and deliver (on average)?"""
        t = self._get(item_id)
        units = int(units)
        return {"item_id": int(item_id), "units": units,
                "meets_moq": units == 0 or units >= int(t["moq_units"]),
                "cost": round(units * float(t["unit_cost"]), 2),
                "expected_delivered": round(units * float(t["availability_pct"]), 1)}

    # -------------------------------------------------------------- helpers
    def _get(self, item_id) -> dict:
        if int(item_id) not in self.terms:
            raise ValueError(f"Unknown item_id {item_id}: no supplier on file.")
        return self.terms[int(item_id)]

    @staticmethod
    def _validate(field, value):
        if field == "supplier_name":
            if not str(value).strip():
                raise ValueError("supplier_name cannot be empty")
        elif field == "availability_pct":
            if not (0 < float(value) <= 1):
                raise ValueError("availability_pct must be in (0, 1]")
        elif field in ("lead_time_days", "moq_units"):
            if not (float(value).is_integer() and value >= (1 if field == "lead_time_days" else 0)):
                raise ValueError(f"{field} must be a whole number (lead time >= 1)")
        elif field == "unit_cost":
            if not (float(value) > 0 and math.isfinite(float(value))):
                raise ValueError("unit_cost must be positive")


if __name__ == "__main__":
    import json
    print(json.dumps(SupplierAgent().run(item_id=1000), indent=2))
