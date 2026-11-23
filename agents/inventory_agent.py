"""Inventory Agent -- WBS #3.

Implements FR2 (Requirements Statement, Section 2.1):
"The system shall track current on-hand inventory per SKU, maintain a
configurable safety-stock threshold per SKU, and flag items nearing
expiration."

It is the stock ledger. For every item it keeps FIFO batches (how many
units, when they arrived, the last day they can still be sold) plus the
orders that are on their way. Other parts of the system record what happened
(`receive`, `sell`, `expire`, `add_in_transit`, `set_count`) and ask for a
status snapshot with `run`. Shelf life counts from the day of delivery: a batch
received on day D with shelf life S can be sold on days D .. D+S-1 and is thrown
away at the end of day D+S-1.

Output contract used by the Optimization / Critic agents (`run`)
  current_stock, batches[{units, received, last_sellable, last_offset}],
  in_transit[{units, arrives_on, offset}], expiring_soon, shelf_life_days,
  safety_stock_units, reorder_point_units, unit_price, flags, rationale
Offsets count days from the decision date (0 = the day the plan is made).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from agents.base_agent import BaseAgent
from agents.data_loader import load_inventory_params, load_items

EXPIRY_WARNING_DAYS = 2   # "nearing expiration" = this many selling days left or fewer


def _to_date(d) -> date:
    return pd.Timestamp(d).date()


class InventoryAgent(BaseAgent):
    name = "inventory_agent"

    def __init__(self, items: pd.DataFrame | None = None,
                 params: pd.DataFrame | None = None):
        items = load_items() if items is None else items
        params = load_inventory_params() if params is None else params
        self.master = {int(r.item_id): {"shelf_life_days": int(r.shelf_life_days),
                                        "unit_price": float(r.unit_price),
                                        "family": r.family}
                       for r in items.itertuples(index=False)}
        self.safety_stock = {int(r.item_id): int(r.safety_stock_units)
                             for r in params.itertuples(index=False)}
        self.reorder_point = {int(r.item_id): int(r.reorder_point_units)
                              for r in params.itertuples(index=False)}
        self.batches: dict[int, list[dict]] = {i: [] for i in self.master}
        self.in_transit: dict[int, list[dict]] = {i: [] for i in self.master}

    # ------------------------------------------------------------ recording
    def receive(self, item_id: int, units: int, on_date) -> None:
        """Record a delivery arriving at the start of `on_date`."""
        units = int(units)
        if units <= 0:
            return
        on = _to_date(on_date)
        shelf = self._master(item_id)["shelf_life_days"]
        self.batches[int(item_id)].append(
            {"units": units, "received": on, "last_sellable": on + timedelta(days=shelf - 1)})

    def sell(self, item_id: int, units: int, on_date) -> int:
        """Sell up to `units` on `on_date`, oldest stock first. Returns units sold."""
        on = _to_date(on_date)
        need = int(units)
        sold = 0
        for b in sorted(self.batches[int(item_id)], key=lambda b: b["last_sellable"]):
            if need <= 0:
                break
            if b["last_sellable"] < on:
                continue                       # already expired; expire() will clear it
            take = min(b["units"], need)
            b["units"] -= take
            need -= take
            sold += take
        self._clean(item_id)
        return sold

    def expire(self, on_date, item_id: int | None = None) -> dict[int, int]:
        """End-of-day clear-out: throw away batches whose last sellable day is
        `on_date` or earlier. Returns {item_id: wasted_units} (only non-zero)."""
        on = _to_date(on_date)
        wasted: dict[int, int] = {}
        for i in ([int(item_id)] if item_id is not None else list(self.batches)):
            w = sum(b["units"] for b in self.batches[i] if b["last_sellable"] <= on)
            if w:
                wasted[i] = w
                self.batches[i] = [b for b in self.batches[i] if b["last_sellable"] > on]
        return wasted

    def add_in_transit(self, item_id: int, units: int, arrives_on) -> None:
        if int(units) > 0:
            self.in_transit[int(item_id)].append(
                {"units": int(units), "arrives_on": _to_date(arrives_on)})

    def pop_due(self, item_id: int, on_date) -> list[dict]:
        """Remove and return the in-transit orders due on or before `on_date`."""
        on = _to_date(on_date)
        due = [o for o in self.in_transit[int(item_id)] if o["arrives_on"] <= on]
        self.in_transit[int(item_id)] = [o for o in self.in_transit[int(item_id)]
                                         if o["arrives_on"] > on]
        return due

    def set_count(self, item_id: int, counted_units: int, as_of) -> dict:
        """Stock correction after a physical count. If the shelf has less than the
        ledger says, the shortfall is taken from the oldest stock (spoilage/shrink
        is most likely there); if it has more, the extra is booked as a batch with
        half its shelf life left (its age is unknown)."""
        on = _to_date(as_of)
        i = int(item_id)
        before = self._on_hand(i, on)
        diff = int(counted_units) - before
        if diff < 0:
            need = -diff
            for b in sorted(self.batches[i], key=lambda b: b["last_sellable"]):
                take = min(b["units"], need)
                b["units"] -= take
                need -= take
                if need <= 0:
                    break
            self._clean(i)
        elif diff > 0:
            self.batches[i].append(self._unknown_age_batch(i, diff, on))
        return {"item_id": i, "ledger_units": before, "counted_units": int(counted_units),
                "adjustment": diff}

    def set_safety_stock(self, item_id: int, units: int) -> None:
        """Make the safety-stock threshold configurable per SKU (FR2)."""
        if int(units) < 0:
            raise ValueError("safety stock cannot be negative")
        self._master(item_id)
        self.safety_stock[int(item_id)] = int(units)

    def reset(self) -> None:
        for i in self.batches:
            self.batches[i] = []
            self.in_transit[i] = []

    # ------------------------------------------------------------------ run
    def run(self, item_id: int, as_of=None, current_stock: int | None = None,
            **kwargs) -> dict:
        """Status snapshot for one item as of the decision date `as_of` (default: today).

        current_stock  Optional ad-hoc override (e.g. a manager typing in a count):
                       treated as one batch of unknown age. It does not change the ledger.
        """
        i = int(item_id)
        m = self._master(i)
        today = _to_date(as_of) if as_of is not None else date.today()
        if current_stock is not None:
            live = [self._unknown_age_batch(i, int(current_stock), today)] if current_stock > 0 else []
        else:
            live = [b for b in self.batches[i] if b["last_sellable"] >= today and b["units"] > 0]
        live = sorted(live, key=lambda b: b["last_sellable"])
        batches = [{"units": int(b["units"]), "received": b["received"].isoformat(),
                    "last_sellable": b["last_sellable"].isoformat(),
                    "last_offset": (b["last_sellable"] - today).days,
                    "days_left": (b["last_sellable"] - today).days + 1} for b in live]
        transit = [{"units": o["units"], "arrives_on": o["arrives_on"].isoformat(),
                    "offset": (o["arrives_on"] - today).days}
                   for o in sorted(self.in_transit[i], key=lambda o: o["arrives_on"])]
        stock = sum(b["units"] for b in batches)
        soon = [{"units": b["units"], "days_left": b["days_left"]}
                for b in batches if b["days_left"] <= EXPIRY_WARNING_DAYS]
        ss, rop = self.safety_stock[i], self.reorder_point[i]
        flags = []
        if stock < ss:
            flags.append("below safety stock")
        if stock < rop:
            flags.append("below reorder point")
        if soon:
            flags.append("units nearing expiration")
        in_t = sum(o["units"] for o in transit)
        rationale = (f"{stock} units on hand"
                     + (f" ({len(batches)} batch(es), oldest has {batches[0]['days_left']} selling "
                        f"day(s) left)" if batches else "")
                     + f", {in_t} on order. Shelf life {m['shelf_life_days']} days. "
                     f"Safety stock {ss}, reorder point {rop}."
                     + (f" Flags: {', '.join(flags)}." if flags else " No flags.")
                     + (" Stock age unknown, so assumed half-fresh." if current_stock else ""))
        return {
            "item_id": i,
            "as_of": today.isoformat(),
            "current_stock": int(stock),
            "batches": batches,
            "in_transit": transit,
            "in_transit_units": int(in_t),
            "expiring_soon": soon,
            "shelf_life_days": m["shelf_life_days"],
            "unit_price": m["unit_price"],
            "safety_stock_units": ss,
            "reorder_point_units": rop,
            "flags": flags,
            "rationale": rationale,
        }

    # -------------------------------------------------------------- helpers
    def _master(self, item_id) -> dict:
        if int(item_id) not in self.master:
            raise ValueError(f"Unknown item_id {item_id}.")
        return self.master[int(item_id)]

    def _unknown_age_batch(self, item_id, units, on: date) -> dict:
        shelf = self._master(item_id)["shelf_life_days"]
        left = max(1, (shelf + 1) // 2)           # selling days left, assumed half
        return {"units": int(units), "received": on - timedelta(days=shelf - left),
                "last_sellable": on + timedelta(days=left - 1)}

    def _on_hand(self, item_id, on: date) -> int:
        return int(sum(b["units"] for b in self.batches[int(item_id)] if b["last_sellable"] >= on))

    def _clean(self, item_id) -> None:
        self.batches[int(item_id)] = [b for b in self.batches[int(item_id)] if b["units"] > 0]


if __name__ == "__main__":
    import json
    inv = InventoryAgent()
    inv.receive(1000, 40, "2026-09-29")
    print(json.dumps(inv.run(item_id=1000, as_of="2026-10-01"), indent=2))
