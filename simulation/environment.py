"""Simulation environment (FR7).

Replays a real stretch of the demand history, one day at a time, while an
ordering policy places an order on each ordering day (every Monday by default):

  * the policy sees ONLY what it could know that morning (history up to the day
    before, current stock, orders on the way, supplier terms);
  * the order arrives after the supplier's lead time; each ordered unit actually
    arrives with probability `availability_pct` (so deliveries are sometimes short);
  * every day, customers buy what the data says they bought, from the oldest stock
    first; anything the shelf can't cover is a lost sale;
  * at the end of each day, stock past its last sellable day is thrown away (waste).

Fairness / repeatability
  * Common random numbers: whether units arrive is drawn from a generator seeded by
    (seed, day, item), so every policy faces the same supplier luck for the same
    week and item -- differences between policies are not noise in the dice.
  * The environment enforces the supplier's minimum order (an order below it is
    shipped as the minimum).
  * Unit conservation is checked: delivered = sold + wasted + still on the shelf.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from agents.data_loader import load_demand, load_items
from agents.demand_agent import DemandForecastingAgent
from agents.inventory_agent import InventoryAgent
from agents.supplier_agent import SupplierAgent

MIN_HISTORY_FOR_FIRST_ORDER = 133   # the Demand agent needs ~19 weeks to have a track record


@dataclass
class SimResult:
    policy: str
    item_ids: np.ndarray
    dates: pd.DatetimeIndex            # one per simulated day
    n_weeks: int
    sold: np.ndarray                   # (items, days)
    lost: np.ndarray
    wasted: np.ndarray
    delivered: np.ndarray
    ordered: np.ndarray                # (items, weeks) units ordered
    unit_cost: np.ndarray              # per item
    unit_price: np.ndarray
    family: np.ndarray
    closing_stock: np.ndarray
    conservation_error: int
    policy_log: list = field(default_factory=list)

    @property
    def demand(self) -> np.ndarray:
        return self.sold + self.lost


class Simulator:
    def __init__(self, demand: pd.DataFrame | None = None, seed: int = 0):
        demand = load_demand() if demand is None else demand
        self.seed = seed
        self.demand_agent = DemandForecastingAgent(demand=demand)
        self.items = load_items().set_index("item_id")
        self.dates = self.demand_agent.dates
        self.item_ids = self.demand_agent.item_ids
        self.Y = self.demand_agent.Y
        self.P = self.demand_agent.P

    # ------------------------------------------------------------ schedule
    def first_decision_index(self) -> int:
        """Index of the first Monday with enough history for the agents' track records."""
        t = MIN_HISTORY_FOR_FIRST_ORDER
        while self.dates[t].dayofweek != 0:
            t += 1
        return t

    def max_weeks(self) -> int:
        return (len(self.dates) - self.first_decision_index()) // 7

    def planned_promotions(self, t: int, days: int = 14) -> dict:
        """Promotions the restaurant has already scheduled for days t..t+days-1."""
        hi = min(t + days, len(self.dates))
        out = {}
        for r, i in enumerate(self.item_ids):
            idx = np.flatnonzero(self.P[r, t:hi])
            if len(idx):
                out[int(i)] = [self.dates[t + k].date() for k in idx]
        return out

    # ----------------------------------------------------------------- run
    def run(self, policy, n_weeks: int | None = None, review_days=(0,)) -> SimResult:
        """Replay `n_weeks` weeks. `review_days` are the weekdays (0 = Monday) the policy
        places an order: (0,) is the weekly cadence from the SOW; (0, 3) adds a Thursday order."""
        review_days = tuple(sorted(review_days))
        d0 = self.first_decision_index()
        n_weeks = n_weeks or self.max_weeks()
        n_weeks = min(n_weeks, self.max_weeks())
        n_days = 7 * n_weeks
        n_items = len(self.item_ids)

        ledger = InventoryAgent()
        supplier = SupplierAgent()
        if hasattr(policy, "reset"):
            policy.reset()
        sold = np.zeros((n_items, n_days)); lost = np.zeros_like(sold)
        wasted = np.zeros_like(sold); delivered = np.zeros_like(sold)
        ordered = np.zeros((n_items, n_weeks))
        row = {int(i): r for r, i in enumerate(self.item_ids)}
        pending: dict[tuple, int] = defaultdict(int)       # (item, arrival date) -> units that will arrive

        for k in range(n_days):
            t = d0 + k
            today = self.dates[t].date()
            week = k // 7
            if k % 7 in review_days:                        # an ordering day (Monday = start of week)
                nxt = [d for d in review_days if d > k % 7]
                cycle = (nxt[0] if nxt else review_days[0] + 7) - k % 7
                ctx = {"as_of": self.dates[t - 1], "decision_date": today, "week": week,
                       "day_idx": t, "cycle_days": cycle, "inventory": ledger, "supplier": supplier,
                       "demand_agent": self.demand_agent,
                       "planned_promotions": self.planned_promotions(t)}
                orders = policy.decide(ctx)
                for item, qty in orders.items():
                    qty = int(qty)
                    if qty <= 0:
                        continue
                    terms = supplier.terms[int(item)]
                    qty = max(qty, int(terms["moq_units"]))  # supplier ships at least the MOQ
                    arrive = today + timedelta(days=int(terms["lead_time_days"]))
                    rng = np.random.default_rng([self.seed, k, int(item)])
                    pending[(int(item), arrive)] += int(rng.binomial(qty, float(terms["availability_pct"])))
                    ledger.add_in_transit(item, qty, arrive)
                    ordered[row[int(item)], week] += qty
            for item in self.item_ids:                      # morning: deliveries arrive
                i = int(item)
                for o in ledger.pop_due(i, today):
                    got = pending.pop((i, o["arrives_on"]), 0)
                    ledger.receive(i, got, o["arrives_on"] if o["arrives_on"] == today else today)
                    delivered[row[i], k] += got
            for item in self.item_ids:                      # day: customers buy
                i = int(item)
                dem = int(round(self.Y[row[i], t]))
                s = ledger.sell(i, dem, today)
                sold[row[i], k] = s
                lost[row[i], k] = dem - s
            for i, w in ledger.expire(today).items():       # night: expired stock is thrown away
                wasted[row[i], k] = w

        closing = np.array([sum(b["units"] for b in ledger.batches[int(i)]) for i in self.item_ids])
        err = int(delivered.sum() - sold.sum() - wasted.sum() - closing.sum())
        meta = self.items.loc[self.item_ids]
        return SimResult(
            policy=getattr(policy, "name", type(policy).__name__), item_ids=self.item_ids,
            dates=self.dates[d0:d0 + n_days], n_weeks=n_weeks, sold=sold, lost=lost,
            wasted=wasted, delivered=delivered, ordered=ordered,
            unit_cost=np.array([supplier.terms[int(i)]["unit_cost"] for i in self.item_ids]),
            unit_price=meta["unit_price"].to_numpy(float), family=meta["family"].to_numpy(),
            closing_stock=closing, conservation_error=err,
            policy_log=getattr(policy, "log", []))
