"""Shared inventory arithmetic (pure functions, no I/O).

The Optimization Agent and the Critic Agent both need to answer the same
question -- "if demand goes like THIS, how much of this stock sells, how much
runs out, and how much expires?" -- so the day-by-day projection lives here once.
Each agent still decides which demand scenario to feed it, so the Critic's check
stays independent of the Optimizer's.

Conventions: day offsets count from the decision day (offset 0 = today).
Stock is sold oldest-first (FIFO). A batch whose last sellable day is d is
thrown away at the END of day d if anything is left.
"""
from __future__ import annotations

import numpy as np

CYCLE_DAYS = 7               # one weekly ordering cycle
DEFAULT_WINDOW_ERROR = 0.30  # window-total error assumed when a forecast has no track record


def project_fifo(batches, arrivals, demand, shelf_life_days: int) -> dict:
    """Project stock, sales, stockouts and expiry day by day.

    batches   [(units, last_sellable_offset)] already on hand
    arrivals  [(arrival_offset, units)] still to come; each is a fresh batch whose
              last sellable day is arrival_offset + shelf_life_days - 1
    demand    units demanded on offsets 0..H-1 (any non-negative floats)
    Returns arrays (length H) sold / unmet / expired, plus `leftover` at the end.
    """
    H = len(demand)
    stock = [[float(q), int(last)] for q, last in batches if q > 0 and last >= 0]
    due: dict[int, list[float]] = {}
    for off, q in arrivals:
        if q > 0 and 0 <= off < H:
            due.setdefault(int(off), []).append(float(q))
    sold = np.zeros(H)
    unmet = np.zeros(H)
    expired = np.zeros(H)
    for d in range(H):
        for q in due.get(d, []):
            stock.append([q, d + int(shelf_life_days) - 1])
        stock.sort(key=lambda b: b[1])
        need = float(demand[d])
        for b in stock:
            if need <= 1e-12:
                break
            take = min(b[0], need)
            b[0] -= take
            need -= take
            sold[d] += take
        unmet[d] = max(need, 0.0)
        for b in stock:
            if b[1] <= d and b[0] > 1e-9:
                expired[d] += b[0]
                b[0] = 0.0
        stock = [b for b in stock if b[0] > 1e-9]
    return {"sold": sold, "unmet": unmet, "expired": expired,
            "leftover": float(sum(b[0] for b in stock))}


def extend_forecast(units, needed: int) -> np.ndarray:
    """Make sure we have `needed` days of forecast; if short, repeat the last
    7-day pattern (the weekday pattern repeats)."""
    f = list(map(float, units))
    n = len(f)
    while len(f) < needed:
        f.append(f[-7] if len(f) >= 7 else f[-1])
    return np.array(f[:max(needed, n)])
