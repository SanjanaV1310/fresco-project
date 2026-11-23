"""Demand Forecasting Agent -- built in Week 2 (see Project Plan, WBS #2).

Implements FR1 (Requirements Statement, Section 2.1):
"The system shall predict expected demand for each SKU over the
coming ordering period, using historical/synthetic sales data."

What it does
------------
For one item, it forecasts the next 7 days of demand (one weekly ordering
cycle) using a weekday-aware model: the item's recent baseline level
multiplied by its weekday pattern (and a promotion lift on days a promotion
is planned). It also returns an 80% prediction interval and a plain-English
rationale, so the Optimization and Critic agents -- and a human reviewer --
can see how much to trust the number (NFR1, Explainability).

Output contract (what the Orchestrator and downstream agents rely on)
---------------------------------------------------------------------
  forecast_units        average demand in UNITS PER DAY over the horizon
                        (same meaning as the Week 1 placeholder, so the
                        Optimization / Critic agents keep working unchanged)
  horizon_total_units   total demand over the whole horizon
  daily_forecast        per-day forecast with lower/upper bounds
  total_interval        80% interval for the horizon total
  method, rationale, backtest, ...  explainability fields
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from agents import forecasting as fc
from agents.base_agent import BaseAgent
from agents.data_loader import load_demand

# Chosen by backtesting (scripts/evaluate_demand_agent.py): within ~0.2 points of
# the most accurate candidate (Holt-Winters, a difference too small to matter) and
# much simpler to explain, which matters for NFR1 (Explainability).
DEFAULT_METHOD = "weekday_level"
BASELINE_METHOD = "trailing_mean_7"   # the Week 1 placeholder, kept for comparison
INTERVAL_LEVEL = 0.80
_Z80 = 1.2816                         # normal z-score for an 80% two-sided interval
_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class DemandForecastingAgent(BaseAgent):
    name = "demand_forecasting_agent"

    def __init__(self, demand: pd.DataFrame | None = None,
                 method: str = DEFAULT_METHOD, horizon_days: int = fc.HORIZON_DAYS):
        if method not in fc.MODELS:
            raise ValueError(f"Unknown method '{method}'. Choose from {sorted(fc.MODELS)}.")
        demand = load_demand() if demand is None else demand
        self.method = method
        self.horizon_days = horizon_days
        self.item_ids, self.dates, self.Y, self.P = fc.build_panel(demand)
        self._dows = self.dates.dayofweek.to_numpy()
        self._item_row = {int(i): r for r, i in enumerate(self.item_ids)}
        self._lift_cache: dict[int, float] = {}
        self._backtest_cache: dict[tuple, tuple] = {}

    # ------------------------------------------------------------------ run
    def run(self, item_id: int, as_of=None, planned_promotions=None,
            method: str | None = None, horizon_days: int | None = None,
            cover_window: tuple[int, int] | None = None,
            include_baseline: bool = True, **kwargs) -> dict:
        """Forecast the next `horizon_days` days for one item.

        item_id              SKU to forecast (must exist in the demand data).
        as_of                Forecast using only data up to and including this
                             date (default: the latest date in the data). Lets
                             the Simulation layer replay past weeks without
                             peeking at the future.
        planned_promotions   Dates in the horizon when this item is on
                             promotion (default: none planned).
        method               Override the model (e.g. for comparisons).
        horizon_days         Override the forecast length (the Inventory/Optimization
                             agents ask for lead time + 7 days, so the forecast covers
                             the whole period an order has to serve).
        cover_window         (start, end) day offsets inside the horizon that one order
                             has to cover. The agent then also reports the typical error
                             of THAT window's total (error_scale["window_relative"]).
        include_baseline     Also score the old 7-day average for the track record
                             (needed only for the explanation; skip for speed).
        """
        method = method or self.method
        if method not in fc.MODELS:
            raise ValueError(f"Unknown method '{method}'. Choose from {sorted(fc.MODELS)}.")
        if int(item_id) not in self._item_row:
            raise ValueError(f"Unknown item_id {item_id}: no demand history for it.")
        row = self._item_row[int(item_id)]

        t = self._cutoff_index(as_of)           # number of history days we may use
        h = int(horizon_days or self.horizon_days)
        last_day = self.dates[t - 1]
        future_dates = pd.date_range(last_day + pd.Timedelta(days=1), periods=h, freq="D")
        future_dows = future_dates.dayofweek.to_numpy()
        promo_days = {pd.Timestamp(d).normalize() for d in (planned_promotions or [])}
        future_promo = np.array([d in promo_days for d in future_dates])

        # --- cold start: too little history for a weekday model -----------
        if t < fc.MIN_HISTORY_DAYS:
            return self._cold_start(item_id, row, t, last_day, future_dates, cover_window)

        lift = self._promo_lift(t)
        y, p, hd = self.Y[row, :t], self.P[row, :t], self._dows[:t]
        pred = fc.predict(method, y, p, hd, future_dows, future_promo, lift)

        # --- uncertainty + accuracy track record from this item's own backtest
        daily_scale, weekly_scale, item_wape, base_wape, n_origins, window_scale = \
            self._item_track_record(row, t, method, h, cover_window, include_baseline)
        if daily_scale is not None:
            sig = daily_scale * pred
            lower = np.maximum(pred - _Z80 * sig, 0.0)
            upper = pred + _Z80 * sig
            total = float(pred.sum())
            tsig = weekly_scale * total
            total_interval = {"level": INTERVAL_LEVEL,
                              "lower": round(max(total - _Z80 * tsig, 0.0), 1),
                              "upper": round(total + _Z80 * tsig, 1)}
        else:
            lower = upper = None
            total_interval = None

        daily = [{"date": d.date().isoformat(), "weekday": _DAY_NAMES[d.dayofweek],
                  "units": round(float(v), 1),
                  "lower": None if lower is None else round(float(lower[i]), 1),
                  "upper": None if upper is None else round(float(upper[i]), 1),
                  "promo": bool(future_promo[i])}
                 for i, (d, v) in enumerate(zip(future_dates, pred))]

        return {
            "item_id": int(item_id),
            "forecast_units": round(float(pred.mean()), 1),
            "forecast_unit": "units_per_day",
            "horizon_days": h,
            "forecast_start": future_dates[0].date().isoformat(),
            "horizon_total_units": round(float(pred.sum()), 1),
            "daily_forecast": daily,
            "total_interval": total_interval,
            "method": method,
            "as_of": last_day.date().isoformat(),
            "history_days": int(t),
            "promo_lift_used": round(lift, 2),
            "backtest": None if item_wape is None else {
                "n_weeks": n_origins,
                "wape": round(item_wape, 3),
                "baseline_wape": None if base_wape is None else round(base_wape, 3),
            },
            "error_scale": {
                "daily_relative": None if daily_scale is None else round(daily_scale, 4),
                "total_relative": None if weekly_scale is None else round(weekly_scale, 4),
                "window_relative": None if window_scale is None else round(window_scale, 4),
            },
            "cover_window": None if cover_window is None else {
                "start": int(cover_window[0]), "end": int(cover_window[1]),
                "units": round(float(pred[cover_window[0]:cover_window[1]].sum()), 1)},
            "rationale": self._rationale(method, y, p, hd, lift, pred, future_promo,
                                         item_wape, base_wape, total_interval),
        }

    # -------------------------------------------------------------- helpers
    def _cutoff_index(self, as_of) -> int:
        """Number of days of history available when forecasting as of `as_of`."""
        if as_of is None:
            return len(self.dates)
        ts = pd.Timestamp(as_of).normalize()
        t = int(self.dates.searchsorted(ts, side="right"))
        if t == 0:
            raise ValueError(f"as_of={as_of} is before the first date in the data "
                             f"({self.dates[0].date()}).")
        return t

    def _promo_lift(self, t: int) -> float:
        if t not in self._lift_cache:
            self._lift_cache[t] = fc.estimate_promo_lift(
                self.Y[:, :t], self.P[:, :t], self._dows[:t])
        return self._lift_cache[t]

    def _backtest(self, t: int, method: str, h: int | None = None):
        """Rolling backtest of the whole panel, using only data before day `t`.

        Each backtest week's forecast is made from data before THAT week, so one
        full-history run can be reused for any cutoff: for cutoff t we keep only
        the weeks that were fully over by t. (Cached; this is what keeps replaying
        many past weeks fast.)
        """
        h = int(h or self.horizon_days)
        key = (method, h)
        if key not in self._backtest_cache:
            self._backtest_cache[key] = fc.rolling_origin_backtest(
                self.Y, self.P, self.dates, method, horizon=h)
        origins, F, A = self._backtest_cache[key]
        n = sum(1 for o in origins if o + h <= t)
        return origins[:n], F[:n], A[:n]

    def _item_track_record(self, row: int, t: int, method: str, h: int,
                           cover_window=None, include_baseline: bool = True):
        """This item's past forecast errors (strictly before the cutoff)."""
        origins, F, A = self._backtest(t, method, h)
        if len(origins) < 6:
            return None, None, None, None, len(origins), None
        d_scale, w_scale = fc.relative_error_scales(F[:, row, :], A[:, row, :])
        item_wape = fc.error_metrics(F[:, row, :], A[:, row, :])["wape"]
        base_wape = None
        if include_baseline:
            _, Fb, Ab = self._backtest(t, BASELINE_METHOD, h)
            base_wape = fc.error_metrics(Fb[:, row, :], Ab[:, row, :])["wape"]
        win_scale = None
        if cover_window is not None:
            a, b = int(cover_window[0]), int(cover_window[1])
            _, win_scale = fc.relative_error_scales(F[:, row, a:b], A[:, row, a:b])
        return d_scale, w_scale, item_wape, base_wape, len(origins), win_scale

    def _cold_start(self, item_id, row, t, last_day, future_dates, cover_window=None) -> dict:
        """Not enough history for a weekday model: flat average, no interval."""
        recent = float(self.Y[row, :t].mean())
        h = len(future_dates)
        return {
            "item_id": int(item_id),
            "forecast_units": round(recent, 1),
            "forecast_unit": "units_per_day",
            "horizon_days": h,
            "forecast_start": future_dates[0].date().isoformat(),
            "horizon_total_units": round(recent * h, 1),
            "daily_forecast": [{"date": d.date().isoformat(), "weekday": _DAY_NAMES[d.dayofweek],
                                "units": round(recent, 1), "lower": None, "upper": None,
                                "promo": False} for d in future_dates],
            "total_interval": None,
            "method": "average_fallback",
            "as_of": last_day.date().isoformat(),
            "history_days": int(t),
            "promo_lift_used": None,
            "backtest": None,
            "error_scale": {"daily_relative": None, "total_relative": None,
                            "window_relative": None},
            "cover_window": None if cover_window is None else {
                "start": int(cover_window[0]), "end": int(cover_window[1]),
                "units": round(recent * (cover_window[1] - cover_window[0]), 1)},
            "rationale": (f"Only {t} days of history (need {fc.MIN_HISTORY_DAYS} for a weekday "
                          f"model), so this is a flat average of {recent:.1f} units/day with no "
                          f"uncertainty range. Treat as a rough placeholder."),
        }

    def _rationale(self, method, y, p, hd, lift, pred, future_promo,
                   item_wape, base_wape, total_interval) -> str:
        parts = []
        if method == "weekday_level":
            y_adj = np.where(p, y / lift, y)
            idx = fc.weekday_index(y_adj, hd)
            level = float(np.mean(y_adj[-28:] / np.maximum(idx[hd[-28:]], 0.05)))
            hi, lo = int(np.argmax(idx)), int(np.argmin(idx))
            parts.append(
                f"Weekday model: recent baseline of {level:.1f} units/day (last 28 days, "
                f"promotions removed) x this item's weekday pattern "
                f"({_DAY_NAMES[hi]} {idx[hi]:.2f}x is the busiest day, "
                f"{_DAY_NAMES[lo]} {idx[lo]:.2f}x the quietest).")
        else:
            parts.append(f"Method: {method}.")
        if future_promo.any():
            parts.append(f"{int(future_promo.sum())} planned promo day(s) lifted by {lift:.2f}x "
                         f"(learned across all items).")
        else:
            parts.append("No promotions planned in the horizon, so none assumed.")
        parts.append(f"Next {len(pred)} days: {pred.sum():.0f} units total "
                     f"({pred.mean():.1f}/day).")
        if total_interval:
            parts.append(f"80% range for the week: {total_interval['lower']:.0f}-"
                         f"{total_interval['upper']:.0f} units.")
        if item_wape is not None and base_wape is not None:
            parts.append(f"Track record on this item: {item_wape:.0%} average daily error vs "
                         f"{base_wape:.0%} for the simple 7-day average it replaces.")
        elif item_wape is not None:
            parts.append(f"Track record on this item: {item_wape:.0%} average daily error.")
        return " ".join(parts)


if __name__ == "__main__":
    import json
    agent = DemandForecastingAgent()
    print(json.dumps(agent.run(item_id=1000), indent=2))
