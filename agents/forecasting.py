"""Forecasting models and backtesting tools for the Demand Forecasting Agent.

Pure NumPy/pandas, no file I/O, so every model can be unit-tested and
backtested without touching the agent or the CSV files.

Conventions used throughout:
  * A history is a 1-D array, oldest -> newest, one value per calendar day.
  * Weekdays are ints, 0 = Monday ... 6 = Sunday.
  * Every forecast is in UNITS PER DAY.
  * Promotions are planned by the restaurant in advance, so the promo flags
    for the forecast horizon are an *input* (default: no promotions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_DAYS = 7          # one weekly ordering cycle
MIN_HISTORY_DAYS = 28     # below this we fall back to a plain average
INDEX_WINDOW = 84         # 12 weeks used to estimate weekday pattern / promo lift
BACKTEST_START = 84       # first backtest origin = 84 days of history
_INDEX_FLOOR = 0.05       # keeps divisions safe for items that skip a weekday


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def weekday_index(y_adj: np.ndarray, dows: np.ndarray, window: int = INDEX_WINDOW) -> np.ndarray:
    """Relative demand level of each weekday (Mon..Sun), averaging to 1.0.

    1.33 on Saturday means "Saturdays sell about 33% more than the average day".
    """
    y, d = y_adj[-window:], dows[-window:]
    means = np.array([y[d == k].mean() if np.any(d == k) else np.nan for k in range(7)])
    overall = np.nanmean(means) if np.any(np.isfinite(means)) else np.nan
    if not np.isfinite(overall) or overall <= 0:
        return np.ones(7)
    idx = np.where(np.isfinite(means), means / overall, 1.0)
    return idx / idx.mean()


def estimate_promo_lift(Y: np.ndarray, P: np.ndarray, dows: np.ndarray,
                        window: int = INDEX_WINDOW, lo: float = 1.0, hi: float = 3.0) -> float:
    """Pooled promotion lift across all items: how many times higher sales are
    on a promo day than on a normal day of the same weekday.

    Pooling across items matters: one item has only a handful of promo days,
    so its own estimate would be very noisy. Returns 1.0 (no lift) if there
    are no promo days to learn from.
    """
    y, p, d = Y[:, -window:], P[:, -window:], dows[-window:]
    num = den = 0.0
    for k in range(7):
        m = d == k
        yk, pk = y[:, m], p[:, m]
        base_cnt = (~pk).sum(axis=1)
        base_sum = np.where(~pk, yk, 0.0).sum(axis=1)
        base_mean = base_sum / np.maximum(base_cnt, 1)
        promo_cnt = pk.sum(axis=1)
        promo_sum = np.where(pk, yk, 0.0).sum(axis=1)
        valid = (base_cnt > 0) & (promo_cnt > 0) & (base_mean > 0)
        num += promo_sum[valid].sum()
        den += (base_mean[valid] * promo_cnt[valid]).sum()
    if den <= 0:
        return 1.0
    return float(np.clip(num / den, lo, hi))


def _remove_promo(y: np.ndarray, promo: np.ndarray, lift: float) -> np.ndarray:
    """Divide promo-day sales by the lift so promos don't distort the baseline."""
    return np.where(promo, y / lift, y)


# --------------------------------------------------------------------------
# Models. All share one signature and return units/day for each future day.
# --------------------------------------------------------------------------
def trailing_mean(y, promo, hist_dows, future_dows, future_promo, lift, window=7, **_):
    """BASELINE (the Week 1 placeholder): average of the last `window` days."""
    return np.full(len(future_dows), y[-window:].mean())


def seasonal_naive(y, promo, hist_dows, future_dows, future_promo, lift, weeks=4, **_):
    """Average of the same weekday over the last `weeks` weeks (non-promo days)."""
    ry, rd, rp = y[-7 * weeks:], hist_dows[-7 * weeks:], promo[-7 * weeks:]
    out = []
    for k in future_dows:
        m = (rd == k) & ~rp
        if not m.any():
            m = rd == k
        out.append(ry[m].mean() if m.any() else y.mean())
    return np.array(out)


def weekday_level(y, promo, hist_dows, future_dows, future_promo, lift, level_window=28, **_):
    """Recent baseline level x weekday pattern (x promo lift on promo days).

    1. Strip promo effects out of history.
    2. Learn the weekday pattern from the last 12 weeks.
    3. Deseasonalise the last `level_window` days and average them -> the
       current underlying level (units/day on an "average" weekday).
    4. Forecast = level x weekday factor, times the lift if a promo is planned.
    """
    y_adj = _remove_promo(y, promo, lift)
    idx = np.maximum(weekday_index(y_adj, hist_dows), _INDEX_FLOOR)
    level = (y_adj[-level_window:] / idx[hist_dows[-level_window:]]).mean()
    f = level * idx[future_dows]
    return np.where(future_promo, f * lift, f)


def holt_winters(y, promo, hist_dows, future_dows, future_promo, lift,
                 alpha=0.2, gamma=0.1, window=INDEX_WINDOW, **_):
    """Multiplicative exponential smoothing with weekly seasonality
    (Holt-Winters, no trend). Level and weekday factors both adapt, with more
    weight on recent days (alpha = level speed, gamma = weekday-pattern speed).
    """
    y_adj = _remove_promo(y, promo, lift)[-window:]
    d = hist_dows[-window:]
    s = np.maximum(weekday_index(y_adj, d, window), _INDEX_FLOOR)
    level = (y_adj[:7] / s[d[:7]]).mean()
    for yt, k in zip(y_adj, d):
        level = alpha * (yt / s[k]) + (1 - alpha) * level
        s[k] = gamma * (yt / max(level, 1e-9)) + (1 - gamma) * s[k]
    f = level * s[future_dows]
    return np.where(future_promo, f * lift, f)


MODELS = {
    "trailing_mean_7":   (trailing_mean,   {}),
    "seasonal_naive_4w": (seasonal_naive,  {}),
    "weekday_level":     (weekday_level,   {"level_window": 28}),
    # alpha/gamma chosen on the first half of backtest origins (see
    # scripts/evaluate_demand_agent.py); slow smoothing wins on this stable data.
    "holt_winters":      (holt_winters,    {"alpha": 0.05, "gamma": 0.02}),
}


def predict(method: str, y, promo, hist_dows, future_dows, future_promo,
            lift: float = 1.0, **overrides) -> np.ndarray:
    """Run one model; forecasts are clipped at zero (no negative demand)."""
    if method not in MODELS:
        raise ValueError(f"Unknown method '{method}'. Choose from {sorted(MODELS)}.")
    fn, params = MODELS[method]
    f = fn(np.asarray(y, float), np.asarray(promo, bool), np.asarray(hist_dows),
           np.asarray(future_dows), np.asarray(future_promo, bool), lift,
           **{**params, **overrides})
    return np.maximum(f, 0.0)


# --------------------------------------------------------------------------
# Data layout + backtesting
# --------------------------------------------------------------------------
def build_panel(demand: pd.DataFrame):
    """Turn the long demand table into aligned arrays.

    Returns (item_ids, dates, Y, P): Y[i, t] = units sold, P[i, t] = on promo.
    Days with no row are treated as zero sales / no promo (the Favorita
    convention: a missing row means nothing sold).
    """
    df = demand.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["onpromotion"] = df["onpromotion"].astype(int)
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    Y = (df.pivot_table(index="item_id", columns="date", values="unit_sales", aggfunc="sum")
           .reindex(columns=dates).fillna(0.0))
    P = (df.pivot_table(index="item_id", columns="date", values="onpromotion", aggfunc="max")
           .reindex(columns=dates).fillna(0).astype(bool))
    return Y.index.to_numpy(), dates, Y.to_numpy(float), P.to_numpy(bool)


def rolling_origin_backtest(Y, P, dates, method, horizon=HORIZON_DAYS, start=BACKTEST_START,
                            step=7, promo_known=True, **overrides):
    """Simulate the weekly process: at each origin, forecast the next `horizon`
    days using ONLY earlier data, then compare to what actually happened.

    Returns (origins, F, A) with F, A shaped (n_origins, n_items, horizon).
    """
    dows = dates.dayofweek.to_numpy()
    T, n_items = Y.shape[1], Y.shape[0]
    origins = list(range(start, T - horizon + 1, step))
    F = np.zeros((len(origins), n_items, horizon))
    A = np.zeros_like(F)
    for oi, t in enumerate(origins):
        lift = estimate_promo_lift(Y[:, :t], P[:, :t], dows[:t])
        fd = dows[t:t + horizon]
        for i in range(n_items):
            fp = P[i, t:t + horizon] if promo_known else np.zeros(horizon, bool)
            F[oi, i] = predict(method, Y[i, :t], P[i, :t], dows[:t], fd, fp, lift, **overrides)
            A[oi, i] = Y[i, t:t + horizon]
    return origins, F, A


def relative_error_scales(F: np.ndarray, A: np.ndarray, min_origins: int = 6):
    """Typical forecast error as a fraction of the forecast, learned from an
    item's own backtest (F, A shaped (origins, horizon)).

    Returns (daily_scale, weekly_scale), or (None, None) if there are too few
    backtest weeks to trust. Demand noise grows with volume, so errors are
    scaled proportionally; this gave the best-calibrated intervals out of sample.
    """
    if F.shape[0] < min_origins:
        return None, None
    e = F - A
    daily = float(np.sqrt(np.mean((e / np.maximum(F, 1e-6)) ** 2)))
    fw, ew = F.sum(-1), e.sum(-1)
    weekly = float(np.sqrt(np.mean((ew / np.maximum(fw, 1e-6)) ** 2)))
    return daily, weekly


def error_metrics(F: np.ndarray, A: np.ndarray) -> dict:
    """Standard accuracy measures. WAPE = total absolute error / total sales
    (lower is better); bias > 0 means over-forecasting, < 0 under-forecasting."""
    e = F - A
    total = A.sum()
    return {
        "wape": float(np.abs(e).sum() / total),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "bias_pct": float(e.sum() / total),
        "weekly_wape": float(np.abs(F.sum(-1) - A.sum(-1)).sum() / total),
    }
