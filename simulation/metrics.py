"""Metrics for a simulation run, plus the paired comparison between two policies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.environment import SimResult


def weekly_table(r: SimResult) -> pd.DataFrame:
    """One row per week: units and dollars."""
    W = r.n_weeks
    def wk(a):  # (items, days) -> (items, weeks)
        return a.reshape(a.shape[0], W, 7).sum(axis=2)
    sold, lost, wasted, dlv = wk(r.sold), wk(r.lost), wk(r.wasted), wk(r.delivered)
    c, p = r.unit_cost[:, None], r.unit_price[:, None]
    return pd.DataFrame({
        "week": np.arange(W),
        "week_start": r.dates[::7][:W],
        "demand": (sold + lost).sum(0), "sold": sold.sum(0), "lost": lost.sum(0),
        "wasted": wasted.sum(0), "delivered": dlv.sum(0),
        "revenue": (sold * p).sum(0), "spend": (dlv * c).sum(0),
        "waste_cost": (wasted * c).sum(0),
        "lost_margin": (lost * (p - c)).sum(0),
        "margin": (sold * (p - c)).sum(0) - (wasted * c).sum(0),
    })


def summarize(wk: pd.DataFrame) -> dict:
    """Headline numbers over a set of weeks."""
    d, s = wk["demand"].sum(), wk["sold"].sum()
    return {
        "fill_rate": s / d,                              # share of demand served
        "stockout_rate": wk["lost"].sum() / d,           # share of demand lost
        "waste_rate": wk["wasted"].sum() / max(s + wk["wasted"].sum(), 1),   # of units sold-or-binned
        "units_wasted": wk["wasted"].sum(),
        "waste_cost": wk["waste_cost"].sum(),
        "lost_margin": wk["lost_margin"].sum(),
        "revenue": wk["revenue"].sum(),
        "spend": wk["spend"].sum(),
        "margin_after_waste": wk["margin"].sum(),
    }


def by_family(r: SimResult, weeks: slice | None = None) -> pd.DataFrame:
    sl = slice(None) if weeks is None else slice(weeks.start * 7, weeks.stop * 7)
    sold, lost, wasted = r.sold[:, sl], r.lost[:, sl], r.wasted[:, sl]
    c, p = r.unit_cost[:, None], r.unit_price[:, None]
    df = pd.DataFrame({
        "family": r.family, "demand": (sold + lost).sum(1), "sold": sold.sum(1),
        "wasted": wasted.sum(1), "waste_cost": (wasted * c).sum(1),
        "lost_margin": (lost * (p - c)).sum(1),
        "margin": (sold * (p - c)).sum(1) - (wasted * c).sum(1)})
    g = df.groupby("family").sum()
    g["fill_rate"] = g["sold"] / g["demand"]
    g["waste_rate"] = g["wasted"] / (g["sold"] + g["wasted"]).clip(lower=1)
    return g


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0) -> dict:
    """Mean weekly difference (a - b) with a 95% interval, resampling WEEKS (the same
    weeks for both policies, so supplier luck and demand noise cancel in the pairing)."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {"mean": float(diff.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "weeks": len(diff),
            "share_weeks_better": float((diff > 0).mean())}
