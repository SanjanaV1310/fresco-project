"""Backtest the Demand Forecasting Agent and write a report + chart.

Run from the project root:
    python scripts/evaluate_demand_agent.py

Simulates the real weekly process: every 7 days (from day 84 onward) it
forecasts the next 7 days using ONLY earlier data, then compares against what
actually sold. Four models are compared against the Week 1 placeholder
(7-day average). Outputs go to reports/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents import forecasting as fc
from agents.data_loader import load_demand, load_items
from agents.demand_agent import DEFAULT_METHOD, BASELINE_METHOD

OUT = ROOT / "reports"
Z80 = 1.2816
HOLIDAY = (pd.Timestamp("2025-11-20"), pd.Timestamp("2026-01-02"))  # bump window in generate.py

LABELS = {
    "trailing_mean_7": "7-day average (Week 1 placeholder)",
    "seasonal_naive_4w": "Same weekday, last 4 weeks",
    "weekday_level": "Level x weekday pattern (agent)",
    "holt_winters": "Holt-Winters smoothing",
}


def noise_floor_wape(ids, dates, Y, P, start, end) -> float:
    """WAPE of an oracle that knows the generator's true weekday/holiday/promo
    effects (data/generate.py). No model can beat this on this synthetic data,
    because the day-to-day noise (sd = 25% of the mean, plus 3% zero days) is
    random by construction."""
    items = load_items().set_index("item_id")
    wm = {0: .9, 1: .85, 2: .9, 3: 1.0, 4: 1.25, 5: 1.4, 6: 1.15}
    base = items.loc[ids, "base_demand"].to_numpy()[:, None]
    mult = np.array([wm[d.weekday()] * (1.3 if ((d.month == 11 and d.day >= 20) or d.month == 12
                     or (d.month == 1 and d.day <= 2)) else 1.0) for d in dates])[None, :]
    expected = base * mult * np.where(P, 1.35, 1.0) * 0.97
    return float(np.abs(expected[:, start:end] - Y[:, start:end]).sum() / Y[:, start:end].sum())


def main():
    OUT.mkdir(exist_ok=True)
    demand = load_demand()
    items = load_items().set_index("item_id")
    ids, dates, Y, P = fc.build_panel(demand)

    # ---------------------------------------------------------- run backtests
    results = {m: fc.rolling_origin_backtest(Y, P, dates, m) for m in fc.MODELS}
    origins = results[DEFAULT_METHOD][0]
    half = len(origins) // 2
    span = (origins[0], origins[-1] + fc.HORIZON_DAYS)

    rows = []
    for m, (_, F, A) in results.items():
        for split, sl in [("all", slice(None)), ("first_half", slice(0, half)), ("second_half", slice(half, None))]:
            r = fc.error_metrics(F[sl], A[sl])
            rows.append({"model": m, "split": split, **r})
    metrics = pd.DataFrame(rows)
    metrics.round(4).to_csv(OUT / "demand_backtest_metrics.csv", index=False)
    allm = metrics[metrics.split == "all"].set_index("model")
    floor = noise_floor_wape(ids, dates, Y, P, *span)

    # ------------------------------------------- long table for the breakdowns
    def long(method):
        _, F, A = results[method]
        n_o, n_i, h = F.shape
        o_idx, i_idx, k_idx = np.meshgrid(range(n_o), range(n_i), range(h), indexing="ij")
        t = np.array(origins)[o_idx] + k_idx
        return pd.DataFrame({"f": F.ravel(), "a": A.ravel(), "date": dates[t.ravel()],
                             "family": items.loc[ids[i_idx.ravel()], "family"].to_numpy(),
                             "promo": P[i_idx.ravel(), t.ravel()]})
    ours, base = long(DEFAULT_METHOD), long(BASELINE_METHOD)

    def seg(df, key):
        g = df.groupby(key)
        return (g.apply(lambda x: (x.f - x.a).abs().sum() / x.a.sum(), include_groups=False))

    for df in (ours, base):
        df["period"] = np.where((df.date >= HOLIDAY[0]) & (df.date <= HOLIDAY[1]), "Holiday bump (Nov 20-Jan 2)", "Normal weeks")
        df["daytype"] = np.where(df.date.dt.dayofweek >= 4, "Fri-Sun", "Mon-Thu")
        df["promo_s"] = np.where(df.promo, "Promo days", "Non-promo days")
    seg_tables = {}
    for title, key in [("By category", "family"), ("By period", "period"),
                       ("By day type", "daytype"), ("By promotion", "promo_s")]:
        seg_tables[title] = pd.DataFrame({"Agent": seg(ours, key), "7-day average": seg(base, key)})
        seg_tables[title]["Error reduction"] = 1 - seg_tables[title]["Agent"] / seg_tables[title]["7-day average"]

    # ---------------------------------------------- interval calibration (out of sample)
    _, F, A = results[DEFAULT_METHOD]
    E = F - A
    cov_d, cov_w = [], []
    for j in range(8, len(origins)):
        e, f = E[:j], F[:j]
        sd = np.sqrt(((e / np.maximum(f, 1e-6)) ** 2).mean(axis=(0, 2)))[:, None]
        lo, hi = np.maximum(F[j] - Z80 * sd * F[j], 0), F[j] + Z80 * sd * F[j]
        cov_d.append(((A[j] >= lo) & (A[j] <= hi)).mean())
        ew, fw = e.sum(-1), f.sum(-1)
        sw = np.sqrt(((ew / np.maximum(fw, 1e-6)) ** 2).mean(axis=0))[:, None]
        fj, aj = F[j].sum(-1)[:, None], A[j].sum(-1)[:, None]
        cov_w.append(((aj >= np.maximum(fj - Z80 * sw * fj, 0)) & (aj <= fj + Z80 * sw * fj)).mean())
    coverage_daily, coverage_weekly = float(np.mean(cov_d)), float(np.mean(cov_w))

    # ----------------------------------------------------------------- chart
    make_chart(results, allm, floor, ids, dates, origins, Y)

    # ---------------------------------------------------------------- report
    write_report(allm, metrics, seg_tables, floor, coverage_daily, coverage_weekly,
                 origins, span, dates, len(ids))
    print(allm[["wape", "weekly_wape", "mae", "rmse", "bias_pct"]].round(4))
    print(f"noise floor WAPE {floor:.4f}; interval coverage daily {coverage_daily:.3f} weekly {coverage_weekly:.3f}")
    print("wrote", OUT)


def make_chart(results, allm, floor, ids, dates, origins, Y):
    INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#8a8984"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(SURF)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y" if ax is ax2 else "x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # left: error by model (one measure, one axis)
    order = allm.sort_values("wape", ascending=False).index.tolist()
    vals = [allm.loc[m, "wape"] * 100 for m in order]
    cols = [BLUE if m == DEFAULT_METHOD else (ORANGE if m == BASELINE_METHOD else NEUTRAL) for m in order]
    y = np.arange(len(order))
    ax1.barh(y, vals, color=cols, height=0.5)
    for yi, v in zip(y, vals):
        ax1.text(v + 0.4, yi, f"{v:.1f}%", va="center", fontsize=10, color=INK)
    ax1.axvline(floor * 100, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax1.set_ylim(-0.9, len(order) - 0.4)
    ax1.text(floor * 100 + 0.4, -0.62, f"noise floor {floor*100:.0f}% (best any model can do)",
             ha="left", va="center", fontsize=9, color=INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels([LABELS[m] for m in order], fontsize=9.5, color=INK)
    ax1.set_xlim(0, 38)
    ax1.set_xlabel("Average daily forecast error (WAPE, lower is better)", fontsize=9, color=MUTED)
    ax1.set_title("Forecast error by model", loc="left",
                  fontsize=11, color=INK, fontweight="bold")

    # right: item 1000, last 8 backtest weeks
    row = int(np.where(ids == 1000)[0][0])
    last = origins[-8:]
    xs, act, ours, base = [], [], [], []
    for o in last:
        oi = origins.index(o)
        xs += list(dates[o:o + fc.HORIZON_DAYS])
        act += list(results[DEFAULT_METHOD][2][oi, row])
        ours += list(results[DEFAULT_METHOD][1][oi, row])
        base += list(results[BASELINE_METHOD][1][oi, row])
    ax2.plot(xs, act, color=INK, lw=1.4, marker="o", ms=3.5, label="Actual sales")
    ax2.plot(xs, base, color=ORANGE, lw=2, label="7-day average")
    ax2.plot(xs, ours, color=BLUE, lw=2, label="Agent forecast")
    ax2.set_ylabel("Units per day", fontsize=9, color=MUTED)
    ax2.set_title("Item 1000, forecast one week ahead", loc="left",
                  fontsize=11, color=INK, fontweight="bold")
    ax2.legend(frameon=False, fontsize=9, loc="upper left", ncols=3, labelcolor=INK)
    ax2.set_ylim(0, max(act + ours + base) * 1.25)
    fig.autofmt_xdate(rotation=0, ha="center")
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(OUT / "demand_agent_evaluation.png", dpi=170, facecolor=SURF)
    plt.close(fig)


def pct(x):
    return f"{x*100:.1f}%"


def md_table(df, fmt):
    cols = list(df.columns)
    out = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        out.append(f"| {idx} | " + " | ".join(fmt(c, r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_report(allm, metrics, seg_tables, floor, cov_d, cov_w, origins, span, dates, n_items):
    base, ours = allm.loc[BASELINE_METHOD], allm.loc[DEFAULT_METHOD]
    gap_closed = (base.wape - ours.wape) / (base.wape - floor)
    first = (dates[origins[0]]).date(); last = dates[span[1] - 1].date()
    lines = [
        "# Demand Forecasting Agent: Week 2 Evaluation",
        "",
        f"Rolling backtest over {len(origins)} weekly forecasts ({first} to {last}), {n_items} items, "
        f"7-day horizon. Each forecast uses only data from before its start date.",
        "",
        "## Headline",
        f"- Average daily error (WAPE) fell from **{pct(base.wape)}** (Week 1 placeholder: 7-day average) "
        f"to **{pct(ours.wape)}** (agent), a **{(1 - ours.wape / base.wape)*100:.0f}% reduction**.",
        f"- The synthetic data has random day-to-day noise that no model can predict. A model that knew the "
        f"true patterns would still score about **{pct(floor)}**, so the agent closes "
        f"**{gap_closed*100:.0f}%** of the achievable gap. More complex models (ML) would gain little here.",
        f"- Over-forecasting bias is small: {ours.bias_pct*100:+.1f}% (positive = over-forecast).",
        f"- Weekly total error: {pct(ours.weekly_wape)} (vs {pct(base.weekly_wape)}), the number that "
        f"matters for a weekly order.",
        "",
        "![Evaluation chart](demand_agent_evaluation.png)",
        "",
        "## Model comparison (all backtest weeks)",
        "| Model | Daily WAPE | Weekly-total WAPE | MAE (units) | RMSE (units) | Bias |",
        "|---|---|---|---|---|---|",
    ]
    for m in allm.sort_values("wape").index:
        r = allm.loc[m]
        lines.append(f"| {LABELS[m]} | {pct(r.wape)} | {pct(r.weekly_wape)} | {r.mae:.2f} | {r.rmse:.2f} | {r.bias_pct*100:+.1f}% |")
    h = metrics[metrics.model.isin([DEFAULT_METHOD, "holt_winters"])]
    lines += [
        "",
        "**Model choice.** `level x weekday pattern` and Holt-Winters finished within noise of each other "
        f"(first half {pct(h[(h.model==DEFAULT_METHOD)&(h.split=='first_half')].wape.iloc[0])} vs "
        f"{pct(h[(h.model=='holt_winters')&(h.split=='first_half')].wape.iloc[0])}; second half "
        f"{pct(h[(h.model==DEFAULT_METHOD)&(h.split=='second_half')].wape.iloc[0])} vs "
        f"{pct(h[(h.model=='holt_winters')&(h.split=='second_half')].wape.iloc[0])}). The agent uses the "
        "simpler, easier-to-explain one. Tuning used only the first half of the weeks.",
        "",
        "## Where it helps and where it doesn't",
    ]
    for title, t in seg_tables.items():
        lines += ["", f"**{title}**", md_table(t, lambda c, v: pct(v)), ""]
    lines += [
        "## Uncertainty ranges",
        f"The agent returns an 80% range. Checked out of sample (range sized from earlier weeks only), "
        f"actual sales landed inside it **{pct(cov_d)}** of the time per day and **{pct(cov_w)}** per week. "
        f"Slightly above the 80% target, so ranges run a little wide (cautious) rather than too narrow.",
        "",
        "## Limitations (state these plainly)",
        "- **Synthetic data.** Patterns are simulated, so absolute error numbers will differ on real "
        "restaurant data. The method and the comparison against the baseline carry over; the exact percentages do not.",
        "- **Promotions are assumed known in advance** (restaurants plan them). With no planned promo the "
        "agent assumes none; an unplanned promo day is a forecast miss.",
        "- **One year of history.** The model cannot learn yearly seasonality, so it cannot anticipate the "
        "holiday bump; it can only adapt after demand has already shifted. The backtest starts Dec 24, so it "
        "sees only the last ~10 days of the bump and the drop after it. The bump's start (Nov 20) falls in the "
        "warm-up period and was not evaluated, so the 'By period' table above says little about that risk.",
        "- **Weekday pattern is learned per item** from 12 weeks, so a brand-new item (under 28 days of "
        "history) gets a flat average with no range instead.",
        "",
        "## Reproduce",
        "`python scripts/evaluate_demand_agent.py` regenerates this report, the chart, and "
        "`demand_backtest_metrics.csv`.",
    ]
    (OUT / "demand_agent_evaluation.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
