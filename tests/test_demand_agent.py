"""Week 2 tests: Demand Forecasting Agent (FR1).

Covers the output contract the rest of the pipeline relies on, correctness
on data where the right answer is known, and the guarantees that make the
accuracy numbers trustworthy (no peeking at the future, honest uncertainty).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from agents import forecasting as fc
from agents.data_loader import load_demand
from agents.demand_agent import DemandForecastingAgent, DEFAULT_METHOD, BASELINE_METHOD
from agents.orchestrator_agent import OrchestratorAgent


# ----------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def demand():
    return load_demand()


@pytest.fixture(scope="module")
def agent(demand):
    return DemandForecastingAgent(demand=demand)


def make_demand(values, start="2026-01-05", item_id=1, promo=None):
    """Tiny single-item demand table. 2026-01-05 is a Monday."""
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({
        "date": dates, "store_nbr": "STORE_1", "item_id": item_id,
        "unit_sales": values,
        "onpromotion": [False] * len(values) if promo is None else promo,
    })


# ------------------------------------------------------- output contract
def test_output_contract(agent):
    r = agent.run(item_id=1000)
    assert r["item_id"] == 1000
    assert r["forecast_unit"] == "units_per_day"
    assert r["method"] == DEFAULT_METHOD
    assert isinstance(r["rationale"], str) and len(r["rationale"]) > 40

    days = r["daily_forecast"]
    assert len(days) == r["horizon_days"] == 7
    dates = [pd.Timestamp(d["date"]) for d in days]
    assert dates[0] == pd.Timestamp("2026-10-01")            # day after the last data
    assert all((b - a).days == 1 for a, b in zip(dates, dates[1:]))

    # headline fields are consistent with the daily detail
    assert r["forecast_units"] == pytest.approx(np.mean([d["units"] for d in days]), abs=0.1)
    assert r["horizon_total_units"] == pytest.approx(sum(d["units"] for d in days), abs=0.5)

    # ranges bracket the forecast
    for d in days:
        assert d["lower"] <= d["units"] <= d["upper"]
    ti = r["total_interval"]
    assert ti["lower"] <= r["horizon_total_units"] <= ti["upper"]


def test_all_items_give_valid_forecasts(agent, demand):
    for item_id in sorted(demand["item_id"].unique()):
        r = agent.run(item_id=int(item_id))
        vals = [d["units"] for d in r["daily_forecast"]]
        assert all(np.isfinite(vals)) and min(vals) >= 0, item_id
        assert r["forecast_units"] >= 0


def test_unknown_item_and_method_raise(agent):
    with pytest.raises(ValueError, match="Unknown item_id"):
        agent.run(item_id=999999)
    with pytest.raises(ValueError, match="Unknown method"):
        agent.run(item_id=1000, method="crystal_ball")
    with pytest.raises(ValueError, match="before the first date"):
        agent.run(item_id=1000, as_of="2020-01-01")


# ---------------------------------------------------- correctness (known answers)
def test_learns_weekday_pattern():
    """Saturdays sell exactly 2x other days; the forecast must reflect that."""
    # Mon..Sun, Saturday (index 5) doubled; 20 weeks, no noise.
    week = [10, 10, 10, 10, 10, 20, 10]
    r = DemandForecastingAgent(demand=make_demand(week * 20)).run(item_id=1)
    by_day = {d["weekday"]: d["units"] for d in r["daily_forecast"]}
    assert by_day["Sat"] == pytest.approx(20, rel=0.02)
    assert by_day["Wed"] == pytest.approx(10, rel=0.02)
    # a flat 7-day average cannot do this -- that is the point of the upgrade
    flat = DemandForecastingAgent(demand=make_demand(week * 20)).run(
        item_id=1, method=BASELINE_METHOD)
    assert len({d["units"] for d in flat["daily_forecast"]}) == 1


def test_follows_a_level_shift():
    """If demand permanently steps up, the forecast must follow within weeks."""
    vals = [10] * 70 + [20] * 56
    r = DemandForecastingAgent(demand=make_demand(vals)).run(item_id=1)
    assert r["forecast_units"] == pytest.approx(20, rel=0.05)


def test_all_zero_history_forecasts_zero():
    r = DemandForecastingAgent(demand=make_demand([0] * 60)).run(item_id=1)
    assert r["forecast_units"] == 0
    assert all(np.isfinite([d["units"] for d in r["daily_forecast"]]))


def test_planned_promotion_lifts_only_that_day(agent):
    plain = agent.run(item_id=1000)
    promo_day = "2026-10-03"
    promo = agent.run(item_id=1000, planned_promotions=[promo_day])
    for a, b in zip(plain["daily_forecast"], promo["daily_forecast"]):
        if a["date"] == promo_day:
            assert b["promo"] is True
            assert b["units"] == pytest.approx(a["units"] * plain["promo_lift_used"], rel=0.02)
        else:
            assert b["units"] == a["units"]


def test_promo_lift_estimation():
    rng = np.random.default_rng(0)
    T, n = 100, 6
    dates = pd.date_range("2026-01-05", periods=T, freq="D")
    P = np.zeros((n, T), bool)
    P[:, ::9] = True                                  # promo every 9th day
    Y = np.full((n, T), 10.0) * np.where(P, 1.5, 1.0)  # exact 1.5x lift
    assert fc.estimate_promo_lift(Y, P, dates.dayofweek.to_numpy()) == pytest.approx(1.5, rel=0.01)
    assert fc.estimate_promo_lift(Y, np.zeros_like(P), dates.dayofweek.to_numpy()) == 1.0


def test_cold_start_falls_back_gracefully():
    r = DemandForecastingAgent(demand=make_demand([5] * 10)).run(item_id=1)
    assert r["method"] == "average_fallback"
    assert r["forecast_units"] == 5
    assert r["total_interval"] is None
    assert "Only 10 days" in r["rationale"]


def test_missing_days_are_treated_as_zero_sales():
    """Favorita convention: no row for a day means nothing sold."""
    df = make_demand([10] * 60)
    df = df.drop(index=[10, 11, 12])                  # three days with no rows
    ids, dates, Y, P = fc.build_panel(df)
    assert Y.shape == (1, 60)
    assert Y[0, 10:13].tolist() == [0, 0, 0]
    DemandForecastingAgent(demand=df).run(item_id=1)  # must not crash


# ------------------------------------- trustworthiness of the accuracy numbers
def test_no_peeking_at_the_future(demand):
    """Forecasting as of a past date must not depend on anything after it.
    We replace every later row with absurd values and require an identical result."""
    as_of = pd.Timestamp("2026-06-30")
    clean = DemandForecastingAgent(demand=demand).run(item_id=1007, as_of=as_of)

    tampered = demand.copy()
    later = pd.to_datetime(tampered["date"]) > as_of
    tampered.loc[later, "unit_sales"] = 99999
    tampered.loc[later, "onpromotion"] = True
    assert DemandForecastingAgent(demand=tampered).run(item_id=1007, as_of=as_of) == clean

    # and equals an agent that only ever had data up to that date
    truncated = demand[~later]
    assert DemandForecastingAgent(demand=truncated).run(item_id=1007) == clean


def test_agent_beats_the_week1_baseline(demand):
    """The upgrade has to be a real improvement, not just more code."""
    ids, dates, Y, P = fc.build_panel(demand)
    _, F, A = fc.rolling_origin_backtest(Y, P, dates, DEFAULT_METHOD)
    _, Fb, Ab = fc.rolling_origin_backtest(Y, P, dates, BASELINE_METHOD)
    ours, base = fc.error_metrics(F, A), fc.error_metrics(Fb, Ab)
    assert ours["wape"] < base["wape"] * 0.90          # >=10% lower error
    assert ours["weekly_wape"] < base["weekly_wape"]
    assert abs(ours["bias_pct"]) < 0.05                # no systematic over/under-forecast


def test_prediction_intervals_are_honest(demand):
    """Replay 8 past weeks through the agent itself. Its 80% weekly range
    should contain the real outcome roughly 80% of the time (not 40%, not 100%)."""
    agent = DemandForecastingAgent(demand=demand)
    ids = [int(i) for i in sorted(demand["item_id"].unique())]
    last = pd.Timestamp(demand["date"].max())
    hits = total = 0
    for weeks_back in range(2, 10):
        as_of = last - pd.Timedelta(days=7 * weeks_back)
        for item_id in ids:
            r = agent.run(item_id=item_id, as_of=as_of)
            actual = demand[(demand["item_id"] == item_id)
                            & (pd.to_datetime(demand["date"]) > as_of)
                            & (pd.to_datetime(demand["date"]) <= as_of + pd.Timedelta(days=7))
                            ]["unit_sales"].sum()
            hits += r["total_interval"]["lower"] <= actual <= r["total_interval"]["upper"]
            total += 1
    assert 0.72 <= hits / total <= 0.95, hits / total


# ---------------------------------------------------------- pipeline wiring
def test_orchestrator_uses_new_forecaster_end_to_end():
    result = OrchestratorAgent().run(item_id=1000, current_stock=10)
    fcst = result["demand_forecast"]
    assert fcst["method"] == DEFAULT_METHOD
    assert fcst["forecast_units"] > 0
    assert result["optimization_result"]["recommended_order_units"] >= 0
    assert isinstance(result["critic_result"]["approved"], bool)
