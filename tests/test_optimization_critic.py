"""Optimization Agent (FR4) and Critic Agent (FR5), on hand-built inputs where the right
answer is known. The Critic is tested on clearly-bad AND clearly-reasonable orders (Risk R10)."""
import pytest

from tests.helpers import make_forecast, make_inventory, make_supplier
from agents.critic_agent import CriticAgent
from agents.inventory_math import project_fifo
from agents.optimization_agent import OptimizationAgent, waste_risk

opt, critic = OptimizationAgent(), CriticAgent()


def plan(fc=None, inv=None, sup=None, **kw):
    fc, inv, sup = fc or make_forecast(), inv or make_inventory(), sup or make_supplier()
    return opt.run(demand_forecast=fc, inventory_status=inv, supplier_info=sup, **kw), fc, inv, sup


def review(p, fc, inv, sup):
    return critic.run(optimization_result=p, demand_forecast=fc, shelf_life_days=inv["shelf_life_days"],
                      inventory_status=inv, supplier_info=sup)


# ------------------------------------------------------- shared arithmetic
def test_project_fifo_expiry_and_stockout():
    r = project_fifo([(10, 1)], [], [4, 4, 4], shelf_life_days=5)
    assert r["sold"].tolist() == [4, 4, 0]            # stock expires end of day 1 ...
    assert r["expired"].tolist() == [0, 2, 0]         # ... 2 units wasted
    assert r["unmet"].tolist() == [0, 0, 4]


def test_project_fifo_arrival_is_fresh():
    r = project_fifo([], [(2, 10)], [0, 0, 5, 5], shelf_life_days=2)
    assert r["sold"].tolist() == [0, 0, 5, 5] and r["leftover"] == 0


# -------------------------------------------------------------- optimizer
def test_covers_forecast_with_a_cushion_for_durable_items():
    p, *_ = plan()                                    # 30/day, window = 7 days = 210
    assert p["recommended_order_units"] >= 210
    assert p["recommended_order_units"] <= 210 * 1.5
    assert p["service_level_target"] > 0.8            # durable item -> high service level
    assert 0 < p["expected"]["service_level"] <= 1


def test_orders_nothing_when_stock_covers_the_window():
    p, *_ = plan(inv=make_inventory(stock=1000))
    assert p["recommended_order_units"] == 0
    assert p["rationale"].startswith("No order this week")


def test_nets_off_in_transit_orders():
    base, *_ = plan()
    # arrives the day our window starts (offset 2 = lead time), so all 100 units count
    withtransit, *_ = plan(inv=make_inventory(in_transit=[{"units": 100, "arrives_on": "x", "offset": 2}]))
    assert withtransit["recommended_order_units"] == pytest.approx(
        base["recommended_order_units"] - 100, abs=3)
    # arriving a day earlier, 30 of them are sold before the window opens
    earlier, *_ = plan(inv=make_inventory(in_transit=[{"units": 100, "arrives_on": "x", "offset": 1}]))
    assert earlier["recommended_order_units"] > withtransit["recommended_order_units"]


def test_stock_that_expires_first_does_not_cover_later_days():
    """100 units on hand, but they all expire today: they cannot help a window that starts on day 2."""
    p, *_ = plan(inv=make_inventory(stock=100, shelf=10, last_offset=0))
    q_empty, *_ = plan(inv=make_inventory(shelf=10))
    assert p["recommended_order_units"] == q_empty["recommended_order_units"]


def test_shelf_life_caps_the_order():
    p, *_ = plan(inv=make_inventory(shelf=2), sup=make_supplier(moq=1))
    assert "shelf_life_cap" in p["binding_constraints"] and p["shelf_capped"]
    assert p["recommended_order_units"] <= 2 * 30 + 1          # two days of demand, not seven
    assert p["late_week_gap_units"] == pytest.approx(5 * 30, abs=1)
    assert p["expected"]["waste_units"] < 3


def test_perishables_get_lower_service_targets_than_durables():
    assert plan(inv=make_inventory(shelf=3))[0]["service_level_target"] < \
           plan(inv=make_inventory(shelf=200))[0]["service_level_target"]
    assert waste_risk(3) == 1.0 and waste_risk(200) == 0.02 and 0.02 < waste_risk(14) < 1


def test_supplier_shortfall_is_grossed_up():
    full, *_ = plan(sup=make_supplier(avail=1.0))
    short, *_ = plan(sup=make_supplier(avail=0.8))
    assert short["recommended_order_units"] == pytest.approx(full["recommended_order_units"] / 0.8, abs=2)


def test_moq_raise_when_margin_beats_waste():
    inv, fc = make_inventory(stock=315), make_forecast()        # short by only a few units
    p, *_ = plan(fc=fc, inv=inv, sup=make_supplier(moq=60, cost=1.0))   # cheap, high margin
    assert p["recommended_order_units"] == 60 and "moq_raise" in p["binding_constraints"]


def test_moq_skip_when_waste_beats_margin():
    inv = make_inventory(stock=205, shelf=3, price=2.2)
    p, *_ = plan(inv=inv, sup=make_supplier(moq=300, cost=2.0))
    assert p["recommended_order_units"] == 0 and "moq_skip" in p["binding_constraints"]


def test_critic_bounds_are_honoured():
    p, *_ = plan(constraints={"max_units": 100})
    assert p["recommended_order_units"] == 100 and "max_units" in p["binding_constraints"]
    p, *_ = plan(constraints={"min_units": 500})
    assert p["recommended_order_units"] == 500
    p, *_ = plan(constraints={"max_units": 5}, sup=make_supplier(moq=10))
    assert p["recommended_order_units"] == 0           # below the MOQ -> no order


def test_mismatched_dates_are_rejected():
    with pytest.raises(ValueError, match="same decision date"):
        plan(fc=make_forecast(start="2026-10-02"))


def test_works_with_other_cycle_lengths():
    p, *_ = plan(fc=make_forecast(cycle=3), cycle_days=3)
    assert p["cycle_days"] == 3 and 90 <= p["recommended_order_units"] <= 90 * 1.5


def test_missing_track_record_is_flagged():
    fc = make_forecast()
    fc["error_scale"]["window_relative"] = None
    p, *_ = plan(fc=fc)
    assert p["low_confidence"] and "assumed" in p["rationale"]


# ----------------------------------------------------------------- budget
def _basket(n=6):
    plans = []
    for k in range(n):
        p, *_ = plan(sup=make_supplier(cost=1.0 + k * 0.5, moq=5, item_id=k),
                     fc=make_forecast(item_id=k), inv=make_inventory(price=4.0, item_id=k))
        plans.append(p)
    return plans


def test_budget_untouched_when_it_fits():
    plans = _basket()
    new, rep = opt.allocate_budget(plans, 1e9)
    assert rep["action"] == "none" and [p["recommended_order_units"] for p in new] == \
           [p["recommended_order_units"] for p in plans]


@pytest.mark.parametrize("frac", [0.95, 0.85, 0.6, 0.3])
def test_budget_is_never_exceeded(frac):
    plans = _basket()
    total = sum(p["order_cost"] for p in plans)
    new, rep = opt.allocate_budget(plans, total * frac)
    assert sum(p["order_cost"] for p in new) <= total * frac + 1e-6
    assert rep["cost_after"] == pytest.approx(sum(p["order_cost"] for p in new))
    assert all(p["recommended_order_units"] <= q["recommended_order_units"] for p, q in zip(new, plans))


def test_budget_trims_cushions_before_dropping_items():
    plans = _basket()
    total = sum(p["order_cost"] for p in plans)
    base_cost = sum(p["base_units"] * p["unit_cost"] for p in plans)
    new, rep = opt.allocate_budget(plans, (total + base_cost) / 2)
    assert rep["action"] == "trimmed_cushions" and rep["dropped"] == []
    assert all(p["recommended_order_units"] >= p["base_units"] for p in new)


def test_budget_drops_least_profitable_per_dollar_first():
    plans = _basket()                       # item 5 has the highest unit cost -> lowest margin per $
    base_cost = sum(p["base_units"] * p["unit_cost"] for p in plans)
    new, rep = opt.allocate_budget(plans, base_cost * 0.8)
    assert rep["action"] == "dropped_items" and rep["dropped"][0] == 5
    assert "budget_dropped" in new[5]["binding_constraints"]


def test_budget_does_not_mutate_inputs():
    plans = _basket()
    before = [p["recommended_order_units"] for p in plans]
    opt.allocate_budget(plans, 10)
    assert [p["recommended_order_units"] for p in plans] == before


# ----------------------------------------------------------------- critic
def test_critic_approves_a_reasonable_order():
    p, fc, inv, sup = plan()
    v = review(p, fc, inv, sup)
    assert v["approved"] and v["verdict"] in ("approve", "approve_with_warnings")
    assert v["objection"] is None and v["suggested"] == {}


def test_critic_rejects_a_clearly_bad_order_with_a_specific_fix():
    """A perishable ordered at 10x what can be sold before it expires."""
    fc, inv, sup = make_forecast(), make_inventory(shelf=5), make_supplier(moq=1)
    p, *_ = plan(fc=fc, inv=inv, sup=sup)
    bad = dict(p, recommended_order_units=1500)
    v = review(bad, fc, inv, sup)
    assert not v["approved"] and v["verdict"] == "object"
    assert v["objections"][0]["code"] == "EXPIRY"
    assert "wasted" in v["objection"] and "1500" in v["objection"]
    cap = v["suggested"]["max_units"]
    assert cap < 1500
    assert review(dict(p, recommended_order_units=cap), fc, inv, sup)["approved"]   # the fix works


def test_critic_rejects_a_far_too_small_order():
    fc, inv, sup = make_forecast(), make_inventory(shelf=60), make_supplier(moq=1)
    p, *_ = plan(fc=fc, inv=inv, sup=sup)
    v = review(dict(p, recommended_order_units=20), fc, inv, sup)
    assert not v["approved"] and v["objections"][0]["code"] == "UNDERSTOCK"
    assert v["suggested"]["min_units"] > 20


def test_critic_does_not_nag_when_shortfall_is_unavoidable():
    """Shelf-life-limited item: running short late in the week is a trade-off, not an error."""
    fc, inv, sup = make_forecast(), make_inventory(shelf=2), make_supplier(moq=1)
    p, *_ = plan(fc=fc, inv=inv, sup=sup)
    v = review(p, fc, inv, sup)
    assert v["approved"] and any(w["code"] == "UNDERSTOCK_ACCEPTED" for w in v["warnings"])
    assert "shelf life" in v["rationale"]


def test_critic_warns_instead_of_objecting_to_moq_forced_waste():
    fc, inv, sup = make_forecast(), make_inventory(stock=300, shelf=4, price=6.0, last_offset=3), make_supplier(moq=80, cost=1.0)
    p, *_ = plan(fc=fc, inv=inv, sup=sup)
    assert "moq_raise" in p["binding_constraints"]
    v = review(p, fc, inv, sup)
    assert v["approved"]


def test_critic_rejects_invalid_quantities():
    p, fc, inv, sup = plan()
    for bad in (-5, 3.5):
        v = review(dict(p, recommended_order_units=bad), fc, inv, sup)
        assert not v["approved"] and v["objections"][0]["code"] == "INVALID"


def test_critic_is_independent_of_the_optimizers_own_numbers():
    """Tampering with the Optimizer's self-reported 'expected' block must not change the verdict."""
    fc, inv, sup = make_forecast(), make_inventory(shelf=5), make_supplier(moq=1)
    p, *_ = plan(fc=fc, inv=inv, sup=sup)
    bad = dict(p, recommended_order_units=1500, expected={"waste_units": 0, "unmet_units": 0,
                                                         "service_level": 1.0, "leftover_units": 0})
    assert not review(bad, fc, inv, sup)["approved"]


def test_critic_basket_review():
    plans = _basket()
    total = sum(p["order_cost"] for p in plans)
    assert critic.review_basket(plans, total + 1)["approved"]
    r = critic.review_basket(plans, total / 2)
    assert not r["approved"] and "over" in r["rationale"]
    assert critic.review_basket(plans, None)["approved"]
