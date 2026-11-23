"""Simulation layer (FR7): the environment obeys the rules, policies are comparable, metrics are right."""
import numpy as np
import pytest

from tests.helpers import START  # noqa: F401  (sets sys.path)
from simulation.environment import Simulator
from simulation.metrics import by_family, paired_bootstrap, summarize, weekly_table
from simulation.policies import FrescoPolicy, OrderUpToBaseline, ReorderPointBaseline


@pytest.fixture(scope="module")
def sim():
    return Simulator()


class _Nothing:
    name = "never orders"
    def decide(self, ctx):
        return {}


class _Flood:
    name = "orders a lot of everything"
    def decide(self, ctx):
        return {int(i): 100000 for i in sim_ids}


sim_ids = Simulator().item_ids


def test_schedule_starts_on_a_monday_with_enough_history(sim):
    t = sim.first_decision_index()
    assert sim.dates[t].dayofweek == 0 and t >= 133
    assert sim.max_weeks() == 32


@pytest.mark.parametrize("policy", [ReorderPointBaseline(), OrderUpToBaseline(), FrescoPolicy(), _Flood()])
def test_units_are_conserved(sim, policy):
    r = sim.run(policy, n_weeks=4)
    assert r.conservation_error == 0
    assert (r.sold >= 0).all() and (r.wasted >= 0).all() and (r.lost >= 0).all()


def test_never_ordering_sells_nothing(sim):
    r = sim.run(_Nothing(), n_weeks=2)
    assert r.sold.sum() == 0 and r.wasted.sum() == 0 and r.delivered.sum() == 0
    assert r.lost.sum() == pytest.approx(sim.Y[:, sim.first_decision_index():sim.first_decision_index() + 14].sum())


def test_demand_is_the_same_whatever_the_policy(sim):
    a, b = sim.run(_Nothing(), n_weeks=2), sim.run(OrderUpToBaseline(), n_weeks=2)
    assert np.array_equal(a.demand, b.demand)


def test_over_ordering_durable_items_serves_everything_but_wastes_perishables(sim):
    r = sim.run(_Flood(), n_weeks=4)
    fam = by_family(r, slice(1, 4))               # skip week 0: nothing has arrived yet
    assert fam.loc["BEVERAGES", "fill_rate"] > 0.97 and fam.loc["BEVERAGES", "waste_rate"] == 0
    assert fam.loc["SEAFOOD", "waste_rate"] > 0.5


def test_environment_enforces_the_supplier_minimum(sim):
    class OneUnit:
        name = "one unit"
        def decide(self, ctx):
            return {1000: 1}
    r = sim.run(OneUnit(), n_weeks=1)
    assert r.ordered[0].sum() == 21                       # item 1000 has MOQ 21


def test_supplier_luck_is_shared_between_policies(sim):
    """Same seed -> same week/item gets the same random draw whichever policy asks."""
    a, b = sim.run(OrderUpToBaseline(), n_weeks=3), sim.run(OrderUpToBaseline(), n_weeks=3)
    assert np.array_equal(a.delivered, b.delivered)
    other = Simulator(seed=1).run(OrderUpToBaseline(), n_weeks=3)
    assert not np.array_equal(a.delivered, other.delivered)


def test_deliveries_fall_slightly_short_on_average(sim):
    r = sim.run(OrderUpToBaseline(), n_weeks=8)
    ratio = r.delivered.sum() / r.ordered.sum()
    assert 0.9 < ratio <= 1.0            # items 1.0 -> ~0.96 availability, but arrivals lag the weeks ordered


def test_policies_only_see_past_demand_through_their_context(sim):
    seen = []
    class Spy:
        name = "spy"
        def decide(self, ctx):
            seen.append(ctx["as_of"])
            return {}
    sim.run(Spy(), n_weeks=3)
    d0 = sim.dates[sim.first_decision_index()]
    assert all(s < d0 + np.timedelta64(21, "D") for s in seen)
    assert seen[0] == d0 - np.timedelta64(1, "D")


def test_twice_weekly_ordering_days(sim):
    cycles = []
    class Spy:
        name = "spy"
        def decide(self, ctx):
            cycles.append((ctx["decision_date"].weekday(), ctx["cycle_days"]))
            return {}
    sim.run(Spy(), n_weeks=2, review_days=(0, 3))
    assert cycles == [(0, 3), (3, 4), (0, 3), (3, 4)]


# ---------------------------------------------------------------- baselines
def test_baselines_return_valid_orders(sim):
    class Probe:
        name = "probe"
        def decide(self, ctx):
            for pol in (ReorderPointBaseline(), OrderUpToBaseline()):
                o = pol.decide(ctx)
                assert all(isinstance(q, int) and q >= int(ctx["supplier"].terms[i]["moq_units"])
                           for i, q in o.items())
            return {}
    sim.run(Probe(), n_weeks=1)


def test_order_up_to_baseline_is_shelf_life_aware(sim):
    r = sim.run(OrderUpToBaseline(), n_weeks=6)
    naive = sim.run(OrderUpToBaseline(shelf_cap=False), n_weeks=6)
    assert summarize(weekly_table(naive))["waste_rate"] > 2 * summarize(weekly_table(r))["waste_rate"]


# ------------------------------------------------------------------ metrics
def test_margin_formula(sim):
    r = sim.run(OrderUpToBaseline(), n_weeks=3)
    wk = weekly_table(r)
    expected = ((r.sold * (r.unit_price - r.unit_cost)[:, None]).sum()
                - (r.wasted * r.unit_cost[:, None]).sum())
    assert wk["margin"].sum() == pytest.approx(expected)
    s = summarize(wk)
    assert s["fill_rate"] + s["stockout_rate"] == pytest.approx(1.0)
    assert by_family(r)["margin"].sum() == pytest.approx(expected)


def test_paired_bootstrap_detects_a_real_difference_and_a_null():
    rng = np.random.default_rng(1)
    base = rng.normal(100, 10, 30)
    real = paired_bootstrap(base + 5 + rng.normal(0, 1, 30), base)
    assert real["lo"] > 0 and real["share_weeks_better"] > 0.9
    null = paired_bootstrap(base + rng.normal(0, 1, 30), base)
    assert null["lo"] < 0 < null["hi"]


# ------------------------------------------------ the acceptance-criteria check
def test_fresco_beats_the_sow_baseline_on_waste_and_margin(sim):
    """SOW 1.6: measurable improvement over the reorder-point rule on waste or stockouts."""
    f = summarize(weekly_table(sim.run(FrescoPolicy(), n_weeks=8)).iloc[2:])
    b = summarize(weekly_table(sim.run(ReorderPointBaseline(), n_weeks=8)).iloc[2:])
    assert f["waste_rate"] < b["waste_rate"] / 5
    assert f["fill_rate"] > b["fill_rate"]
    assert f["margin_after_waste"] > b["margin_after_waste"]
