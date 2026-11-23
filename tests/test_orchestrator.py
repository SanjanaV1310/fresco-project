"""Orchestrator (FR6): pipeline wiring, the Critic->Optimizer replan loop, new-information
replans, budget handling and the no-peeking guarantee."""
import json

import numpy as np
import pytest

from tests.helpers import START  # noqa: F401  (sets sys.path)
from agents.data_loader import load_demand
from agents.demand_agent import DemandForecastingAgent
from agents.optimization_agent import OptimizationAgent
from agents.orchestrator_agent import OrchestratorAgent

AS_OF = "2026-06-14"        # a Sunday; plan is for Monday 2026-06-15


@pytest.fixture(scope="module")
def demand_agent():
    return DemandForecastingAgent()


@pytest.fixture()
def orch(demand_agent):
    return OrchestratorAgent(demand_agent=demand_agent)


# ----------------------------------------------------------- pipeline wiring
def test_item_pipeline_contract(orch):
    r = orch.plan_item(1000, as_of=AS_OF, current_stock=10)
    for k in ("demand_forecast", "inventory_status", "supplier_info", "optimization_result",
              "critic_result", "status", "final_order_units", "explanation", "trace"):
        assert k in r
    assert r["decision_date"] == "2026-06-15"
    assert [t["agent"] for t in r["trace"][:3]] == ["supplier", "inventory", "demand"]
    assert {t["agent"] for t in r["trace"]} >= {"supplier", "inventory", "demand", "optimization", "critic"}
    assert isinstance(r["final_order_units"], int) and r["final_order_units"] >= 0
    # the forecast was asked to cover lead time + one cycle
    L = r["supplier_info"]["lead_time_days"]
    assert r["demand_forecast"]["horizon_days"] == L + 7


def test_every_agent_explains_itself(orch):
    r = orch.plan_item(1004, as_of=AS_OF)
    for t in r["trace"]:
        assert isinstance(t["summary"], str) and len(t["summary"]) > 20
    assert "CRITIC" in r["explanation"]


def test_week1_entry_point_still_works(orch):
    r = orch.run(item_id=1000, current_stock=10)
    assert r["item_id"] == 1000 and r["final_order_units"] >= 0


def test_agents_never_call_each_other():
    """All hand-offs go through the Orchestrator: no agent module imports another agent."""
    import agents.critic_agent as c, agents.optimization_agent as o, agents.inventory_agent as i
    import agents.supplier_agent as s_, agents.demand_agent as d
    names = {"demand_agent", "inventory_agent", "supplier_agent", "optimization_agent",
             "critic_agent", "orchestrator_agent"}
    for mod in (c, o, i, s_, d):
        imported = {ln.split()[1].split(".")[-1] for ln in open(mod.__file__).read().splitlines()
                    if ln.startswith("from agents.")}
        assert not (imported & names), (mod.__name__, imported & names)


# ------------------------------------------------------------- replan loop
class _GreedyOnce(OptimizationAgent):
    """Orders far too much on the first try, then obeys the Critic's bound."""
    def run(self, *a, constraints=None, **kw):
        p = super().run(*a, constraints=constraints, **kw)
        if not constraints:
            p = dict(p, recommended_order_units=p["recommended_order_units"] * 8 + 200,
                     binding_constraints=[])
        return p


class _Stubborn(OptimizationAgent):
    """Ignores every bound, so the loop can never converge."""
    def run(self, *a, constraints=None, **kw):
        p = super().run(*a, constraints=None, **kw)
        return dict(p, recommended_order_units=p["recommended_order_units"] * 8 + 200,
                    binding_constraints=[])


def test_replan_loop_fixes_a_bad_first_proposal(demand_agent):
    o = OrchestratorAgent(demand_agent=demand_agent, optimization_agent=_GreedyOnce())
    r = o.plan_item(1000, as_of=AS_OF, current_stock=0)         # shelf life 3 -> a huge order is wasteful
    assert r["status"] == "approved_after_revision" and r["rounds"] == 2
    assert len(r["revisions"]) == 1 and "EXPIRY" in r["revisions"][0]["objections"]
    good = OrchestratorAgent(demand_agent=demand_agent).plan_item(1000, as_of=AS_OF, current_stock=0)
    assert r["final_order_units"] < r["revisions"][0]["units"] / 4
    assert r["final_order_units"] == pytest.approx(good["final_order_units"], rel=0.35)
    assert "REPLAN" in r["explanation"]


def test_replan_loop_terminates_and_escalates(demand_agent):
    o = OrchestratorAgent(demand_agent=demand_agent, optimization_agent=_Stubborn(), max_rounds=3)
    r = o.plan_item(1000, as_of=AS_OF, current_stock=0)
    assert r["status"] == "needs_human_review" and r["needs_human_review"]
    assert r["rounds"] <= 3
    assert r["final_order_units"] >= 0       # best-so-far is still reported


def test_loop_respects_max_rounds(demand_agent):
    o = OrchestratorAgent(demand_agent=demand_agent, optimization_agent=_Stubborn(), max_rounds=1)
    assert o.plan_item(1000, as_of=AS_OF, current_stock=0)["rounds"] == 1


# --------------------------------------------------------------- whole week
def test_plan_week_covers_every_item_and_is_json_serialisable(orch):
    w = orch.plan_week(as_of=AS_OF)
    assert len(w["items"]) == 54 and w["summary"]["items_planned"] == 54
    assert set(w["orders"]) <= set(w["items"])
    assert all(v > 0 for v in w["orders"].values())
    assert w["summary"]["total_cost"] == pytest.approx(
        sum(it["optimization_result"]["order_cost"] for it in w["items"].values()), abs=0.01)
    json.dumps(w, default=str)                                   # demo/report can dump it


@pytest.mark.parametrize("budget", [14000, 9000, 4000])
def test_week_plan_respects_budget(orch, budget):
    w = orch.plan_week(as_of=AS_OF, budget=budget)
    assert w["summary"]["total_cost"] <= budget + 1e-6 and w["basket"]["approved"]
    free = orch.plan_week(as_of=AS_OF)
    assert w["summary"]["total_cost"] <= free["summary"]["total_cost"]


def test_budget_trimmed_items_are_explained(orch):
    w = orch.plan_week(as_of=AS_OF, budget=6000)
    trimmed = [it for it in w["items"].values()
               if set(it["optimization_result"]["binding_constraints"]) & {"budget_trimmed", "budget_dropped"}]
    assert trimmed and all("BUDGET" in it["explanation"] for it in trimmed)


def test_budget_change_replan_reruns_only_the_basket(orch):
    w = orch.plan_week(as_of=AS_OF, budget=9000)
    w2 = orch.replan(w, {"type": "budget_change", "budget": 14000})
    log = w2["replan_log"][-1]
    assert log["steps_rerun"] == ["budget", "critic (basket)"]
    assert w2["summary"]["total_cost"] > w["summary"]["total_cost"]
    assert w2["summary"]["total_cost"] <= 14000
    w3 = orch.replan(w2, {"type": "budget_change", "budget": 9000})    # and back down again
    assert w3["orders"] == w["orders"]


def test_supplier_update_replans_only_that_item(orch):
    w = orch.plan_week(as_of=AS_OF)
    item = 1010
    new = orch.replan(w, {"type": "supplier_update", "item_id": item,
                          "changes": {"moq_units": 600}, "reason": "min order raised"})
    changed = {c["item_id"] for c in new["replan_log"][-1]["orders_changed"]}
    assert changed <= {item}
    assert new["replan_log"][-1]["steps_rerun"] == ["supplier", "optimization", "critic", "budget"]
    assert new["items"][item]["supplier_info"]["moq_units"] == 600
    assert new["items"][item]["final_order_units"] in (0, 600) or \
           new["items"][item]["final_order_units"] >= 600
    # untouched items keep exactly their old plans
    assert all(new["orders"].get(i) == w["orders"].get(i) for i in w["items"] if i != item)


def test_lead_time_change_also_reruns_the_forecast(orch):
    w = orch.plan_week(as_of=AS_OF, item_ids=[1010])
    L = w["items"][1010]["supplier_info"]["lead_time_days"]
    new = orch.replan(w, {"type": "supplier_update", "item_id": 1010,
                          "changes": {"lead_time_days": L + 2}})
    assert "demand" in new["replan_log"][-1]["steps_rerun"]
    assert new["items"][1010]["demand_forecast"]["horizon_days"] == L + 2 + 7


def test_stock_correction_replan(orch):
    orch.inventory_agent.reset()
    orch.inventory_agent.receive(1004, 200, "2026-06-14")
    w = orch.plan_week(as_of=AS_OF, item_ids=[1004])
    n0 = w["orders"].get(1004, 0)
    new = orch.replan(w, {"type": "stock_correction", "item_id": 1004, "count": 0})
    assert new["orders"].get(1004, 0) > n0
    assert "count 0 vs ledger" in new["replan_log"][-1]["detail"]
    orch.inventory_agent.reset()


def test_unknown_update_type_rejected(orch):
    w = orch.plan_week(as_of=AS_OF, item_ids=[1000])
    with pytest.raises(ValueError):
        orch.replan(w, {"type": "weather"})


# ------------------------------------------------------------ trustworthiness
def test_plan_uses_only_history_before_the_decision_date():
    """Changing sales AFTER the as_of date must not change the plan (no peeking at the future)."""
    d = load_demand()
    d2 = d.copy()
    d2.loc[d2["date"] > AS_OF, "unit_sales"] *= 5
    a = OrchestratorAgent(demand_agent=DemandForecastingAgent(demand=d)).plan_week(as_of=AS_OF, item_ids=[1000, 1010, 1030])
    b = OrchestratorAgent(demand_agent=DemandForecastingAgent(demand=d2)).plan_week(as_of=AS_OF, item_ids=[1000, 1010, 1030])
    assert a["orders"] == b["orders"]


def test_planning_is_deterministic(orch):
    a = orch.plan_week(as_of=AS_OF, item_ids=[1000, 1020, 1040])["orders"]
    b = orch.plan_week(as_of=AS_OF, item_ids=[1000, 1020, 1040])["orders"]
    assert a == b


def test_planned_promotions_raise_the_order(orch):
    base = orch.plan_item(1020, as_of=AS_OF)["final_order_units"]
    days = [f"2026-06-{d}" for d in (17, 18, 19, 20)]
    promo = orch.plan_item(1020, as_of=AS_OF, planned_promotions=days)["final_order_units"]
    assert promo > base
