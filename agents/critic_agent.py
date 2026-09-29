"""Critic Agent -- built in Week 6 (see Project Plan, WBS #6).

Implements FR5 (Requirements Statement, Section 2.1):
"The system shall review the Optimization Agent's proposed order and
flag orders that appear inconsistent with forecasted demand... returning
either an approval or a specific objection."

Per the Risk Management Plan (R10), test this explicitly against both
a clearly-bad order and a clearly-reasonable order before integrating.
"""
from agents.base_agent import BaseAgent


class CriticAgent(BaseAgent):
    name = "critic_agent"

    def run(self, optimization_result: dict, demand_forecast: dict,
            shelf_life_days: int, **kwargs) -> dict:
        # TODO (Week 6): replace with real objection logic that
        # weighs shelf life against forecasted consumption.
        order_qty = optimization_result["recommended_order_units"]
        forecast = demand_forecast["forecast_units"]
        expected_consumption = forecast * shelf_life_days

        if order_qty > expected_consumption * 1.5:
            return {
                "item_id": optimization_result["item_id"],
                "approved": False,
                "objection": (
                    f"Optimizer proposed {order_qty} units, but forecasted "
                    f"demand suggests only ~{expected_consumption:.0f} will "
                    f"be consumed before expiration."
                ),
            }
        return {
            "item_id": optimization_result["item_id"],
            "approved": True,
            "objection": None,
        }
