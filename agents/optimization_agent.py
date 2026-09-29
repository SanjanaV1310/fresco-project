"""Optimization Agent -- built in Week 5 (see Project Plan, WBS #5).

Implements FR4 (Requirements Statement, Section 2.1):
"The system shall compute a recommended order quantity per SKU that
satisfies budget, MOQ, lead-time, and safety-stock constraints while
minimizing expected waste and stockout risk."

Per the Risk Management Plan (R6), start with a simplified constraint
set and add constraints incrementally rather than attempting the full
joint optimization at once.
"""
from agents.base_agent import BaseAgent


class OptimizationAgent(BaseAgent):
    name = "optimization_agent"

    def run(self, demand_forecast: dict, inventory_status: dict,
            supplier_info: dict, **kwargs) -> dict:
        # TODO (Week 5): replace with real constraint-aware optimization.
        # Placeholder logic: order up to the reorder point, rounded up
        # to the supplier's MOQ.
        forecast = demand_forecast["forecast_units"]
        reorder_point = inventory_status["reorder_point_units"]
        current_stock = inventory_status["current_stock"]
        moq = supplier_info["moq_units"]

        raw_need = max(0, reorder_point - current_stock)
        order_qty = max(raw_need, moq) if raw_need > 0 else 0

        return {
            "item_id": demand_forecast["item_id"],
            "recommended_order_units": order_qty,
            "rationale": "Placeholder: order up to reorder point, "
                         "rounded to MOQ. Replace with real optimization "
                         "in Week 5.",
        }
