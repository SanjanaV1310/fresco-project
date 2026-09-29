"""Demand Forecasting Agent -- built in Week 2 (see Project Plan, WBS #2).

Implements FR1 (Requirements Statement, Section 2.1):
"The system shall predict expected demand for each SKU over the
coming ordering period, using historical/synthetic sales data."
"""
from agents.base_agent import BaseAgent
from agents.data_loader import load_demand


class DemandForecastingAgent(BaseAgent):
    name = "demand_forecasting_agent"

    def run(self, item_id: int, **kwargs) -> dict:
        # TODO (Week 2): replace with a real forecasting model
        # (e.g., moving average, exponential smoothing, or ML model)
        # trained on this item's history in demand.csv.
        demand = load_demand()
        item_history = demand[demand["item_id"] == item_id]
        naive_forecast = item_history["unit_sales"].tail(7).mean()

        return {
            "item_id": item_id,
            "forecast_units": round(naive_forecast, 1),
            "rationale": "Placeholder: 7-day trailing average. "
                         "Replace with real model in Week 2.",
        }
