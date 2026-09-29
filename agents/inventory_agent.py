"""Inventory Agent -- built in Week 3 (see Project Plan, WBS #3).

Implements FR2 (Requirements Statement, Section 2.1):
"The system shall track current on-hand inventory per SKU, maintain
a configurable safety-stock threshold per SKU, and flag items nearing
expiration."
"""
from agents.base_agent import BaseAgent
from agents.data_loader import load_inventory_params


class InventoryAgent(BaseAgent):
    name = "inventory_agent"

    def run(self, item_id: int, current_stock: int = 0, **kwargs) -> dict:
        params = load_inventory_params()
        row = params[params["item_id"] == item_id].iloc[0]

        below_safety_stock = current_stock < row["safety_stock_units"]

        return {
            "item_id": item_id,
            "current_stock": current_stock,
            "safety_stock_units": int(row["safety_stock_units"]),
            "reorder_point_units": int(row["reorder_point_units"]),
            "needs_reorder": bool(current_stock < row["reorder_point_units"]),
            "shelf_life_days": int(row["shelf_life_days"]),
            "rationale": "Below safety stock." if below_safety_stock
                         else "Stock above safety threshold.",
        }
