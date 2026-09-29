"""Supplier Agent -- built in Week 4 (see Project Plan, WBS #4).

Implements FR3 (Requirements Statement, Section 2.1):
"The system shall retrieve and maintain supplier price, minimum
order quantity, availability, and lead time for each SKU."
"""
from agents.base_agent import BaseAgent
from agents.data_loader import load_suppliers


class SupplierAgent(BaseAgent):
    name = "supplier_agent"

    def run(self, item_id: int, **kwargs) -> dict:
        suppliers = load_suppliers()
        row = suppliers[suppliers["item_id"] == item_id].iloc[0]

        return {
            "item_id": item_id,
            "supplier_name": row["supplier_name"],
            "unit_cost": float(row["unit_cost"]),
            "lead_time_days": int(row["lead_time_days"]),
            "moq_units": int(row["moq_units"]),
            "availability_pct": float(row["availability_pct"]),
            "rationale": f"Sourced from {row['supplier_name']}, "
                         f"{row['lead_time_days']}-day lead time.",
        }
