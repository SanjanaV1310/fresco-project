"""Manager/Orchestrator Agent -- built in Week 7 (see Project Plan, WBS #7).

Implements FR6 (Requirements Statement, Section 2.1):
"The system shall coordinate the sequence of agent calls (Demand ->
Inventory/Supplier -> Optimization -> Critic), assemble the final
order plan, and support a replan loop when new information requires
revisiting earlier steps."

This is the piece that ties the whole pipeline from the architecture
diagram together: Demand/Inventory/Supplier -> Optimization -> Critic
-> Final Order Plan -> Simulation/Action -> New Information -> REPLAN.
"""
from agents.base_agent import BaseAgent
from agents.demand_agent import DemandForecastingAgent
from agents.inventory_agent import InventoryAgent
from agents.supplier_agent import SupplierAgent
from agents.optimization_agent import OptimizationAgent
from agents.critic_agent import CriticAgent


class OrchestratorAgent(BaseAgent):
    name = "orchestrator_agent"

    def __init__(self):
        self.demand_agent = DemandForecastingAgent()
        self.inventory_agent = InventoryAgent()
        self.supplier_agent = SupplierAgent()
        self.optimization_agent = OptimizationAgent()
        self.critic_agent = CriticAgent()

    def run(self, item_id: int, current_stock: int = 0, **kwargs) -> dict:
        # TODO (Week 7): add the replan loop -- if the Critic objects,
        # feed its objection back into the Optimization Agent and retry
        # instead of stopping here.
        demand_forecast = self.demand_agent.run(item_id=item_id)
        inventory_status = self.inventory_agent.run(
            item_id=item_id, current_stock=current_stock
        )
        supplier_info = self.supplier_agent.run(item_id=item_id)
        optimization_result = self.optimization_agent.run(
            demand_forecast=demand_forecast,
            inventory_status=inventory_status,
            supplier_info=supplier_info,
        )
        critic_result = self.critic_agent.run(
            optimization_result=optimization_result,
            demand_forecast=demand_forecast,
            shelf_life_days=inventory_status["shelf_life_days"],
        )

        return {
            "item_id": item_id,
            "demand_forecast": demand_forecast,
            "inventory_status": inventory_status,
            "supplier_info": supplier_info,
            "optimization_result": optimization_result,
            "critic_result": critic_result,
            "final_order_units": (
                optimization_result["recommended_order_units"]
                if critic_result["approved"] else None
            ),
        }


if __name__ == "__main__":
    # Quick end-to-end smoke test using the Week 1 dataset.
    orchestrator = OrchestratorAgent()
    result = orchestrator.run(item_id=1000, current_stock=10)
    import json
    print(json.dumps(result, indent=2, default=str))
