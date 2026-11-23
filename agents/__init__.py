"""FRESCO agent package -- the six cooperating agents for the Multi-Agent
Inventory & Waste Reduction System. All six are implemented:

  Demand (FR1), Inventory (FR2), Supplier (FR3), Optimization (FR4),
  Critic (FR5), Orchestrator (FR6, wires the others together and runs the replan loop).

agents/forecasting.py and agents/inventory_math.py are shared pure-function helpers.
Agents never import each other; all hand-offs go through the Orchestrator.
"""
