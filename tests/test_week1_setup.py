"""Sanity-check tests for Week 1: confirms the environment and data
are wired together correctly before any real agent logic is built.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data_loader import load_all, validate_consistency
from agents.orchestrator_agent import OrchestratorAgent


def test_data_loads():
    data = load_all()
    assert len(data["items"]) == 54
    assert len(data["demand"]) == 54 * 365
    assert len(data["inventory_params"]) == 54
    assert len(data["suppliers"]) == 54


def test_data_keys_consistent():
    data = load_all()
    validate_consistency(data)  # raises AssertionError if inconsistent


def test_orchestrator_end_to_end_smoke():
    """Not a real correctness test -- just confirms the six-agent
    pipeline runs start to finish without crashing, using Week 1's
    placeholder logic. Each agent's real logic replaces its stub in
    Weeks 2-7."""
    orchestrator = OrchestratorAgent()
    result = orchestrator.run(item_id=1000, current_stock=10)

    assert result["item_id"] == 1000
    assert "demand_forecast" in result
    assert "critic_result" in result
    assert result["optimization_result"]["recommended_order_units"] >= 0


if __name__ == "__main__":
    test_data_loads()
    test_data_keys_consistent()
    test_orchestrator_end_to_end_smoke()
    print("All Week 1 sanity checks passed.")
