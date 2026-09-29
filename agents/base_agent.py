"""Shared base class every agent inherits from.

Keeping a common interface now (Week 1) means the Manager/Orchestrator
(built in Week 7) can call every agent the same way, regardless of
which sprint built it -- this is the direct fix for the "agent
hand-off mismatch" risk (R1) flagged in the Risk Management Plan.
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Every agent implements run() and returns a plain dict, so the
    Orchestrator never needs agent-specific unpacking logic."""

    name: str = "base_agent"

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """Execute this agent's logic and return its output as a dict.

        Convention: always include a `rationale` key explaining *why*,
        not just *what* -- this satisfies NFR1 (Explainability) from
        the Requirements Statement, and is what lets the Critic Agent
        and a human reviewer both sanity-check the output.
        """
        raise NotImplementedError
