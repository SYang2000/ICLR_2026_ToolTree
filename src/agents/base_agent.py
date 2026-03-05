"""Abstract base class for tool-planning agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import ToolTreeConfig


class BaseAgent(ABC):
    """Abstract base class for tool-planning agents.

    All agent implementations (ToolTree, baselines) should inherit
    from this class and implement solve() and reset().
    """

    def __init__(self, config: ToolTreeConfig) -> None:
        """Initialize agent with configuration.

        Args:
            config: Full ToolTree configuration including MCTS params,
                    LLM settings, and benchmark info.
        """
        self.config = config

    @abstractmethod
    def solve(self, query: str, context: dict | None = None) -> dict:
        """Given a query, plan and execute tools to produce a final answer.

        Args:
            query: The user's natural language query.
            context: Optional initial context (images, prior results, etc.).

        Returns:
            Dict with keys:
                - "answer": str -- the final predicted answer.
                - "trajectory": list[dict] -- the tool call sequence used.
                - "metadata": dict -- additional info (rollouts, time, etc.).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state between tasks.

        Clears caches, resets internal counters, and prepares
        the agent for a new query.
        """
