"""ToolTree Agent -- main orchestrator for MCTS-based tool planning.

Ties together all components: ToolTreeSearch, ToolManager, LLMJudge,
ToolRegistry, and the Answer Predictor to solve multi-tool queries.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.config import ToolTreeConfig


class ToolTreeAgent(BaseAgent):
    """Main ToolTree agent that orchestrates MCTS-based tool planning.

    Components initialized:
        - planner_llm: LLM client for action generation.
        - judge_llm: LLM client for pre/post evaluation.
        - tool_registry: Registry of available tool cards.
        - tool_manager: Tool execution with caching.
        - judge: LLM judge for dual evaluation.
        - searcher: MCTS search engine.
    """

    def __init__(self, config: ToolTreeConfig) -> None:
        """Initialize all components from config.

        Sets up LLM clients, tool registry, tool manager, LLM judge,
        and the MCTS searcher.

        Args:
            config: Full ToolTree configuration.
        """
        raise NotImplementedError

    def solve(self, query: str, context: dict | None = None) -> dict:
        """Run ToolTree on a single query.

        Steps:
            1. Retrieve or load available tools (closed-set or open-set).
            2. Run MCTS search to find the best tool trajectory.
            3. Use the answer predictor to generate the final answer
               from the best trajectory.
            4. Return the answer with trajectory and metadata.

        Args:
            query: The user's natural language query.
            context: Optional initial context.

        Returns:
            Dict with keys:
                - "answer": str -- predicted answer.
                - "trajectory": list[dict] -- optimal tool chain found by MCTS.
                - "reward": float -- Q-value of the best trajectory.
                - "rollouts_used": int -- number of MCTS rollouts performed.
        """
        raise NotImplementedError

    def _get_available_tools(self, query: str) -> list[dict]:
        """Get available tools for the current query.

        For closed-set benchmarks (GTA, m&m): returns all tools in the registry.
        For open-set benchmarks (ToolBench, RestBench): retrieves top-K tools
        relevant to the query using the tool registry's retrieval method.

        Args:
            query: The user query for tool retrieval.

        Returns:
            List of tool card dicts.
        """
        raise NotImplementedError

    def _predict_answer(self, query: str, trajectory: list[dict]) -> str:
        """Use the answer predictor LLM to generate the final answer.

        Takes the best MCTS trajectory (sequence of tool calls and outputs)
        and prompts the LLM to synthesize a final answer (Section 3.1 --
        Answer Predictor).

        Args:
            query: The original user query.
            trajectory: The best tool chain from MCTS search.

        Returns:
            The predicted answer string.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear caches and reset state between tasks.

        Clears the tool execution cache and any internal counters.
        """
        raise NotImplementedError
