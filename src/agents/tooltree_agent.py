"""ToolTree Agent -- main orchestrator for MCTS-based tool planning.

Ties together all components: ToolTreeSearch, ToolManager, LLMJudge,
ToolRegistry, and the Answer Predictor to solve multi-tool queries.
"""

from __future__ import annotations

from dataclasses import asdict

from src.agents.base_agent import BaseAgent
from src.config import ToolTreeConfig
from src.evaluation.judge import LLMJudge
from src.llm.client import LLMClient
from src.mcts.tree_search import ToolTreeSearch
from src.prompts.answer_prompt import (
    ANSWER_PREDICTOR_SYSTEM_PROMPT,
    build_answer_message,
)
from src.tools.tool_manager import ToolManager
from src.tools.tool_registry import ToolCard, ToolRegistry


class ToolTreeAgent(BaseAgent):
    """Main ToolTree agent that orchestrates MCTS-based tool planning."""

    def __init__(self, config: ToolTreeConfig) -> None:
        super().__init__(config)
        self.planner_llm = LLMClient(config.planner_llm)
        self.judge_llm = LLMClient(config.judge_llm)
        self.tool_registry = ToolRegistry()
        tools_dir = config.data_path.rstrip("/") + "/tools"
        try:
            self.tool_registry.load_from_directory(tools_dir)
        except Exception:
            pass
        executor = None
        if getattr(config, "tool_execution", "mock") == "real":
            import os

            from src.tools.real_tools import RealToolExecutor

            executor = RealToolExecutor(
                data_root=config.data_path,
                llm_client=self.planner_llm,
                device=os.environ.get("TOOLTREE_TOOL_DEVICE", "cuda:1"),
            )
        self.tool_manager = ToolManager(self.tool_registry, executor=executor)
        self.judge = LLMJudge(self.judge_llm)
        self.searcher = ToolTreeSearch(
            config.mcts,
            self.tool_manager,
            self.judge,
            self.planner_llm,
        )

    def solve(self, query: str, context: dict | None = None) -> dict:
        """Run ToolTree on a single query."""
        context = context or {}
        available_tools = self._get_available_tools(query, context)
        trajectory, best_q = self.searcher.search(query, context, available_tools)
        answer = self._predict_answer(query, trajectory)
        rollouts_used = getattr(self.searcher, "_rollouts_used", 0)
        return {
            "answer": answer,
            "trajectory": trajectory,
            "reward": best_q,
            "rollouts_used": rollouts_used,
        }

    def _get_available_tools(self, query: str, context: dict | None = None) -> list[dict]:
        """Return tool card dicts available for this query.

        Precedence: per-sample tools from context, then registry.
        """
        context = context or {}
        per_sample = context.get("available_tools") or []
        if per_sample:
            return [self._normalize_tool_dict(t) for t in per_sample]
        if self.config.benchmark in ("gta", "mm"):
            tools = self.tool_registry.get_all_tools()
        else:
            tools = self.tool_registry.retrieve_tools(query, k=self.config.tool_retrieval_k)
        return [self._tool_card_to_dict(t) for t in tools]

    @staticmethod
    def _normalize_tool_dict(raw: dict) -> dict:
        """Map external tool dicts (GTA-style) to our tool-card shape."""
        if "input_schema" in raw:
            return raw
        input_schema = {
            "type": "object",
            "properties": {
                (inp.get("name") or f"arg_{i}"): {
                    "type": inp.get("type", "string"),
                    "description": inp.get("description") or "",
                }
                for i, inp in enumerate(raw.get("inputs") or [])
            },
            "required": [inp.get("name") for inp in (raw.get("inputs") or []) if not inp.get("optional") and inp.get("name")],
        }
        output_schema = {
            "type": "object",
            "properties": {
                (out.get("name") or "output"): {
                    "type": out.get("type", "string"),
                    "description": out.get("description") or "",
                }
                for out in (raw.get("outputs") or [])
            },
        }
        return {
            "name": raw.get("name"),
            "description": raw.get("description") or "",
            "input_schema": input_schema,
            "output_schema": output_schema,
            "examples": raw.get("examples") or [],
            "domain": raw.get("domain") or "",
        }

    @staticmethod
    def _tool_card_to_dict(tool_card: ToolCard) -> dict:
        return asdict(tool_card)

    def _predict_answer(self, query: str, trajectory: list[dict]) -> str:
        """Use the answer predictor LLM to generate the final answer."""
        user_msg = build_answer_message(query, trajectory)
        messages = [
            {"role": "system", "content": ANSWER_PREDICTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            return self.planner_llm.generate(messages)
        except Exception as exc:
            return f"<answer_prediction_failed: {exc}>"

    def reset(self) -> None:
        """Clear caches and reset state between tasks."""
        self.tool_manager.clear_cache()
