"""Main MCTS algorithm for ToolTree.

Implements the full ToolTree search procedure (Section 3):
Selection -> Pre-Evaluation -> Expansion -> Execution ->
Post-Evaluation -> Backward Propagation, with early stopping.
"""

from __future__ import annotations

import copy
import json
import re

from src.config import MCTSConfig
from src.llm.client import LLMClient
from src.mcts.node import MCTSNode
from src.mcts.pruning import BidirectionalPruner
from src.evaluation.judge import LLMJudge
from src.tools.tool_manager import ToolManager


class ToolTreeSearch:
    """Monte Carlo Tree Search for tool planning with dual evaluation."""

    def __init__(
        self,
        config: MCTSConfig,
        tool_manager: ToolManager,
        judge: LLMJudge,
        planner_llm: LLMClient,
    ) -> None:
        self.config = config
        self.tool_manager = tool_manager
        self.judge = judge
        self.planner_llm = planner_llm
        self.pruner = BidirectionalPruner(config.tau_pre, config.tau_post)
        self._best_q_history: list[float] = []

    def search(
        self,
        query: str,
        context: dict,
        available_tools: list[dict],
    ) -> tuple[list[dict], float]:
        """Run the full MCTS search and return (best_trajectory, best_q_value)."""
        self._query = query
        self._initial_context = context or {}
        self._available_tools = available_tools or []
        self._best_q_history = []

        root = MCTSNode(state={"context": copy.deepcopy(self._initial_context), "depth": 0})
        rollouts_used = 0

        for _ in range(self.config.max_rollouts):
            rollouts_used += 1
            leaf = self._select(root)

            if leaf.state.get("depth", 0) >= self.config.max_depth:
                leaf.is_terminal = True

            if not leaf.is_terminal:
                candidates = self._get_candidate_actions(leaf.state, self._available_tools)
                new_children = self._expand(leaf, candidates)
                target = new_children[0] if new_children else leaf
            else:
                target = leaf

            if target is not root and target.action is not None:
                output = self._execute(target)
                target.tool_output = output
                r_post, _ = self.judge.post_evaluate(
                    query,
                    leaf.state.get("context", {}),
                    self._tool_card_by_name(target.action.get("tool_name", "")),
                    target.action_args or target.action.get("tool_args") or {},
                    output,
                )
                self.pruner.post_prune(target, r_post)
                self._backpropagate(target, r_post)
            else:
                self._backpropagate(target, 0.0)

            best_q_now = max((c.q_value for c in root.children), default=0.0)
            self._best_q_history.append(best_q_now)

            if self._check_early_stop():
                break

        self._rollouts_used = rollouts_used
        return self._get_best_trajectory(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse from root via best_child(lambda) until a leaf or unexpandable node."""
        current = node
        while not current.is_leaf():
            expandable_children = [c for c in current.children if c.is_expandable]
            if not expandable_children:
                return current
            try:
                current = current.best_child(self.config.exploration_constant)
            except ValueError:
                return current
            if current.is_terminal or not current.is_expandable:
                return current
        return current

    def _get_candidate_actions(
        self,
        state: dict,
        available_tools: list[dict],
    ) -> list[dict]:
        """Use planner LLM to propose up to top_k candidate tool actions."""
        if not available_tools:
            return []
        tool_summaries = []
        for t in available_tools:
            name = t.get("name", "")
            desc = t.get("description", "")
            schema = t.get("input_schema", {})
            tool_summaries.append(
                {"name": name, "description": desc, "input_schema": schema}
            )
        try:
            tools_str = json.dumps(tool_summaries, indent=2, default=str)
            context_str = json.dumps(state.get("context", {}), indent=2, default=str)
            trajectory_str = json.dumps(state.get("trajectory", []), indent=2, default=str)
        except Exception:
            tools_str = str(tool_summaries)
            context_str = str(state.get("context", {}))
            trajectory_str = str(state.get("trajectory", []))

        system = (
            "You are a tool-planning agent. Given the user query, current "
            "context, and a list of available tools, propose up to K candidate "
            "next tool calls. Each candidate must be a JSON object with keys: "
            '"tool_name" (string matching a provided tool), "tool_args" (object), '
            'and "arg_draft" (object, usually equal to tool_args). '
            'Respond with a JSON object {"candidates": [...]}'
        )
        user = (
            f"User query:\n{self._query}\n\n"
            f"Current context:\n{context_str}\n\n"
            f"Prior trajectory:\n{trajectory_str}\n\n"
            f"Available tools:\n{tools_str}\n\n"
            f"Propose up to K={self.config.top_k} candidates."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        candidates: list[dict] = []
        try:
            payload = self.planner_llm.generate_json(messages)
            if isinstance(payload, dict):
                raw = payload.get("candidates") or payload.get("actions") or []
            elif isinstance(payload, list):
                raw = payload
            else:
                raw = []
        except Exception:
            try:
                text = self.planner_llm.generate(messages)
                raw = self._extract_candidates_from_text(text)
            except Exception:
                raw = []

        tool_names = {t.get("name") for t in available_tools}
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            name = item.get("tool_name") or item.get("name")
            if not name or name not in tool_names:
                continue
            tool_args = item.get("tool_args") or item.get("args") or {}
            arg_draft = item.get("arg_draft") or tool_args
            if not isinstance(tool_args, dict):
                tool_args = {}
            if not isinstance(arg_draft, dict):
                arg_draft = {}
            candidates.append(
                {"tool_name": name, "tool_args": tool_args, "arg_draft": arg_draft}
            )
            if len(candidates) >= self.config.top_k:
                break
        return candidates

    @staticmethod
    def _extract_candidates_from_text(text: str) -> list[dict]:
        if not text:
            return []
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                payload = json.loads(m.group(0))
                if isinstance(payload, list):
                    return payload
            except Exception:
                pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                payload = json.loads(m.group(0))
                if isinstance(payload, dict):
                    return payload.get("candidates") or payload.get("actions") or []
            except Exception:
                pass
        return []

    def _expand(
        self,
        node: MCTSNode,
        candidates: list[dict],
    ) -> list[MCTSNode]:
        """Create children for candidates with r_pre >= tau_pre."""
        new_children: list[MCTSNode] = []
        parent_context = node.state.get("context", {})
        for action in candidates:
            tool_card = self._tool_card_by_name(action.get("tool_name", ""))
            r_pre, _ = self.judge.pre_evaluate(
                self._query,
                parent_context,
                tool_card,
                action.get("arg_draft") or action.get("tool_args") or {},
            )
            if self.pruner.should_pre_prune(r_pre):
                continue
            child_state = {
                "context": copy.deepcopy(parent_context),
                "depth": node.state.get("depth", 0) + 1,
                "trajectory": list(node.state.get("trajectory", []))
                + [
                    {
                        "tool": action.get("tool_name"),
                        "args": action.get("tool_args") or {},
                    }
                ],
            }
            child = MCTSNode(state=child_state, action=action, parent=node)
            child.r_pre = r_pre
            child.action_args = action.get("tool_args") or {}
            node.children.append(child)
            new_children.append(child)
        return new_children

    def _tool_card_by_name(self, name: str) -> dict:
        for t in self._available_tools:
            if t.get("name") == name:
                return t
        return {"name": name, "description": "", "input_schema": {}, "output_schema": {}}

    def _execute(self, node: MCTSNode) -> dict:
        """Invoke the tool via ToolManager and cache the result."""
        action = node.action or {}
        tool_name = action.get("tool_name", "")
        args = node.action_args or action.get("tool_args") or {}
        return self.tool_manager.execute(tool_name, args)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Walk from node to root updating Q-values."""
        current: MCTSNode | None = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def _check_early_stop(self) -> bool:
        """Return True if best Q has not improved by delta for patience rollouts."""
        patience = self.config.early_stop_patience
        delta = self.config.early_stop_delta
        if len(self._best_q_history) <= patience:
            return False
        window = self._best_q_history[-(patience + 1):]
        baseline = window[0]
        best_recent = max(window[1:])
        return (best_recent - baseline) < delta

    def _get_best_trajectory(
        self, root: MCTSNode
    ) -> tuple[list[dict], float]:
        """Greedy descent from root following max-Q child at each level."""
        trajectory: list[dict] = []
        current = root
        best_q = 0.0
        while current.children:
            candidates = current.children
            best_child = max(candidates, key=lambda c: c.q_value)
            action = best_child.action or {}
            trajectory.append(
                {
                    "action": action.get("tool_name"),
                    "args": best_child.action_args or action.get("tool_args") or {},
                    "output": best_child.tool_output,
                }
            )
            best_q = best_child.q_value
            if best_child.is_terminal or not best_child.children:
                current = best_child
                break
            current = best_child
        return trajectory, best_q
