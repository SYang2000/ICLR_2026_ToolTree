"""MCTS Node for ToolTree search tree."""

from __future__ import annotations

import math
import random
from typing import Optional


class MCTSNode:
    def __init__(
        self,
        state: dict,
        action: dict | None = None,
        parent: "MCTSNode | None" = None,
    ) -> None:
        self.state = state
        self.action = action
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visit_count: int = 0
        self.q_value: float = 0.0
        self.r_pre: float = 0.0
        self.r_post: float | None = None
        self.is_expandable: bool = True
        self.is_terminal: bool = False
        self.tool_output: dict | None = None
        self.action_args: dict | None = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def is_fully_expanded(self, admissible_actions: list[dict]) -> bool:
        tried = {self._action_key(c.action) for c in self.children if c.action is not None}
        return all(self._action_key(a) in tried for a in admissible_actions)

    @staticmethod
    def _action_key(action: dict | None) -> str:
        if action is None:
            return ""
        args = action.get("tool_args") or {}
        try:
            args_repr = sorted(args.items())
        except Exception:
            args_repr = str(args)
        return f"{action.get('tool_name', '')}::{args_repr}"

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        expandable = [c for c in self.children if c.is_expandable]
        if not expandable:
            raise ValueError("No expandable children.")
        parent_visits = max(self.visit_count, 1)
        log_parent = math.log(parent_visits + 1)

        def uct(child: "MCTSNode") -> float:
            exploit = child.q_value
            prior = child.r_pre if child.r_pre > 0 else 1e-6
            explore = (
                exploration_constant
                * prior
                * math.sqrt(log_parent)
                / (1 + child.visit_count)
            )
            jitter = random.random() * 1e-9
            return exploit + explore + jitter

        return max(expandable, key=uct)

    def update(self, reward: float) -> None:
        self.visit_count += 1
        self.q_value += (reward - self.q_value) / self.visit_count

    def get_trajectory(self) -> list[dict]:
        path = []
        node: Optional[MCTSNode] = self
        while node is not None and node.parent is not None:
            path.append(
                {
                    "action": (node.action or {}).get("tool_name"),
                    "args": node.action_args or (node.action or {}).get("tool_args"),
                    "output": node.tool_output,
                }
            )
            node = node.parent
        return list(reversed(path))

    def __repr__(self) -> str:
        action_name = self.action.get("tool_name", "root") if self.action else "root"
        return (
            f"MCTSNode(action={action_name}, Q={self.q_value:.4f}, "
            f"N={self.visit_count}, r_pre={self.r_pre:.3f})"
        )
