"""MCTS Node for ToolTree search tree."""

from __future__ import annotations

import logging
import math
import random
from typing import Optional


logger = logging.getLogger(__name__)


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
        # is_expandable=False means POST-PRUNED (r_post < tau_post): the edge
        # gets no further budget and is excluded from the final plan/best-Q.
        self.is_expandable: bool = True
        self.is_terminal: bool = False
        # Exhaustion is deliberately separate from post-pruning: an exhausted
        # node yielded good executions but its subtree can produce no NEW
        # information (one-shot expansion done, all children dead). It is
        # excluded from selection but remains eligible for the final plan.
        self.is_exhausted: bool = False
        # Expansion is one-shot per node: A_rem(s_t) is a function of the
        # state, which is frozen once the node is executed, so re-expanding
        # could only replay the same drafts (audit: equivalent-candidate
        # rebuild ring).
        self.expanded: bool = False
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

    def selectable_children(self) -> "list[MCTSNode]":
        """Children eligible for selection: not post-pruned, not exhausted."""
        return [c for c in self.children if c.is_expandable and not c.is_exhausted]

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        """Select a child by Eq. 1: UCT(s,a) = Q + lambda * r_pre * sqrt(ln N(s) / N(s,a)).

        An unvisited action (N(s,a) = 0) has an unbounded bonus, so unvisited
        actions are always tried before revisiting a sibling. Ties break by
        larger visit count, then a small random jitter. Post-pruned and
        exhausted children are not candidates.
        """
        expandable = self.selectable_children()
        if not expandable:
            raise ValueError("No selectable children.")
        parent_visits = max(self.visit_count, 1)
        log_parent = math.log(parent_visits)

        def uct(child: "MCTSNode") -> float:
            if child.visit_count == 0:
                return float("inf")
            prior = child.r_pre if child.r_pre > 0 else 1e-6
            explore = exploration_constant * prior * math.sqrt(
                log_parent / child.visit_count
            )
            return child.q_value + explore

        # AUDIT logging: compute uct once per child
        scored = [
            (uct(c), c.visit_count, random.random() * 1e-9, c) for c in expandable
        ]
        logger.info(
            "AUDIT uct c=%.2f parent_N=%d :: %s",
            exploration_constant,
            self.visit_count,
            " | ".join(
                f"{(c.action or {}).get('tool_name', 'root')}"
                f" Q={c.q_value:.4f} N={c.visit_count} r_pre={c.r_pre:.2f}"
                f" uct={u:.4f}"
                for u, _, _, c in scored
            ),
        )
        return max(scored, key=lambda t: (t[0], t[1], t[2]))[3]

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
