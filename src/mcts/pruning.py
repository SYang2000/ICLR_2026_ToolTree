"""Bidirectional pruning for ToolTree."""

from __future__ import annotations

from src.mcts.node import MCTSNode


class BidirectionalPruner:
    def __init__(self, tau_pre: float = 0.3, tau_post: float = 0.4) -> None:
        self.tau_pre = tau_pre
        self.tau_post = tau_post

    def pre_prune(
        self,
        candidates: list[dict],
        pre_scores: list[float],
    ) -> list[tuple[dict, float]]:
        return [
            (c, s) for c, s in zip(candidates, pre_scores) if s >= self.tau_pre
        ]

    def post_prune(self, node: MCTSNode, r_post: float) -> None:
        node.r_post = r_post
        if r_post < self.tau_post:
            node.is_expandable = False

    def should_pre_prune(self, r_pre: float) -> bool:
        return r_pre < self.tau_pre

    def should_post_prune(self, r_post: float) -> bool:
        return r_post < self.tau_post
