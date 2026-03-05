"""Bidirectional pruning for ToolTree (Section 3.2).

Pre-pruning: discard actions where r_pre < tau_pre before expansion.
Post-pruning: mark nodes as non-expandable where r_post < tau_post after execution.

Together, these rules concentrate rollouts on branches that are both
likely (per r_pre) and useful (per r_post), improving accuracy-per-second
under fixed R_max.
"""

from __future__ import annotations

from src.mcts.node import MCTSNode


class BidirectionalPruner:
    """Implements pre-pruning and post-pruning for ToolTree.

    Pre-pruning filters candidate actions before tool execution based on
    the predictive r_pre score. Post-pruning marks executed nodes as
    non-expandable based on the grounded r_post score.
    """

    def __init__(self, tau_pre: float = 0.3, tau_post: float = 0.4) -> None:
        """Initialize pruner with threshold values.

        Args:
            tau_pre: Pre-pruning threshold; actions with r_pre < tau_pre are discarded.
            tau_post: Post-pruning threshold; nodes with r_post < tau_post are marked
                      non-expandable.
        """
        raise NotImplementedError

    def pre_prune(
        self,
        candidates: list[dict],
        pre_scores: list[float],
    ) -> list[tuple[dict, float]]:
        """Filter candidates by pre-evaluation threshold.

        Args:
            candidates: List of candidate action dicts.
            pre_scores: Corresponding r_pre scores from the LLM judge.

        Returns:
            List of (candidate, r_pre) tuples that pass the threshold (r_pre >= tau_pre).
        """
        raise NotImplementedError

    def post_prune(self, node: MCTSNode, r_post: float) -> None:
        """Mark node as non-expandable if r_post < tau_post.

        When post-pruned, the node's is_expandable flag is set to False,
        preventing further budget allocation on this unproductive branch.

        Args:
            node: The node that was just executed and post-evaluated.
            r_post: The post-evaluation score for this node.
        """
        raise NotImplementedError

    def should_pre_prune(self, r_pre: float) -> bool:
        """Check if an action should be pre-pruned.

        Args:
            r_pre: Pre-evaluation score.

        Returns:
            True if r_pre < tau_pre (action should be discarded).
        """
        raise NotImplementedError

    def should_post_prune(self, r_post: float) -> bool:
        """Check if a node should be post-pruned.

        Args:
            r_post: Post-evaluation score.

        Returns:
            True if r_post < tau_post (node should be marked non-expandable).
        """
        raise NotImplementedError
