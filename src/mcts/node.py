"""MCTS Node definition for ToolTree.

Each node represents a state in the tool planning search tree.
Edges correspond to tool actions. The node stores Q-values, visit
counts, and pre/post evaluation scores as described in Section 3.
"""

from __future__ import annotations


class MCTSNode:
    """A node in the MCTS search tree representing a state in tool planning.

    Attributes:
        state: Current dialogue context and accumulated intermediate results.
        action: The tool action that led to this node (None for root).
        parent: Parent node reference.
        children: List of child nodes.
        visit_count: N(s, a) -- number of times this node has been visited.
        q_value: Q(s, a) -- running mean of post-evaluation rewards.
        r_pre: Pre-evaluation score for this action (prior signal).
        r_post: Post-evaluation score after execution (None if not yet executed).
        is_expandable: Whether this node can be further expanded (not post-pruned).
        is_terminal: Whether this node represents a completed trajectory.
        tool_output: Cached output from tool execution (None if not yet executed).
        action_args: Cached argument draft for the tool call.
    """

    def __init__(
        self,
        state: dict,
        action: dict | None = None,
        parent: "MCTSNode | None" = None,
    ) -> None:
        """Initialize a new MCTS node.

        Args:
            state: The dialogue state at this node (context + intermediate results).
            action: The tool action that led to this node (None for root).
            parent: The parent node (None for root).
        """
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
        """Check if this node is a leaf (has no children).

        Returns:
            True if the node has no children.
        """
        raise NotImplementedError

    def is_root(self) -> bool:
        """Check if this node is the root (has no parent).

        Returns:
            True if this node has no parent.
        """
        raise NotImplementedError

    def is_fully_expanded(self, admissible_actions: list[dict]) -> bool:
        """Check if all admissible actions have been tried from this node.

        Args:
            admissible_actions: List of all valid actions from this state.

        Returns:
            True if every admissible action has a corresponding child node.
        """
        raise NotImplementedError

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        """Select the best child using prior-augmented UCT (Eq. 1 in paper).

        UCT(s, a) = Q(s, a) + lambda * r_pre(s, a) * sqrt(ln N(s)) / (1 + N(s, a))

        Only considers children where is_expandable=True.
        Ties are broken by larger N(s), then small random jitter for diversity.

        Args:
            exploration_constant: lambda -- balances exploration vs exploitation.

        Returns:
            The child node with the highest UCT value.

        Raises:
            ValueError: If no expandable children exist.
        """
        raise NotImplementedError

    def update(self, reward: float) -> None:
        """Update visit count and Q-value using incremental mean.

        N(s, a) <- N(s, a) + 1
        Q(s, a) <- Q(s, a) + (reward - Q(s, a)) / N(s, a)

        Args:
            reward: The r_post score to incorporate (in [0, 1]).
        """
        raise NotImplementedError

    def get_trajectory(self) -> list[dict]:
        """Return the sequence of (action, output) pairs from root to this node.

        Returns:
            List of dicts, each containing "action", "args", and "output" keys.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """String representation showing action, Q-value, and visit count."""
        action_name = self.action.get("tool_name", "root") if self.action else "root"
        return (
            f"MCTSNode(action={action_name}, Q={self.q_value:.4f}, "
            f"N={self.visit_count}, r_pre={self.r_pre:.3f})"
        )
