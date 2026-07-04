"""Main MCTS algorithm for ToolTree.

Implements the full ToolTree search procedure (Section 3):
Selection -> Pre-Evaluation -> Expansion -> Execution ->
Post-Evaluation -> Backward Propagation, with early stopping.
"""

from __future__ import annotations

import copy
import json
import logging

from src.config import MCTSConfig
from src.llm.client import LLMClient
from src.mcts.node import MCTSNode
from src.mcts.pruning import BidirectionalPruner
from src.evaluation.judge import LLMJudge
from src.tools.tool_manager import ToolManager


logger = logging.getLogger(__name__)


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
        self._last_root = root
        rollouts_used = 0

        # Loop structure: an iteration is either INFORMATIVE (a new edge is
        # executed and judged; consumes one rollout of R_max, feeds best_q /
        # early stopping) or a DISCOVERY PASS (selection only revealed that a
        # node is exhausted; consumes no budget, backpropagates nothing).
        # Termination: informative iterations are bounded by max_rollouts;
        # every discovery pass monotonically flips at least one node's
        # `expanded` or `is_exhausted` flag (never unset) over a node set
        # that only grows in informative iterations, so discovery passes are
        # bounded too and the loop cannot livelock.
        while rollouts_used < self.config.max_rollouts:
            if root.is_exhausted:
                # Search space exhausted: nothing new can be executed.
                logger.info(
                    "AUDIT search_exhausted rollouts_used=%d", rollouts_used
                )
                break

            leaf = self._select(root)
            logger.info(
                "AUDIT selected_leaf=%r rollouts_used=%d", leaf, rollouts_used
            )

            if leaf.state.get("depth", 0) >= self.config.max_depth:
                leaf.is_terminal = True

            if not leaf.is_root() and leaf.action is not None and leaf.tool_output is None:
                # Realize this state before planning from it: its own action
                # has never been executed, so its context is still C_t of the
                # parent. Execute and judge it this rollout; expansion happens
                # on a later visit, once C_{t+1} is complete.
                target = leaf
                logger.info(
                    "AUDIT realize_leaf tool=%s", leaf.action.get("tool_name")
                )
            elif not leaf.is_terminal and not leaf.expanded:
                candidates = self._get_candidate_actions(leaf)
                new_children = self._expand(leaf, candidates)
                leaf.expanded = True
                if not new_children:
                    # Expansion is one-shot; nothing survived gating /
                    # pruning / dedup, so this node is exhausted.
                    self._mark_exhausted(leaf)
                    continue
                target = new_children[0]
            else:
                # Terminal, or already expanded with no selectable children:
                # no new information can come from this node. (Normally
                # pre-empted by eager exhaustion propagation; kept as an
                # invariant-preserving fallback.)
                self._mark_exhausted(leaf)
                continue

            rollouts_used += 1
            output = self._execute(target)
            target.tool_output = output
            # Context update (paper Sec. 3, Execution):
            # C_{t+1} = C_t (+) (a, o_{t+1}) in structured form.
            self._record_execution(target, output)
            context_before = (
                target.parent.state.get("context", {}) if target.parent else {}
            )
            r_post, _ = self.judge.post_evaluate(
                query,
                context_before,
                self._tool_card_by_name(target.action.get("tool_name", "")),
                target.action_args or target.action.get("tool_args") or {},
                output,
            )
            self.pruner.post_prune(target, r_post)
            logger.info(
                "AUDIT post_eval tool=%s r_post=%.3f tau_post=%.2f post_pruned=%s",
                target.action.get("tool_name"),
                r_post,
                self.pruner.tau_post,
                not target.is_expandable,
            )
            self._backpropagate(target, r_post)

            if target.state.get("depth", 0) >= self.config.max_depth:
                target.is_terminal = True
            if target.is_terminal:
                # A terminal node can never be expanded: judged once, done.
                self._mark_exhausted(target)
            elif not target.is_expandable:
                # Post-pruned: dead for selection purposes; the parent may
                # now be exhausted.
                self._propagate_exhaustion(target.parent)

            best_q_now = max(
                (c.q_value for c in root.children if c.is_expandable),
                default=0.0,
            )
            self._best_q_history.append(best_q_now)
            logger.info("AUDIT rollout=%d best_q=%.6f", rollouts_used, best_q_now)

            if self._check_early_stop():
                logger.info("AUDIT early_stop triggered at rollout=%d", rollouts_used)
                break

        self._rollouts_used = rollouts_used
        return self._get_best_trajectory(root)

    def _mark_exhausted(self, node: MCTSNode) -> None:
        """Mark `node` exhausted and propagate exhaustion toward the root."""
        if not node.is_exhausted:
            node.is_exhausted = True
            logger.info(
                "AUDIT node_exhausted node=%s depth=%d",
                (node.action or {}).get("tool_name", "root"),
                node.state.get("depth", 0),
            )
        self._propagate_exhaustion(node.parent)

    def _propagate_exhaustion(self, node: MCTSNode | None) -> None:
        """Walk toward the root marking nodes whose subtrees are spent.

        A node is exhausted once its one-shot expansion has happened and
        every child is dead (post-pruned or exhausted). Unexpanded nodes are
        never marked here: they can still yield new children.
        """
        current = node
        while current is not None and not current.is_exhausted:
            if not current.expanded:
                break
            if current.selectable_children():
                break
            current.is_exhausted = True
            logger.info(
                "AUDIT node_exhausted node=%s depth=%d",
                (current.action or {}).get("tool_name", "root"),
                current.state.get("depth", 0),
            )
            current = current.parent

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse from root via best_child(lambda) over selectable children.

        Returns either a node with no selectable children (to be expanded, or
        recognized as exhausted) or an unexecuted child (to be realized).
        Exhausted and post-pruned children are never descended into.
        """
        current = node
        while True:
            if not current.selectable_children():
                return current
            current = current.best_child(self.config.exploration_constant)
            if current.tool_output is None and not current.is_root():
                # Unexecuted: realize (execute + judge) it this rollout.
                return current

    def _get_candidate_actions(self, node: MCTSNode) -> list[dict]:
        """Enumerate the remaining admissible actions A_rem(s_t) (paper Sec. 3).

        A_rem(s_t) = A(s_t) \\ {a_1..a_t}: every available tool is drafted
        against the current context, excluding only actions IDENTICAL to one
        already on the path — an action is (tool, args), not a tool name, so
        re-invoking a tool with different arguments (e.g. OCR on a second
        image) stays admissible, as benchmark gold chains require. Tools are
        gated by input-schema compatibility first. No LLM proposes or filters
        the candidate set; scoring and the top-K shortlist happen in _expand,
        after r_pre is known.

        Candidates identical to an EXISTING CHILD of this node are excluded
        too, so an equivalent node can never be rebuilt (and re-judged) after
        its twin was post-pruned. With one-shot expansion this dedup is a
        defense-in-depth invariant rather than a hot path.
        """
        state = node.state
        available_tools = self._available_tools
        if not available_tools:
            return []
        context = state.get("context", {})
        prior_actions = {
            self._action_fingerprint(step.get("tool"), step.get("args") or {})
            for step in (state.get("trajectory") or [])
        }
        sibling_actions = {
            self._action_fingerprint(
                (c.action or {}).get("tool_name"),
                c.action_args or (c.action or {}).get("tool_args") or {},
            )
            for c in node.children
        }
        candidates: list[dict] = []
        for tool in available_tools:
            name = tool.get("name")
            if not name:
                continue
            if not self.tool_manager.tool_registry.check_schema_compatibility(
                tool, context
            ):
                logger.info("AUDIT schema_gate tool=%s excluded", name)
                continue
            draft = self.tool_manager.generate_arg_draft(
                tool, context, self.planner_llm, query=self._query
            )
            if not isinstance(draft, dict):
                draft = {}
            fingerprint = self._action_fingerprint(name, draft)
            if fingerprint in prior_actions:
                logger.info("AUDIT dedup_prior_action tool=%s", name)
                continue
            if fingerprint in sibling_actions:
                logger.info("AUDIT dedup_existing_child tool=%s", name)
                continue
            candidates.append(
                {"tool_name": name, "tool_args": draft, "arg_draft": draft}
            )
        return candidates

    @staticmethod
    def _action_fingerprint(tool_name, args) -> str:
        try:
            args_repr = json.dumps(args or {}, sort_keys=True, default=str)
        except Exception:
            args_repr = str(args)
        return f"{tool_name}::{args_repr}"

    def _expand(
        self,
        node: MCTSNode,
        candidates: list[dict],
    ) -> list[MCTSNode]:
        """Pre-evaluate ALL candidates, keep A+ = {a : r_pre >= tau_pre},
        shortlist A_keep = top-K(A+; r_pre), and instantiate children in
        descending r_pre order (paper Sec. 3, Pre-Evaluation / Expansion).

        new_children[0] is therefore the highest-r_pre survivor — the action
        executed this rollout.
        """
        parent_context = node.state.get("context", {})
        survivors: list[tuple[float, dict]] = []
        for action in candidates:
            tool_card = self._tool_card_by_name(action.get("tool_name", ""))
            r_pre, _ = self.judge.pre_evaluate(
                self._query,
                parent_context,
                tool_card,
                action.get("arg_draft") or action.get("tool_args") or {},
            )
            if self.pruner.should_pre_prune(r_pre):
                logger.info(
                    "AUDIT pre_prune tool=%s r_pre=%.3f < tau_pre=%.2f",
                    action.get("tool_name"),
                    r_pre,
                    self.pruner.tau_pre,
                )
                continue
            survivors.append((r_pre, action))

        # top-K shortlist BY r_pre (the released code capped in arrival
        # order before scoring — audit B.4 / A.8)
        survivors.sort(key=lambda pair: pair[0], reverse=True)
        kept = survivors[: self.config.top_k]
        for r_pre, action in survivors[self.config.top_k:]:
            logger.info(
                "AUDIT topk_drop tool=%s r_pre=%.3f (K=%d)",
                action.get("tool_name"),
                r_pre,
                self.config.top_k,
            )

        new_children: list[MCTSNode] = []
        for r_pre, action in kept:
            logger.info(
                "AUDIT expand_keep tool=%s r_pre=%.3f >= tau_pre=%.2f",
                action.get("tool_name"),
                r_pre,
                self.pruner.tau_pre,
            )
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

    def _record_execution(self, node: MCTSNode, output: dict) -> None:
        """Append (a, o_{t+1}) to the node's context: C_{t+1} = C_t (+) (a, o).

        Descendants deep-copy this context at expansion, so the planner's
        argument drafts and both judges see all accumulated results on the
        path (paper Sec. 2 state definition; Sec. 3 Execution step).
        """
        action = node.action or {}
        context = node.state.setdefault("context", {})
        history = context.setdefault("history", [])
        history.append(
            {
                "tool": action.get("tool_name"),
                "args": node.action_args or action.get("tool_args") or {},
                "output": output,
            }
        )

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Walk from node to root updating Q-values."""
        current: MCTSNode | None = node
        while current is not None:
            current.update(reward)
            logger.info(
                "AUDIT backprop node=%s reward=%.3f -> Q=%.6f N=%d",
                (current.action or {}).get("tool_name", "root"),
                reward,
                current.q_value,
                current.visit_count,
            )
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
        stop = (best_recent - baseline) < delta
        if stop:
            logger.info(
                "AUDIT early_stop_check baseline=%.6f best_recent=%.6f "
                "improvement=%.6f < delta=%.6f (patience=%d)",
                baseline,
                best_recent,
                best_recent - baseline,
                delta,
                patience,
            )
        return stop

    def _get_best_trajectory(
        self, root: MCTSNode
    ) -> tuple[list[dict], float]:
        """Greedy descent from root following max-Q child at each level.

        Post-pruned (non-expandable) children are excluded: their execution
        scored below tau_post, so they belong neither in the final plan nor
        in the value signal (audit evaluator finding #5).
        """
        trajectory: list[dict] = []
        current = root
        best_q = 0.0
        while current.children:
            candidates = [c for c in current.children if c.is_expandable]
            if not candidates:
                break
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
