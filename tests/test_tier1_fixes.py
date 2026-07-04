"""Behavior tests pinning the Tier-1 faithfulness fixes (audit B.2/B.3/B.4/B.6).

Each test encodes what the PAPER describes (per the third-party audit's
validator report), so they fail on the released c7d30c4 behavior and pass
once the fixes land. Pure stubs — no LLM server needed.
"""

from __future__ import annotations

import copy
import math
import random

import pytest

from src.config import MCTSConfig
from src.mcts.node import MCTSNode
from src.mcts.tree_search import ToolTreeSearch
from src.tools.tool_manager import ToolManager
from src.tools.tool_registry import ToolRegistry

MOCK_MARKER = "<mock: step-by-step mode>"


def make_tool(name, desc="", extra_props=None, required=None):
    props = {"query": {"type": "string"}}
    if extra_props:
        props = extra_props
    return {
        "name": name,
        "description": desc or f"The {name} tool.",
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": required if required is not None else list(props),
        },
        "output_schema": {"type": "object"},
    }


class StubJudge:
    def __init__(self, pre_scores, post_scores):
        self.pre_scores = pre_scores
        self.post_scores = post_scores
        self.pre_calls = []
        self.post_calls = []

    def pre_evaluate(self, query, context, tool_card, arg_draft):
        name = tool_card.get("name")
        self.pre_calls.append(
            {"tool": name, "context": copy.deepcopy(context)}
        )
        return self.pre_scores.get(name, 0.5), "stub"

    def post_evaluate(self, query, context_before, tool_card, args_used, tool_output):
        name = tool_card.get("name")
        self.post_calls.append(
            {"tool": name, "context": copy.deepcopy(context_before)}
        )
        return self.post_scores.get(name, 0.5), "stub"


class StubPlanner:
    """Only role after the B.4 fix: argument drafts (and the answer predictor)."""

    def __init__(self):
        self.json_calls = []

    def generate_json(self, messages):
        self.json_calls.append(messages)
        return {"query": "draft"}

    def generate(self, messages, **kwargs):
        return "{}"


def make_search(tools, pre_scores, post_scores, **cfg_overrides):
    cfg = MCTSConfig(
        exploration_constant=1.4,
        max_rollouts=cfg_overrides.pop("max_rollouts", 10),
        tau_pre=0.3,
        tau_post=0.4,
        early_stop_delta=1e-3,
        early_stop_patience=10,
        max_depth=cfg_overrides.pop("max_depth", 3),
        top_k=cfg_overrides.pop("top_k", 5),
    )
    judge = StubJudge(pre_scores, post_scores)
    planner = StubPlanner()
    search = ToolTreeSearch(cfg, ToolManager(ToolRegistry()), judge, planner)
    return search, judge, planner


# ---------------------------------------------------------------- B.3 / Eq. 1

def test_eq1_forced_first_visit():
    """Eq. 1: an unvisited action has unbounded bonus => tried before any
    visited sibling, regardless of Q or r_pre."""
    random.seed(0)
    root = MCTSNode(state={"context": {}, "depth": 0})
    visited = MCTSNode(state={"context": {}, "depth": 1},
                       action={"tool_name": "a"}, parent=root)
    visited.q_value, visited.visit_count, visited.r_pre = 0.95, 5, 0.9
    fresh = MCTSNode(state={"context": {}, "depth": 1},
                     action={"tool_name": "b"}, parent=root)
    fresh.visit_count, fresh.r_pre = 0, 0.1
    root.children = [visited, fresh]
    root.visit_count = 5
    assert root.best_child(1.4) is fresh


def test_eq1_formula_for_visited_children():
    """UCT(s,a) = Q + c * r_pre * sqrt(ln N(s) / N(s,a)) for visited children."""
    random.seed(0)
    root = MCTSNode(state={"context": {}, "depth": 0})
    root.visit_count = 4
    a = MCTSNode(state={}, action={"tool_name": "a"}, parent=root)
    a.q_value, a.visit_count, a.r_pre = 0.50, 4, 0.8
    b = MCTSNode(state={}, action={"tool_name": "b"}, parent=root)
    b.q_value, b.visit_count, b.r_pre = 0.90, 2, 0.4
    root.children = [a, b]
    uct_a = 0.50 + 1.4 * 0.8 * math.sqrt(math.log(4) / 4)   # ~= 1.1594
    uct_b = 0.90 + 1.4 * 0.4 * math.sqrt(math.log(4) / 2)   # ~= 1.3664
    assert uct_b > uct_a
    assert root.best_child(1.4) is b


def test_eq1_tie_break_prefers_larger_n():
    """Paper tie-break: equal UCT -> larger N wins (before jitter)."""
    random.seed(0)
    root = MCTSNode(state={"context": {}, "depth": 0})
    root.visit_count = 1  # ln(1) = 0 -> explore term vanishes, UCT = Q
    a = MCTSNode(state={}, action={"tool_name": "a"}, parent=root)
    a.q_value, a.visit_count, a.r_pre = 0.5, 1, 0.9
    b = MCTSNode(state={}, action={"tool_name": "b"}, parent=root)
    b.q_value, b.visit_count, b.r_pre = 0.5, 3, 0.9
    root.children = [a, b]
    picks = {root.best_child(1.4) for _ in range(20)}
    assert picks == {b}


# ------------------------------------------------------------------- B.4

def test_expansion_enumerates_scores_all_sorts_and_caps():
    """Enumerate A_rem, pre-eval ALL admissible actions, keep r_pre >= tau_pre,
    order children by r_pre desc, cap at top_k, execute the best one."""
    tools = [make_tool(n) for n in ("t_mid", "t_best", "t_bad")]
    search, judge, planner = make_search(
        tools,
        pre_scores={"t_mid": 0.5, "t_best": 0.9, "t_bad": 0.1},
        post_scores={"t_mid": 0.8, "t_best": 0.8, "t_bad": 0.8},
        max_rollouts=1, top_k=2,
    )
    search.search("find the artemis mission news", {}, tools)
    # enumeration: every admissible tool scored exactly once at the root
    assert sorted(c["tool"] for c in judge.pre_calls) == ["t_bad", "t_best", "t_mid"]
    # argument drafts must be grounded in the user query
    assert all("artemis" in str(msgs) for msgs in planner.json_calls)
    root = search._last_root
    kept = [c.action["tool_name"] for c in root.children]
    assert kept == ["t_best", "t_mid"]          # tau_pre pruned t_bad; sorted desc
    assert judge.post_calls[0]["tool"] == "t_best"  # executed = argmax r_pre


def test_schema_gate_blocks_modality_incompatible_tools():
    """A tool whose required input modality (image) is absent from the context
    must not be enumerated (no pre-eval call, no child)."""
    text_tool = make_tool("t_text")
    img_tool = make_tool(
        "t_img", extra_props={"image": {"type": "image"}}, required=["image"]
    )
    tools = [text_tool, img_tool]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_text": 0.9, "t_img": 0.9},
        post_scores={"t_text": 0.8, "t_img": 0.8},
        max_rollouts=1,
    )
    search.search("q", {}, tools)  # context has no images
    assert [c["tool"] for c in judge.pre_calls] == ["t_text"]


def test_tool_reuse_with_different_args_is_admissible():
    """A_rem subtracts ACTIONS (tool, args), not tool names: re-invoking a
    tool with different arguments must stay admissible (GTA gold chains
    reuse OCR/Calculator with new args), while an identical (tool, args)
    repeat of an ancestor action is excluded."""

    class ContextAwarePlanner(StubPlanner):
        # draft depends on accumulated history => differs at each depth
        def generate_json(self, messages):
            self.json_calls.append(messages)
            depth_hint = str(messages[-1]["content"]).count(MOCK_MARKER)
            return {"query": f"draft-{depth_hint}"}

    tools = [make_tool("t_a")]  # a single tool; chains need reuse
    cfg = MCTSConfig(exploration_constant=1.4, max_rollouts=4, tau_pre=0.3,
                     tau_post=0.4, early_stop_delta=1e-3,
                     early_stop_patience=10, max_depth=3, top_k=5)
    judge = StubJudge({"t_a": 0.9}, {"t_a": 0.9})
    search = ToolTreeSearch(cfg, ToolManager(ToolRegistry()), judge,
                            ContextAwarePlanner())
    trajectory, _ = search.search("q", {}, tools)
    depths = []
    node = search._last_root
    while node.children:
        node = node.children[0]
        depths.append(node.action["tool_name"])
    assert depths.count("t_a") >= 2, (
        f"tool reuse with different args should build a >=2-deep chain, got {depths}"
    )


# ------------------------------------------------------------------- B.2

def test_executed_output_enters_context_of_descendants():
    """C_{t+1} = C_t (+) (a, o): once a node's action is executed, planning
    and judging below it must see the output in the context."""
    tools = [make_tool("t_a"), make_tool("t_b")]
    search, judge, planner = make_search(
        tools,
        pre_scores={"t_a": 0.9, "t_b": 0.6},
        post_scores={"t_a": 0.9, "t_b": 0.9},
        max_rollouts=6, max_depth=3,
    )
    search.search("q", {"user": "x"}, tools)
    # some pre-eval below an executed node must have seen its (a, o) record
    deep_calls = [c for c in judge.pre_calls if c["context"].get("history")]
    assert deep_calls, "no pre-eval ever saw an execution record in context"
    rec = deep_calls[0]["context"]["history"][0]
    assert rec["tool"] in ("t_a", "t_b")
    assert MOCK_MARKER in str(rec["output"])
    # arg-draft prompts (planner) below an executed node see it too
    assert any(MOCK_MARKER in str(m) for m in planner.json_calls), (
        "planner (arg drafts) never saw any execution output"
    )


def test_post_eval_context_is_context_before_the_call():
    """r_post = J(C_t, a, o): judge context must NOT already contain the
    evaluated call's own output (it is C_t, the context before)."""
    tools = [make_tool("t_a"), make_tool("t_b")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_a": 0.9, "t_b": 0.6},
        post_scores={"t_a": 0.9, "t_b": 0.9},
        max_rollouts=6, max_depth=3,
    )
    search.search("q", {}, tools)
    for call in judge.post_calls:
        history_tools = [rec["tool"] for rec in call["context"].get("history", [])]
        # ancestors never repeat a tool (A_rem), so the judged call's own
        # output being in context would mean C_{t+1} leaked into C_t
        assert call["tool"] not in history_tools
    # and the very first post-eval (root expansion) sees no history at all
    assert not judge.post_calls[0]["context"].get("history")


# ------------------------------------------------------------------- B.6

def test_each_edge_judged_at_most_once():
    """Post-evaluation happens once per executed edge: a judged terminal or
    post-pruned edge is marked exhausted and never re-selected, so surplus
    rollouts cannot trigger a second judgment of the same edge."""
    tools = [make_tool("t_a"), make_tool("t_b")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_a": 0.9, "t_b": 0.6},
        post_scores={"t_a": 0.9, "t_b": 0.9},
        max_rollouts=15, max_depth=2, top_k=5,
    )
    search.search("q", {}, tools)
    judged = [c["tool"] for c in judge.post_calls]
    # tree over 2 tools at max_depth=2 has at most 4 executable edges;
    # 15 rollouts must not produce more post-evals than distinct edges
    assert len(judged) <= 4, f"re-judging detected: {judged}"


def test_best_trajectory_skips_post_pruned_children():
    """Post-pruned (non-expandable) nodes receive no further budget and their
    execution failed — the final trajectory and the best-Q signal must not
    range over them (audit evaluator finding #5)."""
    tools = [make_tool("t_good"), make_tool("t_dead")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_good": 0.9, "t_dead": 0.6},
        post_scores={"t_good": 0.9, "t_dead": 0.0},  # t_dead always fails
        max_rollouts=8, max_depth=3,
    )
    trajectory, best_q = search.search("q", {}, tools)
    steps = [s["action"] for s in trajectory]
    assert "t_dead" not in steps, f"post-pruned step leaked into plan: {steps}"
    assert best_q > 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
