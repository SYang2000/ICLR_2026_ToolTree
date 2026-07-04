"""Behavior tests pinning the release fixes (post-exhaustion dynamics,
judge memoization, metric definitions, client hygiene).

Fix 1 — after the reachable tree is exhausted the search must STOP:
no cached-reward re-backprop (Q pumping), no rebuilding of equivalent
candidates after a twin was post-pruned, no budget burned on no-ops.
Pure stubs — no LLM server needed.
"""

from __future__ import annotations

import logging

import pytest

from test_tier1_fixes import StubJudge, StubPlanner, make_search, make_tool, MOCK_MARKER

from src.config import MCTSConfig
from src.mcts.tree_search import ToolTreeSearch
from src.tools.tool_manager import ToolManager
from src.tools.tool_registry import ToolRegistry


def _iter_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


# ------------------------------------------------------------------ Fix 1


def test_exhausted_search_stops_without_revisit_backprop(caplog):
    """Two tools, depth cap 3 => exactly 4 executable edges. A large budget
    must NOT be burned re-backpropagating cached r_post: one judged
    execution per rollout, then termination via search_exhausted."""
    caplog.set_level(logging.INFO)
    tools = [make_tool("t_a"), make_tool("t_b")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_a": 0.9, "t_b": 0.6},
        post_scores={"t_a": 0.9, "t_b": 0.9},
        max_rollouts=50, max_depth=3,
    )
    search.search("q", {}, tools)
    root = search._last_root
    # constant drafts => edges: a, b, a->b, b->a; deeper nodes dedup away
    assert len(judge.post_calls) == 4, judge.post_calls
    assert search._rollouts_used == 4  # budget of 50 not burned on no-ops
    assert root.visit_count == 4      # no N inflation from revisits
    assert root.is_exhausted
    assert "search_exhausted" in caplog.text


def test_best_q_history_only_records_informative_rollouts(caplog):
    """Uninformative (discovery) passes must not append to the best-Q
    history the early-stop criterion reads."""
    caplog.set_level(logging.INFO)
    tools = [make_tool("t_a"), make_tool("t_b")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_a": 0.9, "t_b": 0.6},
        post_scores={"t_a": 0.9, "t_b": 0.9},
        max_rollouts=50, max_depth=3,
    )
    search.search("q", {}, tools)
    # Guard the test's own premise: the scenario must actually contain
    # discovery passes, otherwise the equality below is vacuous. Exhaustion
    # (not budget) ends the search, nodes were flipped exhausted, and more
    # selection iterations ran than rollouts were consumed.
    assert search._last_root.is_exhausted
    assert "node_exhausted" in caplog.text
    assert caplog.text.count("selected_leaf") > search._rollouts_used
    assert len(search._best_q_history) == search._rollouts_used == len(judge.post_calls)


def test_no_equivalent_candidate_rebuild_after_post_prune():
    """A candidate identical to a post-pruned child must never be rebuilt
    (audit: one failing tool re-judged 11x in a single GTA sample)."""
    tools = [make_tool("t_good"), make_tool("t_dead")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_good": 0.9, "t_dead": 0.6},
        post_scores={"t_good": 0.9, "t_dead": 0.0},  # t_dead always post-pruned
        max_rollouts=30, max_depth=3,
    )
    search.search("q", {}, tools)
    dead_judgments = [c for c in judge.post_calls if c["tool"] == "t_dead"]
    # t_dead can be instantiated once under root and once under the good
    # chain (distinct states); each such edge is judged exactly once
    assert len(dead_judgments) <= 2, f"rebuild ring: {len(dead_judgments)} judgments"
    # no two children of any node share a (tool, args) fingerprint
    for node in _iter_nodes(search._last_root):
        fps = [
            ToolTreeSearch._action_fingerprint(
                (c.action or {}).get("tool_name"),
                c.action_args or (c.action or {}).get("tool_args") or {},
            )
            for c in node.children
        ]
        assert len(fps) == len(set(fps)), f"duplicate children: {fps}"


def test_exhausted_good_nodes_stay_in_best_trajectory():
    """Exhaustion (is_exhausted) must stay separate from post-pruning
    (is_expandable): a fully-explored good chain is exhausted for
    selection but must still form the final plan."""

    class ContextAwarePlanner(StubPlanner):
        def generate_json(self, messages):
            self.json_calls.append(messages)
            depth_hint = str(messages[-1]["content"]).count(MOCK_MARKER)
            return {"query": f"draft-{depth_hint}"}

    tools = [make_tool("t_a")]
    cfg = MCTSConfig(exploration_constant=1.4, max_rollouts=30, tau_pre=0.3,
                     tau_post=0.4, early_stop_delta=1e-3,
                     early_stop_patience=10, max_depth=3, top_k=5)
    judge = StubJudge({"t_a": 0.9}, {"t_a": 0.9})
    search = ToolTreeSearch(cfg, ToolManager(ToolRegistry()), judge,
                            ContextAwarePlanner())
    trajectory, best_q = search.search("q", {}, tools)
    root = search._last_root
    assert root.is_exhausted  # whole (chain-shaped) tree fully explored
    assert len(trajectory) == 3  # exhausted nodes still form the plan
    assert best_q > 0.0
    # every node on the plan is exhausted yet NOT post-pruned
    node = root
    while node.children:
        node = node.children[0]
        assert node.is_exhausted and node.is_expandable


def test_terminal_leaf_judged_once_then_exhausted():
    """A max-depth leaf is executed and judged exactly once; revisiting it
    must be impossible (it becomes exhausted, not a cached-reward pump)."""
    tools = [make_tool("t_a")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_a": 0.9},
        post_scores={"t_a": 0.9},
        max_rollouts=40, max_depth=1,  # root's child is instantly terminal
    )
    search.search("q", {}, tools)
    assert len(judge.post_calls) == 1
    assert search._rollouts_used == 1
    leaf = search._last_root.children[0]
    assert leaf.is_terminal and leaf.is_exhausted and leaf.is_expandable


def test_candidate_enumeration_dedups_against_existing_children():
    """Defense in depth under one-shot expansion: even if a node with
    children were ever re-enumerated, a candidate matching an existing
    child's (tool, args) fingerprint must be excluded."""
    from src.mcts.node import MCTSNode

    tools = [make_tool("t_a"), make_tool("t_b")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_a": 0.9, "t_b": 0.9},
        post_scores={"t_a": 0.0, "t_b": 0.9},
        max_rollouts=1, max_depth=3,
    )
    search._query = "q"
    search._available_tools = tools
    node = MCTSNode(state={"context": {}, "depth": 0})
    dead_twin = MCTSNode(
        state={"context": {}, "depth": 1},
        action={"tool_name": "t_a", "tool_args": {"query": "draft"}},
        parent=node,
    )
    dead_twin.action_args = {"query": "draft"}
    dead_twin.is_expandable = False  # post-pruned
    node.children = [dead_twin]
    candidates = search._get_candidate_actions(node)
    names = [c["tool_name"] for c in candidates]
    assert names == ["t_b"], f"post-pruned twin was rebuilt: {names}"


def test_all_candidates_pruned_at_root_exhausts_search():
    """If nothing survives pre-pruning at the root, the search must report
    exhaustion with zero executions instead of idling on zero-reward
    backprops."""
    tools = [make_tool("t_bad")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_bad": 0.1},   # < tau_pre = 0.3
        post_scores={"t_bad": 0.9},
        max_rollouts=40, max_depth=3,
    )
    trajectory, best_q = search.search("q", {}, tools)
    assert trajectory == [] and best_q == 0.0
    assert len(judge.post_calls) == 0
    assert search._rollouts_used == 0
    assert search._last_root.is_exhausted
    assert search._last_root.visit_count == 0  # no fabricated 0-reward backprop


# ------------------------------------------------------------------ Fix 2


class CountingLLM:
    """Stub LLM returning a fresh score per call — models the observed
    non-determinism of temperature-0 vLLM sampling."""

    def __init__(self):
        self.calls = 0

    def generate_json(self, messages):
        self.calls += 1
        return {"score": 0.5 + 0.01 * self.calls, "explanation": f"call-{self.calls}"}


class FailingLLM:
    def __init__(self):
        self.calls = 0

    def generate_json(self, messages):
        self.calls += 1
        raise ValueError("boom")


def _judge_inputs():
    tool = make_tool("t_a")
    return "the query", {"history": [{"tool": "t_b", "output": "o"}]}, tool


def test_judge_memoizes_identical_pre_evaluations():
    from src.evaluation.judge import LLMJudge

    llm = CountingLLM()
    judge = LLMJudge(llm)
    query, context, tool = _judge_inputs()
    first = judge.pre_evaluate(query, context, tool, {"query": "x"})
    second = judge.pre_evaluate(query, context, tool, {"query": "x"})
    assert llm.calls == 1
    assert first == second  # consistency even under a non-deterministic LLM
    # different content => a fresh judgment
    judge.pre_evaluate(query, context, tool, {"query": "y"})
    assert llm.calls == 2


def test_judge_memoizes_identical_post_evaluations():
    from src.evaluation.judge import LLMJudge

    llm = CountingLLM()
    judge = LLMJudge(llm)
    query, context, tool = _judge_inputs()
    args, output = {"query": "x"}, {"output": "result"}
    first = judge.post_evaluate(query, context, tool, args, output)
    second = judge.post_evaluate(query, context, tool, args, output)
    assert llm.calls == 1
    assert first == second
    # a different tool OUTPUT is a different judgment
    judge.post_evaluate(query, context, tool, args, {"output": "other"})
    assert llm.calls == 2


def test_judge_pre_and_post_memo_never_collide():
    from src.evaluation.judge import LLMJudge

    llm = CountingLLM()
    judge = LLMJudge(llm)
    query, context, tool = _judge_inputs()
    # same (query, context, tool, args); post has no extra output arg content
    judge.pre_evaluate(query, context, tool, {"query": "x"})
    judge.post_evaluate(query, context, tool, {"query": "x"}, {})
    assert llm.calls == 2


def test_judge_does_not_memoize_failures():
    from src.evaluation.judge import LLMJudge

    llm = FailingLLM()
    judge = LLMJudge(llm)
    query, context, tool = _judge_inputs()
    score, note = judge.pre_evaluate(query, context, tool, {})
    assert score == 0.0 and "pre_eval_parse_error" in note
    judge.pre_evaluate(query, context, tool, {})
    assert llm.calls == 2  # failure was retried, not served from memo


class ShapeErrorLLM:
    """Stub LLM that returns a malformed payload once, then a valid one —
    models a transient bad completion from the judge model."""

    def __init__(self, bad_payload):
        self.bad_payload = bad_payload
        self.calls = 0

    def generate_json(self, messages):
        self.calls += 1
        if self.calls == 1:
            return self.bad_payload
        return {"score": 0.7, "explanation": "recovered"}


@pytest.mark.parametrize(
    "bad_payload",
    [
        ["not", "a", "dict"],            # non-dict payload
        "0.9",                            # non-dict payload (bare string)
        {"explanation": "no score key"},  # missing 'score'
        {"score": "high"},                # non-numeric 'score'
    ],
)
def test_judge_does_not_memoize_shape_errors(bad_payload):
    """A completion with the wrong shape must score 0.0 WITHOUT being
    memoized: the next identical judgment retries the LLM and can recover.
    (Evaluator finding: silent 0.0 from _extract_score used to slip past
    the except branch and pin the (C,a,o) to 0.0 for the whole process.)"""
    from src.evaluation.judge import LLMJudge

    llm = ShapeErrorLLM(bad_payload)
    judge = LLMJudge(llm)
    query, context, tool = _judge_inputs()
    score, note = judge.pre_evaluate(query, context, tool, {"query": "x"})
    assert score == 0.0 and "pre_eval_parse_error" in note
    # identical input again: must hit the LLM, not a poisoned memo entry
    score2, note2 = judge.pre_evaluate(query, context, tool, {"query": "x"})
    assert llm.calls == 2
    assert score2 == pytest.approx(0.7) and note2 == "recovered"
    # the recovered (valid) judgment IS memoized
    judge.pre_evaluate(query, context, tool, {"query": "x"})
    assert llm.calls == 2


# ------------------------------------------------------------------ Fix 3


def test_exec_f1_omitted_when_no_gold_execution():
    """A run without execution references must not report exec_f1 at all —
    and in particular an empty (failed) plan must not score exec_f1 = 1.0
    against a missing gold (this inflated tier-2 GTA avg_f1)."""
    from src.evaluation.metrics import compute_all_metrics

    gold_plan = [{"tool": "OCR", "args": {"image": "a.jpg"}}]
    predictions = [
        {"plan": [], "execution": []},                      # failed empty plan
        {"plan": gold_plan, "execution": [{"tool": "OCR", "output": "x"}]},
    ]
    references = [
        {"gold_plan": gold_plan, "gold_execution": []},
        {"gold_plan": gold_plan, "gold_execution": []},
    ]
    metrics = compute_all_metrics(predictions, references, "gta")
    assert metrics["exec_f1"] is None
    # avg over the three available metrics only
    expected_avg = (
        metrics["tool_f1"] + metrics["arg_f1"] + metrics["plan_f1"]
    ) / 3
    assert metrics["avg_f1"] == pytest.approx(expected_avg)


def test_exec_f1_scores_only_gradeable_samples():
    """Samples without gold_execution are excluded from the exec_f1 average
    instead of contributing spurious 0/1 scores."""
    from src.evaluation.metrics import compute_all_metrics

    plan = [{"tool": "OCR", "args": {}}]
    execution = [{"tool": "OCR", "output": "text"}]
    predictions = [
        {"plan": [], "execution": []},          # not gradeable, empty plan
        {"plan": plan, "execution": execution}, # gradeable, exact match
    ]
    references = [
        {"gold_plan": plan, "gold_execution": []},
        {"gold_plan": plan, "gold_execution": execution},
    ]
    metrics = compute_all_metrics(predictions, references, "gta")
    assert metrics["exec_f1"] == pytest.approx(1.0)  # sample 0 skipped, not 1.0'd


def test_arg_f1_counts_repeated_tool_calls_per_step():
    """arg_f1 must flatten per step: both calls of a reused tool count.
    The released {tool: args} folding scored this reordered-duplicate case
    0.0; per-step multiset flattening scores it 1.0."""
    from src.evaluation.metrics import argument_f1

    pred = [
        {"tool": "OCR", "args": {"image": "a.jpg"}},
        {"tool": "OCR", "args": {"image": "b.jpg"}},
    ]
    gold = [
        {"tool": "OCR", "args": {"image": "b.jpg"}},
        {"tool": "OCR", "args": {"image": "a.jpg"}},
    ]
    assert argument_f1(pred, gold) == pytest.approx(1.0)


def test_arg_f1_multiset_multiplicity():
    """Multiset semantics: predicting one of two identical gold calls is
    partial credit (2/3), not a full match."""
    from src.evaluation.metrics import argument_f1

    pred = [{"tool": "Calculator", "args": {"expression": "1+1"}}]
    gold = [
        {"tool": "Calculator", "args": {"expression": "1+1"}},
        {"tool": "Calculator", "args": {"expression": "1+1"}},
    ]
    assert argument_f1(pred, gold) == pytest.approx(2.0 / 3.0)


# ------------------------------------------------------------------ Fix 4


class _CapturingCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs

        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _client_with_capture(**cfg_overrides):
    import types

    from src.config import LLMConfig
    from src.llm.client import LLMClient

    client = LLMClient(LLMConfig(**cfg_overrides))
    completions = _CapturingCompletions()
    client.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )
    return client, completions


def test_client_sends_vllm_extra_body_by_default():
    """Default (shipped YAMLs): behavior unchanged — the vLLM/Qwen
    extra_body is sent on every request."""
    client, completions = _client_with_capture()
    client.generate([{"role": "user", "content": "hi"}])
    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_client_omits_extra_body_when_toggle_disabled():
    """enable_thinking_toggle: false must strip the vLLM-specific
    extra_body so non-vLLM OpenAI-compatible endpoints are safe."""
    client, completions = _client_with_capture(enable_thinking_toggle=False)
    client.generate([{"role": "user", "content": "hi"}])
    assert "extra_body" not in completions.kwargs


def test_llm_config_toggle_defaults_true_and_loads_from_yaml(tmp_path):
    """Shipped configs (which omit the key) keep the old behavior; the key
    is settable from YAML."""
    from src.config import LLMConfig, load_config

    assert LLMConfig().enable_thinking_toggle is True
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "planner_llm:\n  model_name: gpt-4o\n  enable_thinking_toggle: false\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.planner_llm.enable_thinking_toggle is False
    assert cfg.judge_llm.enable_thinking_toggle is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
