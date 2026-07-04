"""Evaluation metrics for ToolTree."""

from __future__ import annotations

from collections import Counter
from typing import Callable


def _multiset_f1(pred: list, gold: list) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pc, gc = Counter(pred), Counter(gold)
    tp = sum((pc & gc).values())
    if tp == 0:
        return 0.0
    precision = tp / sum(pc.values())
    recall = tp / sum(gc.values())
    return 2 * precision * recall / (precision + recall)


def tool_f1(predicted_tools: list[str], gold_tools: list[str]) -> float:
    return _multiset_f1(predicted_tools, gold_tools)


def _flatten_step_args(plan: list[dict]) -> list[str]:
    """Flatten a plan's arguments per STEP into `tool::key=value` tokens.

    Flattening is per step, not per tool name: two calls to the same tool
    contribute both argument sets with multiplicity. (The released code
    folded steps into a {tool: args} dict, silently dropping all but the
    last call of a reused tool — contradicting A_rem semantics, under which
    tool reuse with different args is admissible and required by GTA gold
    chains.)
    """
    flat = []
    for step in (plan or []):
        tool = step.get("tool")
        for k, v in (step.get("args") or {}).items():
            flat.append(f"{tool}::{k}={v}")
    return flat


def argument_f1(predicted_plan: list[dict], gold_plan: list[dict]) -> float:
    """Multiset F1 over per-step flattened `tool::key=value` argument tokens.

    Takes the two plans (lists of {tool, args} steps); duplicate tool calls
    are NOT collapsed.
    """
    return _multiset_f1(
        _flatten_step_args(predicted_plan), _flatten_step_args(gold_plan)
    )


def _plan_tokens(plan: list[dict]) -> list[str]:
    out = []
    for step in (plan or []):
        args = step.get("args") or {}
        try:
            args_repr = sorted(args.items())
        except Exception:
            args_repr = str(args)
        out.append(f"{step.get('tool')}::{args_repr}")
    return out


def plan_f1(predicted_plan: list[dict], gold_plan: list[dict]) -> float:
    return _multiset_f1(_plan_tokens(predicted_plan), _plan_tokens(gold_plan))


def _exec_tokens(exec_list: list[dict]) -> list[str]:
    return [f"{step.get('tool')}::{step.get('output')}" for step in (exec_list or [])]


def execution_f1(predicted_exec: list[dict], gold_exec: list[dict]) -> float:
    return _multiset_f1(_exec_tokens(predicted_exec), _exec_tokens(gold_exec))


def pass_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.get("passed"))
    return passed / len(results)


def win_rate(
    our_results: list[dict],
    baseline_results: list[dict],
    judge_fn: Callable,
) -> float:
    if not our_results:
        return 0.0
    wins = 0
    for ours, base in zip(our_results, baseline_results):
        if judge_fn(ours, base) == "win":
            wins += 1
    return wins / len(our_results)


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_all_metrics(
    predictions: list[dict],
    references: list[dict],
    benchmark: str,
) -> dict[str, float | None]:
    """Compute benchmark metrics.

    Closed-set (gta/mm) scoring scope — a deliberate semantics change from
    the released code:

    - exec_f1 is computed only over samples whose reference carries a
      non-empty `gold_execution`; a sample without an execution reference
      cannot be graded on execution. In particular, an EMPTY prediction
      against a missing gold no longer scores exec_f1 = 1.0 (the released
      behavior let failed empty plans inflate exec_f1/avg_f1). When no
      sample is gradeable, exec_f1 is reported as None (JSON null) and
      excluded from avg_f1.
    - avg_f1 averages the AVAILABLE metrics only.
    - arg_f1 flattens arguments per step (multiset), so repeated calls to
      the same tool all count (see argument_f1).
    """
    if benchmark in ("gta", "mm"):
        tool_scores, arg_scores, plan_scores, exec_scores = [], [], [], []
        for p, r in zip(predictions, references):
            pred_tools = [s.get("tool") for s in (p.get("plan") or [])]
            gold_tools = [s.get("tool") for s in (r.get("gold_plan") or [])]
            tool_scores.append(tool_f1(pred_tools, gold_tools))

            arg_scores.append(
                argument_f1(p.get("plan") or [], r.get("gold_plan") or [])
            )

            plan_scores.append(plan_f1(p.get("plan") or [], r.get("gold_plan") or []))
            if r.get("gold_execution"):
                exec_scores.append(
                    execution_f1(p.get("execution") or [], r.get("gold_execution"))
                )

        tool_avg = _avg(tool_scores)
        arg_avg = _avg(arg_scores)
        plan_avg = _avg(plan_scores)
        exec_avg = _avg(exec_scores) if exec_scores else None
        available = [m for m in (tool_avg, arg_avg, plan_avg, exec_avg) if m is not None]
        return {
            "tool_f1": tool_avg,
            "arg_f1": arg_avg,
            "plan_f1": plan_avg,
            "exec_f1": exec_avg,
            "avg_f1": _avg(available),
        }
    elif benchmark in ("toolbench", "restbench"):
        results = [{"passed": bool(p.get("passed"))} for p in predictions]
        return {"pass_rate": pass_rate(results)}
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
