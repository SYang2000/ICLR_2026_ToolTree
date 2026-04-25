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


def _flatten_args(args_dict: dict) -> list[str]:
    flat = []
    for tool, args in (args_dict or {}).items():
        for k, v in (args or {}).items():
            flat.append(f"{tool}::{k}={v}")
    return flat


def argument_f1(predicted_args: dict, gold_args: dict) -> float:
    return _multiset_f1(_flatten_args(predicted_args), _flatten_args(gold_args))


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
) -> dict[str, float]:
    if benchmark in ("gta", "mm"):
        tool_scores, arg_scores, plan_scores, exec_scores = [], [], [], []
        for p, r in zip(predictions, references):
            pred_tools = [s.get("tool") for s in (p.get("plan") or [])]
            gold_tools = [s.get("tool") for s in (r.get("gold_plan") or [])]
            tool_scores.append(tool_f1(pred_tools, gold_tools))

            pred_args = {s.get("tool"): s.get("args") for s in (p.get("plan") or [])}
            gold_args = {s.get("tool"): s.get("args") for s in (r.get("gold_plan") or [])}
            arg_scores.append(argument_f1(pred_args, gold_args))

            plan_scores.append(plan_f1(p.get("plan") or [], r.get("gold_plan") or []))
            exec_scores.append(execution_f1(p.get("execution") or [], r.get("gold_execution") or []))

        tool_avg = _avg(tool_scores)
        arg_avg = _avg(arg_scores)
        plan_avg = _avg(plan_scores)
        exec_avg = _avg(exec_scores)
        return {
            "tool_f1": tool_avg,
            "arg_f1": arg_avg,
            "plan_f1": plan_avg,
            "exec_f1": exec_avg,
            "avg_f1": _avg([tool_avg, arg_avg, plan_avg, exec_avg]),
        }
    elif benchmark in ("toolbench", "restbench"):
        results = [{"passed": bool(p.get("passed"))} for p in predictions]
        return {"pass_rate": pass_rate(results)}
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
