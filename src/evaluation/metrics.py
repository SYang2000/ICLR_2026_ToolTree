"""Evaluation metrics for ToolTree.

Closed-set metrics (GTA, m&m):
    - Tool F1: Tool selection F1 score.
    - Argument F1: Argument prediction F1 score.
    - Plan F1: Planning F1 score (end-to-end mode).
    - Execution F1: Execution F1 score (end-to-end mode).

Open-set metrics (ToolBench, RestBench):
    - Pass Rate: Fraction of correctly solved tasks.
    - Win Rate: Head-to-head comparison against baselines via LLM judge.
"""

from __future__ import annotations


def tool_f1(predicted_tools: list[str], gold_tools: list[str]) -> float:
    """Compute Tool selection F1 score (closed-set).

    Measures the overlap between predicted and gold-standard tool sequences.

    Args:
        predicted_tools: List of predicted tool names in order.
        gold_tools: List of gold-standard tool names in order.

    Returns:
        F1 score in [0, 1].
    """
    raise NotImplementedError


def argument_f1(predicted_args: dict, gold_args: dict) -> float:
    """Compute Argument prediction F1 score (closed-set).

    Measures the overlap between predicted and gold-standard arguments
    for each tool call in the sequence.

    Args:
        predicted_args: Dict mapping tool names to predicted argument dicts.
        gold_args: Dict mapping tool names to gold-standard argument dicts.

    Returns:
        F1 score in [0, 1].
    """
    raise NotImplementedError


def plan_f1(predicted_plan: list[dict], gold_plan: list[dict]) -> float:
    """Compute Planning F1 score (closed-set, end-to-end mode).

    Evaluates the full planning sequence including tool order and arguments.

    Args:
        predicted_plan: List of predicted {tool, args} dicts.
        gold_plan: List of gold-standard {tool, args} dicts.

    Returns:
        F1 score in [0, 1].
    """
    raise NotImplementedError


def execution_f1(predicted_exec: list[dict], gold_exec: list[dict]) -> float:
    """Compute Execution F1 score (closed-set, end-to-end mode).

    Evaluates execution correctness by comparing predicted and gold
    tool outputs.

    Args:
        predicted_exec: List of predicted execution result dicts.
        gold_exec: List of gold-standard execution result dicts.

    Returns:
        F1 score in [0, 1].
    """
    raise NotImplementedError


def pass_rate(results: list[dict]) -> float:
    """Compute Pass Rate for open-set benchmarks (ToolBench, RestBench).

    Fraction of tasks where the agent produced a correct solution,
    as determined by the benchmark's judge protocol.

    Args:
        results: List of result dicts, each containing a "passed" bool key.

    Returns:
        Pass rate in [0, 1].
    """
    raise NotImplementedError


def win_rate(
    our_results: list[dict],
    baseline_results: list[dict],
    judge_fn: object,
) -> float:
    """Compute Win Rate via head-to-head judge comparison (open-set).

    Compares our agent's outputs against a baseline using an LLM judge
    to determine which solution is better for each task.

    Args:
        our_results: List of our agent's result dicts.
        baseline_results: List of baseline result dicts (same tasks).
        judge_fn: A callable judge (e.g., LLMJudge instance) for comparison.

    Returns:
        Win rate in [0, 1].
    """
    raise NotImplementedError


def compute_all_metrics(
    predictions: list[dict],
    references: list[dict],
    benchmark: str,
) -> dict[str, float]:
    """Compute all relevant metrics for a given benchmark type.

    Dispatches to the appropriate metric functions based on benchmark:
        - "gta", "mm": tool_f1, argument_f1, plan_f1, execution_f1
        - "toolbench", "restbench": pass_rate, win_rate

    Args:
        predictions: List of prediction dicts from the agent.
        references: List of gold-standard reference dicts.
        benchmark: Benchmark name ("gta", "mm", "toolbench", "restbench").

    Returns:
        Dict mapping metric names to their computed values.
    """
    raise NotImplementedError
