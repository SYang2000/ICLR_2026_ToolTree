"""Main entry point for running ToolTree experiments.

Usage:
    python run.py --config configs/gta.yaml
    python run.py --config configs/mm.yaml --override mcts.max_rollouts=30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import asdict

from src.config import load_config
from src.agents.tooltree_agent import ToolTreeAgent
from src.evaluation.benchmarks import get_loader
from src.evaluation.metrics import compute_all_metrics


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run ToolTree on a benchmark.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value, e.g. --override mcts.max_rollouts=30",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write results JSON (overrides config output_path).",
    )
    return parser.parse_args()


def _coerce(value: str):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        pass
    try:
        return json.loads(value)
    except Exception:
        return value


def _apply_override(cfg, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown config path: {dotted_key}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"Unknown config path: {dotted_key}")
    setattr(target, leaf, value)


def apply_overrides(cfg, overrides: list[str]) -> None:
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override (expected key=value): {override}")
        key, raw_val = override.split("=", 1)
        _apply_override(cfg, key.strip(), _coerce(raw_val.strip()))


def _redacted_config(cfg) -> dict:
    """Resolved config as a dict, with API keys stripped."""
    resolved = asdict(cfg)
    for key in ("planner_llm", "judge_llm"):
        resolved.get(key, {}).pop("api_key", None)
    return resolved


def main() -> None:
    """Main entry point for running ToolTree experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.seed is not None:
        cfg.seed = args.seed
    # echo the effective config so run artifacts can reconstruct it
    logger.info("Resolved config: %s", json.dumps(_redacted_config(cfg), default=str))

    random.seed(cfg.seed)
    try:
        import numpy as np

        np.random.seed(cfg.seed)
    except Exception:
        pass

    loader = get_loader(cfg.benchmark, cfg.data_path)
    items = loader.load()
    if cfg.num_samples is not None:
        items = items[: cfg.num_samples]
    logger.info("Loaded %d samples for benchmark=%s", len(items), cfg.benchmark)

    agent = ToolTreeAgent(cfg)

    predictions: list[dict] = []
    references: list[dict] = []
    started_at = time.time()
    for idx, item in enumerate(items):
        agent.reset()
        query = item.get("query", "")
        context = dict(item.get("context", {}) or {})
        if item.get("available_tools"):
            context["available_tools"] = item["available_tools"]
        t0 = time.time()
        try:
            result = agent.solve(query, context)
        except Exception as exc:
            logger.exception("solve() failed on sample %d: %s", idx, exc)
            result = {
                "answer": f"<error: {exc}>",
                "trajectory": [],
                "reward": 0.0,
                "rollouts_used": 0,
            }
        elapsed = time.time() - t0

        plan = [
            {"tool": step.get("action"), "args": step.get("args") or {}}
            for step in result.get("trajectory", [])
            if step.get("action")
        ]
        execution = [
            {"tool": step.get("action"), "output": step.get("output")}
            for step in result.get("trajectory", [])
            if step.get("action")
        ]
        predictions.append(
            {
                "query": query,
                "answer": result.get("answer"),
                "plan": plan,
                "execution": execution,
                "reward": result.get("reward", 0.0),
                "rollouts_used": result.get("rollouts_used", 0),
                "elapsed_seconds": elapsed,
            }
        )
        references.append(
            {
                "query": query,
                "gold_tools": item.get("gold_tools", []),
                "gold_plan": item.get("gold_plan", []),
                "gold_execution": item.get("gold_execution", []),
                "gold_answer": item.get("gold_answer"),
            }
        )
        logger.info(
            "[%d/%d] reward=%.3f rollouts=%d elapsed=%.1fs",
            idx + 1,
            len(items),
            result.get("reward", 0.0),
            result.get("rollouts_used", 0),
            elapsed,
        )

    metrics = compute_all_metrics(predictions, references, cfg.benchmark)
    total_time = time.time() - started_at
    logger.info("Metrics: %s", json.dumps(metrics))
    logger.info("Total runtime: %.1fs", total_time)

    output_path = args.output or cfg.output_path
    if output_path:
        if os.path.isdir(output_path) or output_path.endswith(os.sep) or not output_path.endswith(".json"):
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, "results.json")
        else:
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            output_file = output_path
        payload = {
            "config_path": args.config,
            "resolved_config": _redacted_config(cfg),
            "benchmark": cfg.benchmark,
            "num_samples": len(items),
            "metrics": metrics,
            "predictions": predictions,
            "references": references,
            "total_seconds": total_time,
        }
        with open(output_file, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Wrote results to %s", output_file)


if __name__ == "__main__":
    main()
