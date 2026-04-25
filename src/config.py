"""Configuration dataclasses for ToolTree."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import yaml


@dataclass
class MCTSConfig:
    exploration_constant: float = 1.4
    max_rollouts: int = 60
    tau_pre: float = 0.3
    tau_post: float = 0.4
    early_stop_delta: float = 1e-3
    early_stop_patience: int = 10
    max_depth: int = 15
    top_k: int = 5


@dataclass
class LLMConfig:
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str = ""
    base_url: str = ""


@dataclass
class ToolTreeConfig:
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    planner_llm: LLMConfig = field(default_factory=LLMConfig)
    judge_llm: LLMConfig = field(default_factory=LLMConfig)
    benchmark: str = "gta"
    tool_retrieval_k: int = 20
    data_path: str = "data/"
    output_path: str = "outputs/"
    seed: int = 42
    num_samples: int | None = None


def _merge_into_dataclass(instance: Any, data: dict) -> Any:
    if not is_dataclass(instance) or not isinstance(data, dict):
        return instance
    for f in fields(instance):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(instance, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(instance, f.name, value)
    return instance


def load_config(yaml_path: str) -> ToolTreeConfig:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = ToolTreeConfig()
    _merge_into_dataclass(cfg, raw)
    return cfg
