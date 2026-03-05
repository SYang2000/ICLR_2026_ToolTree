"""Configuration dataclasses for ToolTree.

Hyperparameters from Appendix B.4:
    λ = 1.4 (exploration constant)
    R_max = 60 (max rollouts)
    τ_pre = 0.3 (pre-pruning threshold)
    τ_post = 0.4 (post-pruning threshold)
    Early stop if Q improvement < 1e-3 over 10 consecutive rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class MCTSConfig:
    """Configuration for the Monte Carlo Tree Search algorithm.

    Attributes:
        exploration_constant: λ in the prior-augmented UCT formula (Eq. 1).
        max_rollouts: R_max — maximum number of MCTS rollouts per query.
        tau_pre: Pre-pruning threshold; discard actions with r_pre < τ_pre.
        tau_post: Post-pruning threshold; mark nodes non-expandable if r_post < τ_post.
        early_stop_delta: Minimum Q-value improvement to continue searching.
        early_stop_patience: Number of consecutive rollouts with no improvement before stopping.
        max_depth: Maximum depth of tool chains (number of sequential tool calls).
    """

    exploration_constant: float = 1.4
    max_rollouts: int = 60
    tau_pre: float = 0.3
    tau_post: float = 0.4
    early_stop_delta: float = 1e-3
    early_stop_patience: int = 10
    max_depth: int = 15


@dataclass
class LLMConfig:
    """Configuration for an LLM API client.

    Attributes:
        model_name: Model identifier (e.g., "gpt-4o", "gpt-4o-mini").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        api_key: API key (loaded from environment if empty).
    """

    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str = ""


@dataclass
class ToolTreeConfig:
    """Top-level configuration for the ToolTree framework.

    Attributes:
        mcts: MCTS algorithm hyperparameters.
        planner_llm: LLM config for action generation / planning.
        judge_llm: LLM config for pre- and post-evaluation judging.
        benchmark: Benchmark name ("gta", "mm", "toolbench", "restbench").
        tool_retrieval_k: Top-K for open-set tool retrieval.
        data_path: Path to benchmark data directory.
        output_path: Path to save results.
    """

    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    planner_llm: LLMConfig = field(default_factory=LLMConfig)
    judge_llm: LLMConfig = field(default_factory=LLMConfig)
    benchmark: str = "gta"
    tool_retrieval_k: int = 20
    data_path: str = "data/"
    output_path: str = "outputs/"


def load_config(yaml_path: str) -> ToolTreeConfig:
    """Load configuration from a YAML file and return a ToolTreeConfig.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A fully populated ToolTreeConfig instance.
    """
    raise NotImplementedError
