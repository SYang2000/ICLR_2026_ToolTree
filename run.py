"""Main entry point for running ToolTree experiments.

Usage:
    python run.py --config configs/gta.yaml
    python run.py --config configs/toolbench.yaml --override mcts.max_rollouts=30
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from src.config import load_config
from src.agents.tooltree_agent import ToolTreeAgent
from src.evaluation.benchmarks import get_loader
from src.evaluation.metrics import compute_all_metrics


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace with config path and optional overrides.
    """
    raise NotImplementedError


def main() -> None:
    """Main entry point for running ToolTree experiments.

    Steps:
        1. Parse CLI arguments and load configuration from YAML.
        2. Initialize the benchmark data loader.
        3. Initialize the ToolTree agent.
        4. Run the agent on each benchmark instance.
        5. Compute and report evaluation metrics.
        6. Save results to the output directory.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
