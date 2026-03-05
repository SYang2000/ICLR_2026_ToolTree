"""Benchmark dataset loaders for ToolTree.

Supports four benchmarks across two regimes:
    Closed-set: GTA (14 tools), m&m (33 tools)
    Open-set: ToolBench (16,464 APIs), RestBench (143 endpoints)
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BenchmarkLoader(ABC):
    """Abstract loader for benchmark datasets."""

    @abstractmethod
    def load(self, split: str = "test") -> list[dict]:
        """Load benchmark data for the specified split.

        Each item in the returned list contains:
            - "query": str -- the user query.
            - "gold_tools": list[str] -- gold-standard tool sequence.
            - "gold_args": list[dict] -- gold-standard arguments per tool.
            - "context": dict -- any additional context (images, etc.).

        Args:
            split: Data split to load ("train", "val", "test").

        Returns:
            List of benchmark instance dicts.
        """


class GTALoader(BenchmarkLoader):
    """Loader for GTA benchmark (General Tool Agent).

    GTA provides 14 APIs with typed I/O and multi-hop compositional tasks.
    Evaluated in both step-by-step and end-to-end modes.

    Metrics: Tool F1, Arg F1, Plan F1, Execution F1.
    """

    def __init__(self, data_path: str) -> None:
        """Initialize GTA loader.

        Args:
            data_path: Path to the GTA dataset directory.
        """
        raise NotImplementedError

    def load(self, split: str = "test") -> list[dict]:
        """Load GTA benchmark instances.

        Args:
            split: Data split ("test" by default).

        Returns:
            List of GTA task dicts.
        """
        raise NotImplementedError


class MMLoader(BenchmarkLoader):
    """Loader for m&m benchmark (Multi-modal and Multi-step Tool Use).

    m&m features 33 APIs spanning vision, text, and arithmetic tasks.
    Emphasizes input schema matching and argument consistency.

    Metrics: Tool F1, Arg F1, Plan F1, Execution F1.
    """

    def __init__(self, data_path: str) -> None:
        """Initialize m&m loader.

        Args:
            data_path: Path to the m&m dataset directory.
        """
        raise NotImplementedError

    def load(self, split: str = "test") -> list[dict]:
        """Load m&m benchmark instances.

        Args:
            split: Data split ("test" by default).

        Returns:
            List of m&m task dicts.
        """
        raise NotImplementedError


class ToolBenchLoader(BenchmarkLoader):
    """Loader for ToolBench benchmark (open-set, 16,464 APIs).

    Each task requires: (1) retrieve relevant APIs from the pool,
    (2) generate valid input arguments, and (3) compose executable
    tool sequences. Evaluated via Pass Rate and Win Rate.
    """

    def __init__(self, data_path: str) -> None:
        """Initialize ToolBench loader.

        Args:
            data_path: Path to the ToolBench dataset directory.
        """
        raise NotImplementedError

    def load(self, split: str = "test") -> list[dict]:
        """Load ToolBench benchmark instances.

        Args:
            split: Data split ("test" by default).

        Returns:
            List of ToolBench task dicts.
        """
        raise NotImplementedError


class RestBenchLoader(BenchmarkLoader):
    """Loader for RestBench benchmark (TMDB + Spotify, 143 endpoints).

    Tasks require multi-step planning, slot filling, and reasoning
    over RESTful API endpoint chains.

    Metrics: Pass Rate, Win Rate.
    """

    def __init__(self, data_path: str, domain: str = "tmdb") -> None:
        """Initialize RestBench loader.

        Args:
            data_path: Path to the RestBench dataset directory.
            domain: Domain to load ("tmdb" or "spotify").
        """
        raise NotImplementedError

    def load(self, split: str = "test") -> list[dict]:
        """Load RestBench benchmark instances for the specified domain.

        Args:
            split: Data split ("test" by default).

        Returns:
            List of RestBench task dicts.
        """
        raise NotImplementedError


def get_loader(benchmark: str, data_path: str, **kwargs) -> BenchmarkLoader:
    """Factory function to get the appropriate loader by benchmark name.

    Args:
        benchmark: Benchmark name ("gta", "mm", "toolbench", "restbench").
        data_path: Path to the dataset directory.
        **kwargs: Additional arguments passed to the loader (e.g., domain).

    Returns:
        An instance of the appropriate BenchmarkLoader subclass.

    Raises:
        ValueError: If the benchmark name is not recognized.
    """
    raise NotImplementedError
