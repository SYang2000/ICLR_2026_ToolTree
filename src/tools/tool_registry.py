"""Tool card storage and retrieval for ToolTree.

Supports both closed-set (fixed tools like GTA's 14 APIs) and open-set
(retrieval-based like ToolBench's 16,464 APIs) scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCard:
    """Metadata for a single tool (see Appendix B.6 for schema).

    Attributes:
        name: Tool name (e.g., "Medical Object Detection").
        description: What the tool does.
        input_schema: Dict describing input parameters and their types.
        output_schema: Dict describing output structure.
        examples: List of example input/output pairs.
        domain: Domain category (e.g., "medical", "vision", "math").
    """

    name: str
    description: str
    input_schema: dict
    output_schema: dict
    examples: list[dict] = field(default_factory=list)
    domain: str = ""


class ToolRegistry:
    """Registry that stores and retrieves tool cards.

    Supports both closed-set (all tools loaded upfront) and open-set
    (retrieval via BM25/embedding similarity) scenarios.
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        raise NotImplementedError

    def register_tool(self, tool_card: ToolCard) -> None:
        """Add a tool card to the registry.

        Args:
            tool_card: The tool card to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        raise NotImplementedError

    def load_from_directory(self, path: str) -> None:
        """Load all tool cards from JSON files in a directory.

        Each JSON file should contain a single tool card following
        the schema in Appendix B.6.

        Args:
            path: Path to directory containing tool card JSON files.
        """
        raise NotImplementedError

    def get_tool(self, name: str) -> ToolCard:
        """Retrieve a tool card by exact name.

        Args:
            name: The tool name to look up.

        Returns:
            The matching ToolCard.

        Raises:
            KeyError: If no tool with that name exists.
        """
        raise NotImplementedError

    def get_all_tools(self) -> list[ToolCard]:
        """Return all registered tool cards.

        Returns:
            List of all ToolCard instances in the registry.
        """
        raise NotImplementedError

    def retrieve_tools(self, query: str, k: int = 20) -> list[ToolCard]:
        """Retrieve top-K relevant tools for a query (open-set retrieval).

        Uses BM25 or embedding-based retrieval over tool descriptions
        to find the most relevant tools for the given query.

        Args:
            query: The user query to match tools against.
            k: Number of tools to retrieve.

        Returns:
            List of top-K ToolCard instances sorted by relevance.
        """
        raise NotImplementedError

    def check_schema_compatibility(
        self, tool_card: ToolCard, context: dict
    ) -> bool:
        """Check if a tool's input schema is type-compatible with current context.

        Verifies that the data types and modalities available in the
        context match the tool's required input schema.

        Args:
            tool_card: The tool to check compatibility for.
            context: The current dialogue context with available data.

        Returns:
            True if the tool can be invoked given the current context.
        """
        raise NotImplementedError
