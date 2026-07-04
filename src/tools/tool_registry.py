"""Tool card storage and retrieval for ToolTree.

Supports both closed-set (fixed tools like GTA's 14 APIs) and open-set
(retrieval-based like ToolBench's 16,464 APIs) scenarios.
"""

from __future__ import annotations

import json
import os
import re
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


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", (text or "").lower()) if t]


class ToolRegistry:
    """Registry that stores and retrieves tool cards."""

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, ToolCard] = {}

    def register_tool(self, tool_card: ToolCard) -> None:
        """Add a tool card to the registry."""
        if tool_card.name in self._tools:
            raise ValueError(f"Tool already registered: {tool_card.name}")
        self._tools[tool_card.name] = tool_card

    def load_from_directory(self, path: str) -> None:
        """Load all tool cards from JSON files in a directory."""
        if not os.path.isdir(path):
            return
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(path, fname)
            try:
                with open(fpath, "r") as f:
                    payload = json.load(f)
            except Exception:
                continue
            if isinstance(payload, list):
                for item in payload:
                    self._register_from_dict(item)
            elif isinstance(payload, dict):
                self._register_from_dict(payload)

    def _register_from_dict(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        name = payload.get("name")
        if not name or name in self._tools:
            return
        card = ToolCard(
            name=name,
            description=payload.get("description", ""),
            input_schema=payload.get("input_schema", {}) or {},
            output_schema=payload.get("output_schema", {}) or {},
            examples=payload.get("examples", []) or [],
            domain=payload.get("domain", "") or "",
        )
        self._tools[name] = card

    def get_tool(self, name: str) -> ToolCard:
        """Retrieve a tool card by exact name."""
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_tools(self) -> list[ToolCard]:
        """Return all registered tool cards."""
        return list(self._tools.values())

    def retrieve_tools(self, query: str, k: int = 20) -> list[ToolCard]:
        """Retrieve top-K relevant tools for a query using BM25 or keyword overlap."""
        tools = list(self._tools.values())
        if not tools:
            return []
        corpus = [
            _tokenize(f"{t.name} {t.description} {t.domain}") for t in tools
        ]
        query_tokens = _tokenize(query)
        if not query_tokens:
            return tools[:k]
        try:
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query_tokens)
        except Exception:
            scores = []
            for doc in corpus:
                doc_set = set(doc)
                scores.append(sum(1.0 for t in query_tokens if t in doc_set))
        ranked = sorted(
            zip(tools, scores), key=lambda pair: pair[1], reverse=True
        )
        return [t for t, _ in ranked[:k]]

    #: Input parameter types that require a matching modality in the context
    #: (plain JSON types — string/number/boolean/object/array — are always
    #: satisfiable from the dialogue text).
    _MODALITY_TYPES = {"image", "file", "audio", "video"}

    def check_schema_compatibility(
        self, tool_card: "ToolCard | dict", context: dict
    ) -> bool:
        """Check if a tool's input schema is type-compatible with the context
        (paper Sec. 3: admissibility gating at selection and expansion).

        A tool whose REQUIRED input parameter declares a non-text modality
        (image/file/audio/video) is admissible only if the context can supply
        it: initial attachments (files/images/resources) or a prior execution
        output in the accumulated history. Finer-grained I/O type matching is
        deferred to the judge's pre-evaluation.
        """
        if isinstance(tool_card, dict):
            schema = tool_card.get("input_schema") or {}
        else:
            schema = tool_card.input_schema or {}
        props = schema.get("properties") or {}
        required = schema.get("required")
        if required is None:
            required = list(props)

        context = context or {}
        available: set[str] = set()
        if context.get("images"):
            available.add("image")
        if context.get("files") or context.get("resources"):
            available |= self._MODALITY_TYPES
        if context.get("history"):
            # prior tool outputs may provide any modality
            available |= self._MODALITY_TYPES

        for pname in required:
            ptype = (props.get(pname) or {}).get("type", "")
            if ptype in self._MODALITY_TYPES and ptype not in available:
                return False
        return True
