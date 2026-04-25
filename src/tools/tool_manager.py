"""Tool execution and caching manager for ToolTree.

Handles invoking tools/APIs with deterministic caching keyed by
(action, args) to avoid redundant calls within a rollout.
Persistent failures attach an error token so downstream scoring
can handle the outcome explicitly.
"""

from __future__ import annotations

import hashlib
import json
import re

from src.llm.client import LLMClient
from src.tools.tool_registry import ToolCard, ToolRegistry


class ToolManager:
    """Manages tool execution with deterministic caching."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        """Initialize with a tool registry and empty execution cache."""
        self.tool_registry = tool_registry
        self._cache: dict[str, dict] = {}

    def execute(self, tool_name: str, args: dict) -> dict:
        """Execute a tool call, using cache if available."""
        key = self._get_cache_key(tool_name, args)
        if key in self._cache:
            return self._cache[key]
        try:
            output = self._invoke_tool(tool_name, args)
        except Exception as exc:
            output = {
                "tool_name": tool_name,
                "args": args,
                "error": f"invocation_failed: {exc}",
            }
        self._cache[key] = output
        return output

    def _invoke_tool(self, tool_name: str, args: dict) -> dict:
        """Actually invoke the tool.

        Step-by-step evaluation mode: returns a mock dict describing the
        intended call without running the underlying API. This keeps the
        search tractable on benchmarks where we only score plan-level F1.
        """
        return {
            "tool_name": tool_name,
            "args": args or {},
            "output": "<mock: step-by-step mode>",
        }

    def _get_cache_key(self, tool_name: str, args: dict) -> str:
        """Generate a deterministic cache key from (tool_name, args)."""
        try:
            args_str = json.dumps(args or {}, sort_keys=True, default=str)
        except Exception:
            args_str = str(sorted((args or {}).items()))
        digest = hashlib.sha1(f"{tool_name}::{args_str}".encode("utf-8")).hexdigest()
        return f"{tool_name}::{digest}"

    def is_cached(self, tool_name: str, args: dict) -> bool:
        """Check if a result for this (tool_name, args) is already cached."""
        return self._get_cache_key(tool_name, args) in self._cache

    def get_cached(self, tool_name: str, args: dict) -> dict | None:
        """Return cached result or None if not cached."""
        return self._cache.get(self._get_cache_key(tool_name, args))

    def clear_cache(self) -> None:
        """Clear the execution cache (e.g., between tasks)."""
        self._cache.clear()

    def generate_arg_draft(
        self,
        tool_card: ToolCard,
        context: dict,
        planner_llm: LLMClient,
    ) -> dict:
        """Generate a minimal, schema-valid argument draft for a tool."""
        try:
            schema_str = json.dumps(tool_card.input_schema or {}, indent=2, default=str)
        except Exception:
            schema_str = str(tool_card.input_schema or {})
        try:
            context_str = json.dumps(context or {}, indent=2, default=str)
        except Exception:
            context_str = str(context or {})

        system = (
            "You generate JSON argument drafts for a tool call. "
            "Output must be a single JSON object whose keys match the tool's "
            "input schema. Do not include any explanation."
        )
        user = (
            f"Tool: {tool_card.name}\n"
            f"Description: {tool_card.description}\n"
            f"Input schema:\n{schema_str}\n\n"
            f"Current context:\n{context_str}\n\n"
            "Produce a JSON object with the argument draft."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return planner_llm.generate_json(messages)
        except Exception:
            try:
                text = planner_llm.generate(messages)
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    return json.loads(m.group(0))
            except Exception:
                pass
            return {}
