"""Benchmark dataset loaders for ToolTree.

Supports four benchmarks across two regimes:
    Closed-set: GTA (14 tools), m&m (33 tools)
    Open-set: ToolBench (16,464 APIs), RestBench (143 endpoints)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod


class BenchmarkLoader(ABC):
    """Abstract loader for benchmark datasets."""

    @abstractmethod
    def load(self, split: str = "test") -> list[dict]:
        """Load benchmark data for the specified split."""


def _read_json_file(path: str) -> list | dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_items(data_path: str) -> list[dict]:
    """Read items from either a JSON/JSONL file or a directory of JSON files."""
    if os.path.isfile(data_path):
        if data_path.endswith(".jsonl"):
            items: list[dict] = []
            with open(data_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return items
        payload = _read_json_file(data_path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "items", "tasks", "examples"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        return []
    if os.path.isdir(data_path):
        items: list[dict] = []
        for fname in sorted(os.listdir(data_path)):
            if fname.endswith(".json"):
                fpath = os.path.join(data_path, fname)
                try:
                    payload = _read_json_file(fpath)
                except Exception:
                    continue
                if isinstance(payload, list):
                    items.extend(payload)
                elif isinstance(payload, dict):
                    items.append(payload)
        return items
    return []


def _extract_query_from_dialogs(dialogs: list) -> str:
    for msg in dialogs or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from") or ""
        if role.lower() in ("user", "human"):
            content = msg.get("content") or msg.get("value") or ""
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict):
                        texts.append(part.get("text") or part.get("content") or "")
                    else:
                        texts.append(str(part))
                return " ".join(t for t in texts if t)
            return str(content)
    return ""


def _extract_gold_plan(item: dict) -> list[dict]:
    """Try common fields for the gold tool-call plan."""
    for key in ("gold_plan", "plan", "steps", "tool_calls", "actions"):
        value = item.get(key)
        if isinstance(value, list):
            plan: list[dict] = []
            for step in value:
                if not isinstance(step, dict):
                    continue
                tool = (
                    step.get("tool")
                    or step.get("tool_name")
                    or step.get("name")
                    or step.get("api")
                )
                args = (
                    step.get("args")
                    or step.get("arguments")
                    or step.get("tool_args")
                    or step.get("input")
                    or {}
                )
                if tool:
                    plan.append({"tool": tool, "args": args if isinstance(args, dict) else {}})
            if plan:
                return plan
    dialogs = item.get("dialogs") or item.get("dialog") or []
    plan = []
    for msg in dialogs if isinstance(dialogs, list) else []:
        if not isinstance(msg, dict):
            continue
        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool = call.get("name") or call.get("tool") or call.get("function", {}).get("name")
                args = call.get("arguments") or call.get("args") or call.get("function", {}).get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                if tool:
                    plan.append({"tool": tool, "args": args if isinstance(args, dict) else {}})
    return plan


def _extract_gold_tools(item: dict, plan: list[dict]) -> list[str]:
    for key in ("gold_tools", "tools_used", "tool_sequence"):
        value = item.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
    return [step["tool"] for step in plan]


def _extract_context(item: dict) -> dict:
    raw = item.get("context")
    if isinstance(raw, dict):
        context = dict(raw)
    else:
        context = {}
    for key in ("files", "images", "resources"):
        value = item.get(key)
        if value and key not in context:
            context[key] = value
    return context


def _normalize_item(item: dict) -> dict:
    query = item.get("query") or item.get("question") or item.get("instruction")
    if not query:
        dialogs = item.get("dialogs") or item.get("dialog") or []
        if isinstance(dialogs, list):
            query = _extract_query_from_dialogs(dialogs)
    plan = _extract_gold_plan(item)
    gold_tools = _extract_gold_tools(item, plan)
    context = _extract_context(item)
    available_tools = item.get("available_tools") or item.get("tools") or []
    return {
        "query": query or "",
        "gold_tools": gold_tools,
        "gold_plan": plan,
        "context": context,
        "available_tools": available_tools,
    }


class GTALoader(BenchmarkLoader):
    """Loader for GTA benchmark (General Tool Agent)."""

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def load(self, split: str = "test") -> list[dict]:
        candidates = [
            self.data_path,
            os.path.join(self.data_path, f"{split}.json"),
            os.path.join(self.data_path, "gta.json"),
            os.path.join(self.data_path, "dataset.json"),
        ]
        raw: list[dict] = []
        for path in candidates:
            if os.path.exists(path):
                raw = _load_items(path)
                if raw:
                    break
        return [_normalize_item(item) for item in raw]


class MMLoader(BenchmarkLoader):
    """Loader for m&m benchmark (Multi-modal and Multi-step Tool Use)."""

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def load(self, split: str = "test") -> list[dict]:
        candidates = [
            self.data_path,
            os.path.join(self.data_path, f"{split}.json"),
            os.path.join(self.data_path, "mm.json"),
            os.path.join(self.data_path, "mnm.json"),
            os.path.join(self.data_path, "dataset.json"),
        ]
        raw: list[dict] = []
        for path in candidates:
            if os.path.exists(path):
                raw = _load_items(path)
                if raw:
                    break
        return [_normalize_item(item) for item in raw]


class ToolBenchLoader(BenchmarkLoader):
    """Loader for ToolBench benchmark (open-set, 16,464 APIs)."""

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def load(self, split: str = "test") -> list[dict]:
        raise NotImplementedError("Not supported in step-by-step release")


class RestBenchLoader(BenchmarkLoader):
    """Loader for RestBench benchmark (TMDB + Spotify, 143 endpoints)."""

    def __init__(self, data_path: str, domain: str = "tmdb") -> None:
        self.data_path = data_path
        self.domain = domain

    def load(self, split: str = "test") -> list[dict]:
        raise NotImplementedError("Not supported in step-by-step release")


def get_loader(benchmark: str, data_path: str, **kwargs) -> BenchmarkLoader:
    """Factory function to get the appropriate loader by benchmark name."""
    name = (benchmark or "").lower()
    if name == "gta":
        return GTALoader(data_path)
    if name in ("mm", "mnm", "m&m"):
        return MMLoader(data_path)
    if name == "toolbench":
        return ToolBenchLoader(data_path)
    if name == "restbench":
        return RestBenchLoader(data_path, domain=kwargs.get("domain", "tmdb"))
    raise ValueError(f"Unknown benchmark: {benchmark}")
