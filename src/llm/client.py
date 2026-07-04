"""Unified LLM client against an OpenAI-compatible endpoint (vLLM, OpenAI, etc.)."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from src.config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        from openai import OpenAI

        self.config = config
        base_url = config.base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Chat-complete `messages` and return the text content.

        When config.enable_thinking_toggle is True (default, matching the
        shipped vLLM/Qwen configs), the request carries the vLLM-specific
        extra_body {"chat_template_kwargs": {"enable_thinking": false}} to
        suppress Qwen thinking blocks. Set enable_thinking_toggle: false in
        the LLM config for non-vLLM endpoints (e.g. the OpenAI API), which
        may reject or misinterpret that field.
        """
        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if getattr(self.config, "enable_thinking_toggle", True):
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def generate_json(self, messages: list[dict[str, str]]) -> dict:
        text = self.generate(messages, json_mode=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise ValueError(f"Could not parse JSON from response: {text[:200]}")
