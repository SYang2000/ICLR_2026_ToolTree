"""Unified LLM API wrapper supporting OpenAI, Anthropic, and local models."""

from __future__ import annotations

from src.config import LLMConfig


class LLMClient:
    """Unified wrapper for LLM API calls.

    Supports OpenAI (GPT-4o, GPT-4o-mini), Anthropic, and local
    models (LLaMA, Qwen) via a consistent interface.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLM client with model configuration.

        Args:
            config: LLM configuration specifying model, temperature, etc.
        """
        raise NotImplementedError

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send messages to the LLM and return the text response.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            temperature: Override default temperature if provided.
            max_tokens: Override default max tokens if provided.
            json_mode: If True, request JSON-formatted output.

        Returns:
            The model's text response.
        """
        raise NotImplementedError

    def generate_json(self, messages: list[dict[str, str]]) -> dict:
        """Send messages and parse the response as a JSON object.

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        raise NotImplementedError
