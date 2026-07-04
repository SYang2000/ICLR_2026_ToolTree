"""LLM Judge for dual pre- and post-evaluation scoring.

Uses prompt templates from Appendix B.7 (pre-evaluation) and
Appendix B.8 (post-evaluation) to score tool calls on [0, 1].
"""

from __future__ import annotations

import hashlib
import json
import re

from src.llm.client import LLMClient
from src.prompts.pre_eval_prompt import (
    PRE_EVAL_SYSTEM_PROMPT,
    build_pre_eval_user_message,
)
from src.prompts.post_eval_prompt import (
    POST_EVAL_SYSTEM_PROMPT,
    build_post_eval_user_message,
)


class LLMJudge:
    """LLM-based judge for pre-evaluation and post-evaluation scoring.

    Pre-evaluation (r_pre): Estimates the utility of a tool call BEFORE
    execution, based on schema compatibility, relevance, and context fit.

    Post-evaluation (r_post): Assesses the actual contribution of a tool
    call AFTER execution, based on task-consistency, correctness, and
    constraint satisfaction.

    Judgments are memoized by content: a sha256 over the canonical JSON of
    (kind, query, context, tool card, args[, output]). This serves two
    purposes: (1) an identical judgment input is never paid for twice, even
    when it reaches the judge through a different tree node; (2) consistency —
    temperature-0 sampling on the vLLM backend is NOT bit-deterministic in
    practice (identical prompts have returned different completions in
    verification runs), so without the memo the same (C_t, a, o) could
    receive different scores within one search. Only successful judgments
    are memoized (parse errors AND payload shape errors — non-dict payload,
    missing or non-numeric ``score`` — return 0.0 and stay retryable). The
    memo lives for the lifetime of the judge instance; keys embed the full
    query and context, so cross-sample hits are true content hits.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize judge with an LLM client.

        Args:
            llm_client: The LLM client used for judge inference.
        """
        self.llm = llm_client
        self._memo: dict[str, tuple[float, str]] = {}

    @staticmethod
    def _memo_key(kind: str, *parts: object) -> str | None:
        """Content hash of a judgment input; None disables memoization
        for inputs that cannot be canonically serialized."""
        try:
            payload = json.dumps([kind, *parts], sort_keys=True, default=str)
        except Exception:
            return None
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def pre_evaluate(
        self,
        query: str,
        context: dict,
        tool_card: dict,
        arg_draft: dict,
    ) -> tuple[float, str]:
        """Compute r_pre: predictive score before tool execution.

        Memoized on (query, context, tool_card, arg_draft) content.
        """
        key = self._memo_key("pre", query, context, tool_card, arg_draft)
        if key is not None and key in self._memo:
            return self._memo[key]
        messages = self._build_pre_eval_messages(query, context, tool_card, arg_draft)
        try:
            payload = self.llm.generate_json(messages)
            result = self._extract_score(payload)
        except Exception as exc:
            return 0.0, f"pre_eval_parse_error: {exc}"
        if key is not None:
            self._memo[key] = result
        return result

    def post_evaluate(
        self,
        query: str,
        context_before: dict,
        tool_card: dict,
        args_used: dict,
        tool_output: dict,
    ) -> tuple[float, str]:
        """Compute r_post: grounded utility score after tool execution.

        Memoized on (query, context_before, tool_card, args_used,
        tool_output) content.
        """
        key = self._memo_key(
            "post", query, context_before, tool_card, args_used, tool_output
        )
        if key is not None and key in self._memo:
            return self._memo[key]
        messages = self._build_post_eval_messages(
            query, context_before, tool_card, args_used, tool_output
        )
        try:
            payload = self.llm.generate_json(messages)
            result = self._extract_score(payload)
        except Exception as exc:
            return 0.0, f"post_eval_parse_error: {exc}"
        if key is not None:
            self._memo[key] = result
        return result

    def _build_pre_eval_messages(
        self,
        query: str,
        context: dict,
        tool_card: dict,
        arg_draft: dict,
    ) -> list[dict]:
        """Construct the message list for pre-evaluation judge call (Appendix B.7)."""
        context_str = context if isinstance(context, str) else json.dumps(
            context or {}, ensure_ascii=False, default=str
        )
        user_msg = build_pre_eval_user_message(query, context_str, tool_card, arg_draft)
        return [
            {"role": "system", "content": PRE_EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _build_post_eval_messages(
        self,
        query: str,
        context_before: dict,
        tool_card: dict,
        args_used: dict,
        tool_output: dict,
    ) -> list[dict]:
        """Construct the message list for post-evaluation judge call (Appendix B.8)."""
        context_str = context_before if isinstance(context_before, str) else json.dumps(
            context_before or {}, ensure_ascii=False, default=str
        )
        user_msg = build_post_eval_user_message(
            query, context_str, tool_card, args_used, tool_output
        )
        return [
            {"role": "system", "content": POST_EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _parse_judge_response(self, response: str) -> tuple[float, str]:
        """Parse JSON response from judge into (score, explanation)."""
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                raise ValueError(f"Could not parse judge response: {response[:200]}")
            payload = json.loads(m.group(0))
        return self._extract_score(payload)

    @staticmethod
    def _extract_score(payload: dict) -> tuple[float, str]:
        """Extract (score, explanation) from a judge payload.

        Raises ValueError on shape errors — non-dict payload, missing
        ``score`` key, non-numeric ``score`` — so pre/post_evaluate route
        them through their parse-error branch and never memoize them.
        """
        if not isinstance(payload, dict):
            raise ValueError(
                f"judge payload is not a dict: {type(payload).__name__}"
            )
        if "score" not in payload:
            raise ValueError("judge payload has no 'score' key")
        try:
            score = float(payload["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"judge 'score' is not numeric: {payload['score']!r}"
            ) from exc
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        explanation = str(payload.get("explanation", ""))
        return score, explanation
