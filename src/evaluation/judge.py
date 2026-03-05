"""LLM Judge for dual pre- and post-evaluation scoring.

Uses prompt templates from Appendix B.7 (pre-evaluation) and
Appendix B.8 (post-evaluation) to score tool calls on [0, 1].
"""

from __future__ import annotations

from src.llm.client import LLMClient


class LLMJudge:
    """LLM-based judge for pre-evaluation and post-evaluation scoring.

    Pre-evaluation (r_pre): Estimates the utility of a tool call BEFORE
    execution, based on schema compatibility, relevance, and context fit.

    Post-evaluation (r_post): Assesses the actual contribution of a tool
    call AFTER execution, based on task-consistency, correctness, and
    constraint satisfaction.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize judge with an LLM client.

        Args:
            llm_client: The LLM client used for judge inference.
        """
        raise NotImplementedError

    def pre_evaluate(
        self,
        query: str,
        context: dict,
        tool_card: dict,
        arg_draft: dict,
    ) -> tuple[float, str]:
        """Compute r_pre: predictive score before tool execution.

        Evaluates how promising a candidate tool call is based on:
        - Schema and modality compatibility
        - Relevance to the current query and context
        - Avoidance of duplicate or speculative calls

        Args:
            query: Original user query.
            context: Current dialogue/planning context.
            tool_card: Tool metadata (name, description, I/O schema, examples).
            arg_draft: Proposed argument draft for the tool call.

        Returns:
            Tuple of (score, explanation) where score is in [0, 1].
        """
        raise NotImplementedError

    def post_evaluate(
        self,
        query: str,
        context_before: dict,
        tool_card: dict,
        args_used: dict,
        tool_output: dict,
    ) -> tuple[float, str]:
        """Compute r_post: grounded utility score after tool execution.

        Evaluates the actual contribution of the executed tool call based on:
        - Task-consistency: does the output address the query or sub-goal?
        - Correctness/plausibility: are there obvious errors?
        - Relevance: is the output focused on what is needed now?
        - Constraint satisfaction: does it respect safety and formatting?

        Args:
            query: Original user query.
            context_before: Dialogue context before this tool call.
            tool_card: Tool metadata (name, description, I/O schema, examples).
            args_used: Actual arguments that were used.
            tool_output: Raw output from tool execution.

        Returns:
            Tuple of (score, explanation) where score is in [0, 1].
        """
        raise NotImplementedError

    def _build_pre_eval_messages(
        self,
        query: str,
        context: dict,
        tool_card: dict,
        arg_draft: dict,
    ) -> list[dict]:
        """Construct the message list for pre-evaluation judge call.

        Follows the prompt template from Appendix B.7.

        Args:
            query: User query.
            context: Current context.
            tool_card: Tool metadata.
            arg_draft: Proposed arguments.

        Returns:
            List of message dicts with "role" and "content" keys.
        """
        raise NotImplementedError

    def _build_post_eval_messages(
        self,
        query: str,
        context_before: dict,
        tool_card: dict,
        args_used: dict,
        tool_output: dict,
    ) -> list[dict]:
        """Construct the message list for post-evaluation judge call.

        Follows the prompt template from Appendix B.8.

        Args:
            query: User query.
            context_before: Context before the call.
            tool_card: Tool metadata.
            args_used: Arguments used.
            tool_output: Raw tool output.

        Returns:
            List of message dicts with "role" and "content" keys.
        """
        raise NotImplementedError

    def _parse_judge_response(self, response: str) -> tuple[float, str]:
        """Parse JSON response from judge into (score, explanation).

        Expected format: {"score": <float 0-1>, "explanation": "<string>"}

        Args:
            response: Raw text response from the LLM judge.

        Returns:
            Tuple of (score, explanation).

        Raises:
            ValueError: If the response cannot be parsed.
        """
        raise NotImplementedError
