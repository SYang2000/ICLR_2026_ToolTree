"""Answer predictor prompt template.

The answer predictor takes the best MCTS trajectory (tool chain with
outputs) and generates the final answer to the user's query.
"""

from __future__ import annotations

ANSWER_PREDICTOR_SYSTEM_PROMPT: str = (
    "You are an expert assistant. You have been given the result of executing "
    "a sequence of tool calls to answer a user's question. Based on the tool "
    "outputs provided, synthesize a clear, accurate, and concise final answer. "
    "Only use information from the tool outputs; do not hallucinate."
)


def build_answer_message(query: str, trajectory: list[dict]) -> str:
    """Build the user message for the answer predictor.

    Formats the best MCTS trajectory into a structured prompt for the
    LLM to synthesize a final answer.

    Args:
        query: The original user query.
        trajectory: List of dicts from the best MCTS path, each containing
                    "action" (tool name), "args" (arguments used), and
                    "output" (tool execution result).

    Returns:
        Formatted user message string for the answer predictor.
    """
    raise NotImplementedError
