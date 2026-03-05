"""Post-evaluation judge prompt templates (Appendix B.8).

The post-evaluation judge scores the grounded utility of an executed
tool call, based on task-consistency, correctness, relevance, and
constraint satisfaction.
"""

from __future__ import annotations

POST_EVAL_SYSTEM_PROMPT: str = (
    "Role. You are a strict tool-planning judge for a language-agent that "
    "solves user tasks by calling tools in sequence.\n\n"
    "Inputs. You are given:\n"
    "- the original user query and conversation context before the call;\n"
    "- the tool card;\n"
    "- the concrete arguments that were used;\n"
    "- the actual tool output.\n\n"
    "Output format. You must output a single JSON object with:\n"
    '- "score": a real number between 0.0 and 1.0 (inclusive) measuring the '
    "grounded utility of this executed tool call;\n"
    '- "explanation": a brief natural-language justification (2-4 sentences).\n\n'
    "Scoring guideline. Use a coarse scale in [0, 1]. Choose a value that "
    "roughly reflects how helpful this call was; you do not need to finely "
    "distinguish very small differences.\n\n"
    "When assigning the score, consider:\n"
    "- Task-consistency: does the output address the user's query or current sub-goal?\n"
    "- Correctness / plausibility: are there obvious errors or contradictions?\n"
    "- Relevance: is the output focused on what is needed now, rather than generic or noisy?\n"
    "- Constraint satisfaction: does it respect safety, formatting, and domain constraints?\n\n"
    "Important. You are judging only this tool call's incremental contribution "
    "from the previous context to the new context. Do not re-evaluate the entire plan."
)


def build_post_eval_user_message(
    query: str,
    context_before: str,
    tool_card: dict,
    args_used: dict,
    tool_output: dict,
) -> str:
    """Build the user message for post-evaluation judge (Appendix B.8 template).

    Constructs a structured prompt containing the query, pre-call context,
    tool card, arguments used, and actual tool output for the judge to evaluate.

    Args:
        query: The original user query.
        context_before: Dialogue context before this tool call.
        tool_card: Tool metadata dict.
        args_used: Actual arguments that were used for execution.
        tool_output: Raw output from tool execution.

    Returns:
        Formatted user message string for the judge.
    """
    raise NotImplementedError
