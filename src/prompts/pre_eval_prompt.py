"""Pre-evaluation judge prompt templates (Appendix B.7).

The pre-evaluation judge scores how promising a candidate tool call is
BEFORE execution, based on schema compatibility, relevance, and context.
"""

from __future__ import annotations

PRE_EVAL_SYSTEM_PROMPT: str = (
    "Role. You are a strict tool-planning judge for a language-agent that "
    "solves user tasks by calling tools in sequence.\n\n"
    "Inputs. You are given:\n"
    "- the original user query and current conversation context;\n"
    "- a tool card (name, description, I/O schema, examples);\n"
    "- a concrete argument draft that is syntactically valid for the tool.\n\n"
    "Output format. You must output a single JSON object with:\n"
    '- "score": a real number between 0.0 and 1.0 (inclusive) measuring how '
    "promising this tool call is before running it;\n"
    '- "explanation": a brief natural-language justification (2-4 sentences).\n\n'
    "Scoring guideline. Use a coarse scale in [0, 1]. There is no need to "
    "finely distinguish every small difference; choose a value that roughly "
    "reflects your judgment of usefulness.\n\n"
    "What to penalize. Give low scores to candidate tool calls that:\n"
    "- mismatch the required modality or domain;\n"
    "- ignore key constraints or required fields in the schema;\n"
    "- duplicate a previous call with effectively identical arguments and no "
    "clear new benefit;\n"
    "- are speculative when a more direct or specific tool is available.\n\n"
    "Important. Do not simulate the tool output; you are judging only the "
    "promised usefulness of this tool call as the next action."
)


def build_pre_eval_user_message(
    query: str,
    context: str,
    tool_card: dict,
    arg_draft: dict,
) -> str:
    """Build the user message for pre-evaluation judge (Appendix B.7 template).

    Constructs a structured prompt containing the query, context, tool card
    metadata, and proposed argument draft for the judge to evaluate.

    Args:
        query: The original user query.
        context: Current dialogue/planning context as a string.
        tool_card: Tool metadata dict with keys: name, description,
                   input_schema, output_schema, examples.
        arg_draft: Proposed argument draft dict for the tool call.

    Returns:
        Formatted user message string for the judge.
    """
    raise NotImplementedError
