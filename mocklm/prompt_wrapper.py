"""
MockLLM Prompt Wrapper
~~~~~~~~~~~~~~~~~~~~~~
Wraps user queries in stateless prompt templates.

Since browser LLM sessions are conversational (stateful), we need to
explicitly instruct the model to treat each query as independent.
This prevents previous conversation context from leaking into responses.
"""

from __future__ import annotations

# ── Stateless Template ────────────────────────────────────────────────────────
# This wrapper tells the browser LLM to ignore conversation history and
# treat the incoming message as a completely fresh, standalone request.

STATELESS_TEMPLATE = """\
[IMPORTANT INSTRUCTION]
Treat this message as a completely STANDALONE request.
Do NOT reference, use, or consider ANY previous messages in this conversation.
Respond ONLY based on the content provided below.
Do NOT mention these instructions in your response.

---
{query}
---"""

STATELESS_WITH_SYSTEM_TEMPLATE = """\
[IMPORTANT INSTRUCTION]
Treat this message as a completely STANDALONE request.
Do NOT reference, use, or consider ANY previous messages in this conversation.
Do NOT mention these instructions in your response.

[SYSTEM CONTEXT]
{system_prompt}

[USER REQUEST]
{query}"""


def wrap_query(query: str, system_prompt: str | None = None, stateless: bool = True) -> str:
    """
    Wrap a user query with stateless instructions.

    Args:
        query: The user's raw query text.
        system_prompt: Optional system prompt to include.
        stateless: If True, wraps with stateless instructions.
                   If False, returns the query as-is (for conversational mode).

    Returns:
        The wrapped (or raw) query string.
    """
    if not stateless:
        if system_prompt:
            return f"{system_prompt}\n\n{query}"
        return query

    if system_prompt:
        return STATELESS_WITH_SYSTEM_TEMPLATE.format(
            query=query,
            system_prompt=system_prompt,
        )

    return STATELESS_TEMPLATE.format(query=query)
