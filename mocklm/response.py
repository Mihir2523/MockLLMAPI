"""
MockLLM Response Objects
~~~~~~~~~~~~~~~~~~~~~~~~
OpenAI-compatible response format.

The response structure mirrors the OpenAI Chat Completion API exactly,
so users can swap MockLLM into any codebase that expects OpenAI responses
with zero code changes.

Example response dict:
    {
        "id": "mocklm-abc123",
        "object": "chat.completion",
        "created": 1717000000,
        "model": "gemini-web",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    """A single chat message, matching OpenAI's message object."""
    role: str = "assistant"
    content: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Choice:
    """A single completion choice, matching OpenAI's choice object."""
    index: int = 0
    message: Message = field(default_factory=Message)
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason,
        }


@dataclass
class Usage:
    """Token usage info. Always 0 for MockLLM (we can't count browser tokens)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class MockCompletionResponse:
    """
    OpenAI-compatible chat completion response.

    Access the response text the same way you would with OpenAI:
        response.choices[0].message.content

    Or use the convenience property:
        response.text

    Convert to dict for JSON serialization:
        response.to_dict()
    """
    id: str = field(default_factory=lambda: f"mocklm-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    # ── Extra fields (not in OpenAI spec, but useful) ─────────────────────
    provider: str = ""
    elapsed: float = 0.0
    success: bool = True
    error: Optional[str] = None

    # ── Convenience ───────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        """Shortcut to get the response text content."""
        if self.choices:
            return self.choices[0].message.content
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible dict (for JSON serialization)."""
        d: dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict() for c in self.choices],
            "usage": self.usage.to_dict(),
        }
        # Include extra fields under a namespace so they don't collide
        d["_mocklm"] = {
            "provider": self.provider,
            "elapsed": self.elapsed,
            "success": self.success,
            "error": self.error,
        }
        return d

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        preview = self.text[:80] + "..." if len(self.text) > 80 else self.text
        return f"MockCompletionResponse(model='{self.model}', text='{preview}')"


def create_response(
    text: str,
    provider: str,
    model: str = "",
    elapsed: float = 0.0,
    finish_reason: str = "stop",
    success: bool = True,
    error: Optional[str] = None,
) -> MockCompletionResponse:
    """
    Factory function to build a complete OpenAI-compatible response.

    Args:
        text: The LLM's response text.
        provider: Provider name ("gemini", "chatgpt", etc.).
        model: Model name string (e.g. "gemini-web").
        elapsed: Time taken in seconds.
        finish_reason: "stop" for normal, "length" for truncated, "error" for failures.
        success: Whether the response was captured successfully.
        error: Error message if failed.

    Returns:
        A fully populated MockCompletionResponse.
    """
    if not model:
        model = f"{provider}-web"

    return MockCompletionResponse(
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        provider=provider,
        elapsed=elapsed,
        success=success,
        error=error,
    )


def create_error_response(
    error_msg: str,
    provider: str,
    elapsed: float = 0.0,
) -> MockCompletionResponse:
    """Create a response representing a failed request."""
    return create_response(
        text="",
        provider=provider,
        elapsed=elapsed,
        finish_reason="error",
        success=False,
        error=error_msg,
    )
