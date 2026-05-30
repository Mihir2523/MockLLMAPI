"""
MockLLM — Browser-Based Free LLM API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Turn free LLM web interfaces (Gemini, ChatGPT, Grok, etc.) into a
Python API — no API keys needed.

Quickstart:
    from mocklm import MockLLM

    llm = MockLLM(provider="gemini")
    llm.start()

    response = llm.chat("What is quantum computing?")
    print(response.choices[0].message.content)

    llm.close()
"""

from .client import MockLLM
from .response import MockCompletionResponse, Message, Choice, Usage
from .exceptions import (
    MockLLMError,
    BrowserNotReadyError,
    ProviderNotFoundError,
    LoginRequiredError,
    ResponseTimeoutError,
    SelectorNotFoundError,
    BrowserCrashedError,
)

__all__ = [
    # Main class
    "MockLLM",
    # Response types
    "MockCompletionResponse",
    "Message",
    "Choice",
    "Usage",
    # Exceptions
    "MockLLMError",
    "BrowserNotReadyError",
    "ProviderNotFoundError",
    "LoginRequiredError",
    "ResponseTimeoutError",
    "SelectorNotFoundError",
    "BrowserCrashedError",
]

__version__ = "0.1.0"
