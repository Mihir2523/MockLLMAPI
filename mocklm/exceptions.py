"""
MockLLM Exceptions
~~~~~~~~~~~~~~~~~~
Custom exception hierarchy for clear, actionable error messages.
"""


class MockLLMError(Exception):
    """Base exception for all MockLLM errors."""
    pass


class BrowserNotReadyError(MockLLMError):
    """Raised when trying to use the browser before it's been started."""

    def __init__(self, message: str = "Browser is not started. Call `start()` or `astart()` first."):
        super().__init__(message)


class ProviderNotFoundError(MockLLMError):
    """Raised when an unsupported provider name is given."""

    def __init__(self, provider: str, available: list[str]):
        self.provider = provider
        self.available = available
        msg = (
            f"Provider '{provider}' is not supported. "
            f"Available providers: {', '.join(available)}"
        )
        super().__init__(msg)


class LoginRequiredError(MockLLMError):
    """Raised when the user doesn't appear to be logged in."""

    def __init__(self, provider: str):
        msg = (
            f"You don't appear to be logged into {provider}. "
            f"Please log in through your browser first, then try again."
        )
        super().__init__(msg)


class ResponseTimeoutError(MockLLMError):
    """Raised when the LLM response takes too long."""

    def __init__(self, timeout: float, provider: str):
        msg = (
            f"Response from {provider} timed out after {timeout:.0f}s. "
            f"The model may be overloaded — try again."
        )
        super().__init__(msg)


class SelectorNotFoundError(MockLLMError):
    """Raised when no matching DOM selector is found on the page."""

    def __init__(self, element_name: str, selectors: list[str]):
        msg = (
            f"Could not find '{element_name}' on the page. "
            f"Tried selectors: {selectors}. "
            f"The website UI may have changed — selectors may need updating."
        )
        super().__init__(msg)


class BrowserCrashedError(MockLLMError):
    """Raised when the browser process terminates unexpectedly."""

    def __init__(self):
        super().__init__("Browser process crashed or was closed unexpectedly.")
