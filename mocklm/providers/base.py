"""
MockLLM Base Provider
~~~~~~~~~~~~~~~~~~~~~
Abstract base class that all LLM website providers must implement.

Each provider defines the CSS selectors and URLs needed to interact
with a specific LLM website. The BrowserManager uses these to know
where to type, what to click, and how to detect response completion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """
    Abstract provider for an LLM web interface.

    Subclasses define selectors for interacting with a specific LLM website.
    Selectors are returned as LISTS in priority order — the browser manager
    will try each one until it finds a match, providing resilience against
    DOM changes.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    name: str = ""              # e.g. "gemini"
    base_url: str = ""          # e.g. "https://gemini.google.com/app"
    model_name: str = ""        # e.g. "gemini-web" (used in response objects)

    # ── Selectors (subclasses MUST implement) ─────────────────────────────

    @abstractmethod
    def get_input_selectors(self) -> list[str]:
        """
        CSS selectors for the text input / prompt box.
        Returns a list in priority order (first = most reliable).
        """
        ...

    @abstractmethod
    def get_submit_selectors(self) -> list[str]:
        """
        CSS selectors for the send / submit button.
        """
        ...

    @abstractmethod
    def get_response_selectors(self) -> list[str]:
        """
        CSS selectors for the latest response text container.
        Should target the LAST / most recent response on the page.
        """
        ...

    @abstractmethod
    def get_loading_selectors(self) -> list[str]:
        """
        CSS selectors for elements visible WHILE the model is generating.
        e.g. a "Stop generating" button, loading spinner, typing indicator.
        """
        ...

    @abstractmethod
    def get_done_selectors(self) -> list[str]:
        """
        CSS selectors for elements that appear AFTER generation is complete.
        e.g. "Copy", "Share", "Regenerate" buttons on the response.
        """
        ...

    @abstractmethod
    def get_new_chat_url(self) -> str:
        """
        URL to navigate to for starting a new/fresh chat.
        Usually the same as base_url.
        """
        ...

    # ── Optional hooks ────────────────────────────────────────────────────

    def get_input_method(self) -> str:
        """
        How to enter text into the input field.
        Returns "type" for keyboard typing (default) or "fill" for direct fill.
        Most providers need "type" to avoid bot detection.
        """
        return "type"

    def get_submit_method(self) -> str:
        """
        How to submit after typing.
        Returns "click" (click the send button) or "enter" (press Enter key).
        """
        return "enter"

    def needs_enter_key(self) -> bool:
        """Whether Enter key submits the prompt (vs needing button click)."""
        return True

    def clean_response(self, raw_text: str) -> str:
        """
        Optional post-processing to clean the raw response text.
        Override for provider-specific cleanup (strip artifacts, UI text, etc).
        """
        return raw_text.strip()

    def get_dismiss_selectors(self) -> list[str]:
        """
        Optional selectors for popups/modals to dismiss after page load.
        e.g. cookie banners, "what's new" dialogs, upgrade prompts.
        """
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' url='{self.base_url}'>"
