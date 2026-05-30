"""
MockLLM Client
~~~~~~~~~~~~~~
The main user-facing class. Provides both sync and async APIs.

Usage (sync):
    from mocklm import MockLLM

    llm = MockLLM(provider="gemini")
    llm.start()
    response = llm.chat("What is Python?")
    print(response.choices[0].message.content)
    llm.close()

Usage (async):
    import asyncio
    from mocklm import MockLLM

    async def main():
        llm = MockLLM(provider="gemini")
        await llm.astart()
        response = await llm.achat("What is Python?")
        print(response.choices[0].message.content)
        await llm.aclose()

    asyncio.run(main())

Usage (context manager):
    with MockLLM(provider="gemini") as llm:
        response = llm.chat("Hello!")
        print(response.text)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .browser_manager import BrowserManager
from .config import HEADLESS_DEFAULT, LOG_PREFIX
from .exceptions import BrowserNotReadyError, MockLLMError
from .prompt_wrapper import wrap_query
from .providers import get_provider, list_providers
from .providers.base import BaseProvider
from .response import MockCompletionResponse, create_error_response, create_response

logger = logging.getLogger("mocklm")


def _setup_logging() -> None:
    """Configure logging with a clean format if not already configured."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class MockLLM:
    """
    Mock LLM API client that uses browser automation to interact with
    free LLM web interfaces.

    Provides an OpenAI-compatible response format so you can swap this
    into any codebase that expects OpenAI's API with zero code changes.

    Args:
        provider: Name of the LLM provider ("gemini", "chatgpt", "grok", etc.).
        headless: Whether to run the browser in headless mode.
                  Default False (visible) so the user can verify login status.
        stateless: Whether to wrap prompts with stateless instructions.
                   Default True — each query is treated as independent.
        system_prompt: Optional system prompt to include with every query.

    Example:
        >>> llm = MockLLM(provider="gemini")
        >>> llm.start()
        >>> resp = llm.chat("What is 2+2?")
        >>> print(resp.choices[0].message.content)
        "4"
        >>> llm.close()
    """

    def __init__(
        self,
        provider: str = "gemini",
        headless: bool = HEADLESS_DEFAULT,
        stateless: bool = True,
        system_prompt: Optional[str] = None,
        cdp_url: Optional[str] = None,
    ):
        _setup_logging()

        self._provider_name = provider.lower().strip()
        self._provider: BaseProvider = get_provider(self._provider_name)
        self._headless = headless
        self._stateless = stateless
        self._system_prompt = system_prompt
        self._cdp_url = cdp_url
        self._browser: Optional[BrowserManager] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = False

        logger.info(f"{LOG_PREFIX} Initialized with provider: {self._provider_name}")
        if self._cdp_url:
            logger.info(f"{LOG_PREFIX} Using CDP connection to existing browser: {self._cdp_url}")
        else:
            logger.info(
                f"{LOG_PREFIX} ⚠️  Make sure you are logged into "
                f"{self._provider_name} in your browser before calling start()."
            )

    # ══════════════════════════════════════════════════════════════════════
    #  ASYNC API
    # ══════════════════════════════════════════════════════════════════════

    async def astart(self) -> None:
        """
        (Async) Launch the browser and navigate to the LLM provider page.
        Call this once before using achat().
        """
        self._browser = BrowserManager(
            provider=self._provider,
            headless=self._headless,
            cdp_url=self._cdp_url,
        )
        await self._browser.launch()
        self._started = True

    async def achat(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        stateless: Optional[bool] = None,
    ) -> MockCompletionResponse:
        """
        (Async) Send a query to the LLM and return an OpenAI-compatible response.

        Args:
            query: The user's prompt/question.
            system_prompt: Override the instance-level system prompt for this call.
            stateless: Override the instance-level stateless setting for this call.

        Returns:
            MockCompletionResponse in OpenAI's chat completion format.
        """
        if not self._started or not self._browser or not self._browser.is_ready:
            raise BrowserNotReadyError()

        # Determine effective settings
        use_stateless = stateless if stateless is not None else self._stateless
        use_system = system_prompt if system_prompt is not None else self._system_prompt

        # Wrap the query
        wrapped = wrap_query(query, system_prompt=use_system, stateless=use_stateless)

        start_time = time.monotonic()

        try:
            # Send to browser and get response
            response_text = await self._browser.send_and_receive(wrapped)
            elapsed = time.monotonic() - start_time

            logger.info(
                f"{LOG_PREFIX} ✅ Response received ({elapsed:.1f}s, "
                f"{len(response_text)} chars)"
            )

            return create_response(
                text=response_text,
                provider=self._provider_name,
                model=self._provider.model_name,
                elapsed=elapsed,
            )

        except MockLLMError:
            raise
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"{LOG_PREFIX} ❌ Error: {e}")
            return create_error_response(
                error_msg=str(e),
                provider=self._provider_name,
                elapsed=elapsed,
            )

    async def anew_chat(self) -> None:
        """(Async) Start a fresh conversation (navigates to new chat URL)."""
        if not self._started or not self._browser:
            raise BrowserNotReadyError()
        await self._browser.navigate_new_chat()

    async def aclose(self) -> None:
        """(Async) Gracefully close the browser."""
        if self._browser:
            await self._browser.shutdown()
        self._started = False

    # ══════════════════════════════════════════════════════════════════════
    #  SYNC API (wrappers around async methods)
    # ══════════════════════════════════════════════════════════════════════

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for sync operations."""
        try:
            loop = asyncio.get_running_loop()
            # If a loop is already running, we can't use run_until_complete.
            # This happens when called from within async code.
            raise RuntimeError(
                "Cannot use sync API inside an already running async loop. "
                "Use the async API (astart, achat, aclose) instead."
            )
        except RuntimeError as e:
            if "no current event loop" in str(e).lower() or "no running event loop" in str(e).lower():
                if self._loop is None or self._loop.is_closed():
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
                return self._loop
            raise

    def start(self) -> None:
        """
        (Sync) Launch the browser and navigate to the LLM provider page.
        Call this once before using chat().
        """
        loop = self._get_or_create_loop()
        loop.run_until_complete(self.astart())

    def chat(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        stateless: Optional[bool] = None,
    ) -> MockCompletionResponse:
        """
        (Sync) Send a query to the LLM and return an OpenAI-compatible response.

        Args:
            query: The user's prompt/question.
            system_prompt: Override the instance-level system prompt for this call.
            stateless: Override the instance-level stateless setting for this call.

        Returns:
            MockCompletionResponse in OpenAI's chat completion format.
            Access the text via: response.choices[0].message.content
            Or shortcut:        response.text
        """
        loop = self._get_or_create_loop()
        return loop.run_until_complete(
            self.achat(query, system_prompt=system_prompt, stateless=stateless)
        )

    def new_chat(self) -> None:
        """(Sync) Start a fresh conversation."""
        loop = self._get_or_create_loop()
        loop.run_until_complete(self.anew_chat())

    def close(self) -> None:
        """(Sync) Gracefully close the browser."""
        loop = self._get_or_create_loop()
        loop.run_until_complete(self.aclose())
        if self._loop and not self._loop.is_closed():
            self._loop.close()
            self._loop = None

    # ══════════════════════════════════════════════════════════════════════
    #  CONTEXT MANAGER
    # ══════════════════════════════════════════════════════════════════════

    def __enter__(self) -> MockLLM:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    async def __aenter__(self) -> MockLLM:
        await self.astart()
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    # ══════════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ══════════════════════════════════════════════════════════════════════

    @property
    def is_ready(self) -> bool:
        """True if the browser is started and ready to accept queries."""
        return self._started and self._browser is not None and self._browser.is_ready

    @property
    def provider_name(self) -> str:
        """The name of the active provider."""
        return self._provider_name

    @staticmethod
    def available_providers() -> list[str]:
        """Return a list of all supported provider names."""
        return list_providers()

    def __repr__(self) -> str:
        status = "ready" if self.is_ready else "not started"
        return f"MockLLM(provider='{self._provider_name}', status='{status}')"
