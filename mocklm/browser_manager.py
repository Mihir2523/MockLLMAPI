"""
MockLLM Browser Manager
~~~~~~~~~~~~~~~~~~~~~~~~
Playwright-based browser lifecycle and interaction engine.

Handles:
- Launching/closing a persistent Chromium browser
- Human-like typing and clicking
- Submitting prompts to LLM web UIs
- Detecting when streaming responses are complete
- Extracting response text from the DOM
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Optional

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeout,
)

from .config import (
    BROWSER_ARGS,
    BROWSER_DATA_BASE,
    DEFAULT_TIMEOUT,
    HEADLESS_DEFAULT,
    NAVIGATION_TIMEOUT,
    POST_SUBMIT_DELAY,
    PRE_CLICK_DELAY,
    SELECTOR_TIMEOUT,
    STREAM_IDLE_TIMEOUT,
    STREAM_POLL_INTERVAL,
    TYPING_DELAY_MAX,
    TYPING_DELAY_MIN,
    LOG_PREFIX,
)
from .exceptions import (
    BrowserCrashedError,
    BrowserNotReadyError,
    ResponseTimeoutError,
    SelectorNotFoundError,
)
from .providers.base import BaseProvider

logger = logging.getLogger("mocklm")


class BrowserManager:
    """
    Manages a persistent Playwright Chromium browser for interacting
    with LLM web interfaces.

    The browser uses a persistent profile directory so the user only
    needs to log in once — cookies and sessions are preserved between runs.
    """

    def __init__(
        self,
        provider: BaseProvider,
        headless: bool = HEADLESS_DEFAULT,
        cdp_url: Optional[str] = None,
    ):
        self.provider = provider
        self.headless = headless
        self.cdp_url = cdp_url

        self._playwright: Optional[Playwright] = None
        self._browser_obj = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._ready = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def launch(self) -> None:
        """
        Launch the browser with a persistent profile and navigate to
        the provider's URL.
        """
        if self.cdp_url:
            logger.info(f"{LOG_PREFIX} 🔌 Connecting to existing browser at {self.cdp_url}...")
            self._playwright = await async_playwright().start()
            self._browser_obj = await self._playwright.chromium.connect_over_cdp(self.cdp_url)
            
            # Use existing context or create one
            if self._browser_obj.contexts:
                self._context = self._browser_obj.contexts[0]
            else:
                self._context = await self._browser_obj.new_context(
                    viewport={"width": 1280, "height": 800},
                    ignore_https_errors=True,
                )
        else:
            logger.info(f"{LOG_PREFIX} 🚀 Launching browser for {self.provider.name}...")

            # Create persistent profile directory for this provider
            profile_dir = BROWSER_DATA_BASE / self.provider.name
            profile_dir.mkdir(parents=True, exist_ok=True)

            self._playwright = await async_playwright().start()

            # Use persistent context to reuse login sessions
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=self.headless,
                args=BROWSER_ARGS,
                viewport={"width": 1280, "height": 800},
                ignore_https_errors=True,
            )

        # Use the first page or create one
        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = await self._context.new_page()

        # Navigate to the LLM website
        logger.info(f"{LOG_PREFIX} 🌐 Navigating to {self.provider.base_url}...")
        try:
            await self._page.goto(
                self.provider.base_url,
                wait_until="domcontentloaded",
                timeout=NAVIGATION_TIMEOUT * 1000,
            )
        except PlaywrightTimeout:
            logger.warning(
                f"{LOG_PREFIX} ⚠️  Page load timed out, but continuing — "
                f"the page may still be usable."
            )

        # Wait a moment for dynamic content to settle
        await asyncio.sleep(2)

        # Try to dismiss any popups/modals
        await self._dismiss_popups()

        self._ready = True
        logger.info(f"{LOG_PREFIX} ✅ Browser ready! Provider: {self.provider.name}")
        if not self.cdp_url:
            logger.info(
                f"{LOG_PREFIX} 📋 Make sure you are logged into {self.provider.name}. "
                f"If not, please log in through the browser window now."
            )
        else:
            logger.info(f"{LOG_PREFIX} 📋 Reusing your active system browser session via CDP.")

    async def shutdown(self) -> None:
        """Gracefully close the browser."""
        self._ready = False
        try:
            if self.cdp_url:
                if self._page:
                    await self._page.close()
                if self._browser_obj:
                    await self._browser_obj.close()
            else:
                if self._context:
                    await self._context.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Browser cleanup error (safe to ignore): {e}")
        finally:
            self._context = None
            self._page = None
            self._browser_obj = None
            self._playwright = None
            logger.info(f"{LOG_PREFIX} 🛑 Browser closed.")

    @property
    def is_ready(self) -> bool:
        return self._ready and self._page is not None

    # ── Core Interaction ──────────────────────────────────────────────────

    async def send_and_receive(self, prompt: str) -> str:
        """
        Type a prompt into the LLM input, submit it, wait for the
        response to complete, and return the response text.

        Args:
            prompt: The full prompt text (already wrapped if needed).

        Returns:
            The LLM's response text (cleaned by the provider).

        Raises:
            BrowserNotReadyError: If browser isn't started.
            SelectorNotFoundError: If UI elements can't be found.
            ResponseTimeoutError: If response takes too long.
        """
        if not self.is_ready:
            raise BrowserNotReadyError()

        page = self._page
        assert page is not None  # for type checker

        # Step 1: Find and clear the input field
        input_el = await self._find_element(
            self.provider.get_input_selectors(), "input field"
        )
        await input_el.click()
        await input_el.focus()
        await asyncio.sleep(0.3)

        # Clear any existing text in the input
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        await asyncio.sleep(0.2)

        # Step 2: Input the prompt
        input_method = self.provider.get_input_method()
        if input_method == "fill":
            logger.info(f"{LOG_PREFIX} 📥 Filling prompt instantly ({len(prompt)} chars)...")
            await input_el.fill(prompt)
        else:
            logger.info(f"{LOG_PREFIX} ⌨️  Typing prompt ({len(prompt)} chars)...")
            await self._human_type(prompt)

        # Step 3: Submit
        await asyncio.sleep(PRE_CLICK_DELAY)
        if self.provider.needs_enter_key():
            await page.keyboard.press("Enter")
        else:
            submit_el = await self._find_element(
                self.provider.get_submit_selectors(), "submit button"
            )
            await submit_el.click()

        logger.info(f"{LOG_PREFIX} 📤 Prompt submitted. Waiting for response...")

        # Step 4: Wait for response to complete
        await asyncio.sleep(POST_SUBMIT_DELAY)
        await self._wait_for_response()

        # Step 5: Extract the response text
        raw_text = await self._extract_response()
        cleaned = self.provider.clean_response(raw_text)

        return cleaned

    async def navigate_new_chat(self) -> None:
        """Navigate to a new/fresh chat page."""
        if not self.is_ready:
            raise BrowserNotReadyError()

        url = self.provider.get_new_chat_url()
        logger.info(f"{LOG_PREFIX} 🔄 Starting new chat...")
        await self._page.goto(url, wait_until="domcontentloaded",
                              timeout=NAVIGATION_TIMEOUT * 1000)
        await asyncio.sleep(2)
        await self._dismiss_popups()

    # ── Private Helpers ───────────────────────────────────────────────────

    async def _find_element(self, selectors: list[str], element_name: str):
        """
        Try each selector in order and return the first visible match.
        """
        page = self._page
        assert page is not None

        for selector in selectors:
            try:
                locator = page.locator(selector).first
                await locator.wait_for(
                    state="visible",
                    timeout=SELECTOR_TIMEOUT * 1000 // max(len(selectors), 1),
                )
                return locator
            except (PlaywrightTimeout, Exception):
                continue

        raise SelectorNotFoundError(element_name, selectors)

    async def _human_type(self, text: str) -> None:
        """Type text character by character with random delays."""
        page = self._page
        assert page is not None

        for char in text:
            if char == "\n":
                await page.keyboard.press("Shift+Enter")
            else:
                await page.keyboard.type(char)
            delay = random.randint(TYPING_DELAY_MIN, TYPING_DELAY_MAX) / 1000.0
            await asyncio.sleep(delay)

    async def _wait_for_response(self) -> None:
        """
        Wait for the LLM response to finish streaming.

        Strategy:
        1. Wait for a loading indicator to appear (optional — it may appear
           too fast to catch).
        2. Then wait for it to disappear OR for a completion indicator to appear.
        3. Fallback: monitor the DOM text — when it stops growing for
           STREAM_IDLE_TIMEOUT seconds, assume done.
        4. Hard timeout: DEFAULT_TIMEOUT seconds.
        """
        page = self._page
        assert page is not None

        start_time = time.monotonic()
        deadline = start_time + DEFAULT_TIMEOUT
        last_text = ""
        last_change_time = time.monotonic()

        # Phase 1: Briefly try to detect a loading indicator
        loading_selectors = self.provider.get_loading_selectors()
        try:
            for sel in loading_selectors:
                try:
                    await page.locator(sel).first.wait_for(
                        state="visible", timeout=3000
                    )
                    logger.debug(f"{LOG_PREFIX} Loading indicator found: {sel}")
                    break
                except (PlaywrightTimeout, Exception):
                    continue
        except Exception:
            pass

        # Phase 2: Poll until response stabilizes
        while time.monotonic() < deadline:
            # Check if a "done" indicator appeared
            done_selectors = self.provider.get_done_selectors()
            for sel in done_selectors:
                try:
                    visible = await page.locator(sel).first.is_visible()
                    if visible:
                        logger.debug(f"{LOG_PREFIX} Completion indicator found: {sel}")
                        await asyncio.sleep(0.5)  # Small grace period
                        return
                except Exception:
                    continue

            # Check if loading indicator disappeared (meaning done)
            all_loading_gone = True
            for sel in loading_selectors:
                try:
                    visible = await page.locator(sel).first.is_visible()
                    if visible:
                        all_loading_gone = False
                        break
                except Exception:
                    continue

            # Fallback: monitor DOM text stability
            current_text = await self._get_latest_response_text()
            if current_text != last_text:
                last_text = current_text
                last_change_time = time.monotonic()
            elif (
                current_text
                and time.monotonic() - last_change_time >= STREAM_IDLE_TIMEOUT
                and all_loading_gone
            ):
                # Text stopped changing and no loading indicators visible
                logger.debug(f"{LOG_PREFIX} Response stabilized (idle timeout)")
                return

            await asyncio.sleep(STREAM_POLL_INTERVAL)

        # If we got here, we timed out
        elapsed = time.monotonic() - start_time
        raise ResponseTimeoutError(elapsed, self.provider.name)

    async def _get_latest_response_text(self) -> str:
        """Try to extract the current (possibly incomplete) response text."""
        page = self._page
        assert page is not None

        for selector in self.provider.get_response_selectors():
            try:
                elements = page.locator(selector)
                count = await elements.count()
                if count > 0:
                    # Get the LAST matching element (latest response)
                    last = elements.nth(count - 1)
                    text = await last.inner_text(timeout=2000)
                    return text
            except Exception:
                continue

        return ""

    async def _extract_response(self) -> str:
        """Extract the final response text after streaming is complete."""
        text = await self._get_latest_response_text()
        if not text:
            logger.warning(
                f"{LOG_PREFIX} ⚠️  Could not extract response text. "
                f"The page structure may have changed."
            )
        return text

    async def _dismiss_popups(self) -> None:
        """Try to dismiss any modals/popups that appear after navigation."""
        page = self._page
        assert page is not None

        dismiss_selectors = self.provider.get_dismiss_selectors()
        for selector in dismiss_selectors:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible():
                    await btn.click()
                    logger.debug(f"{LOG_PREFIX} Dismissed popup: {selector}")
                    await asyncio.sleep(0.5)
            except Exception:
                continue
