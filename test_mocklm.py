"""
MockLLM Test Suite
~~~~~~~~~~~~~~~~~~
Unit tests for MockLLM API response objects, prompt wrapping, exception handling,
registry systems, and client interfaces.
"""

import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, ".")

from mocklm import MockLLM, MockCompletionResponse, MockLLMError
from mocklm.exceptions import (
    BrowserNotReadyError,
    BrowserCrashedError,
    ResponseTimeoutError,
    SelectorNotFoundError,
    ProviderNotFoundError,
)
from mocklm.prompt_wrapper import wrap_query
from mocklm.providers import get_provider, list_providers
from mocklm.response import create_response, create_error_response


class TestMockLLMResponse(unittest.TestCase):
    """Test MockCompletionResponse behavior and OpenAI compatibility."""

    def test_successful_response(self):
        """Test response generation with standard fields and OpenAI dictionary structure."""
        resp = create_response(
            text="Mocked output response text.",
            provider="gemini",
            model="gemini-web",
            elapsed=4.5,
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.text, "Mocked output response text.")
        self.assertEqual(resp.choices[0].message.content, "Mocked output response text.")
        self.assertEqual(resp.choices[0].message.role, "assistant")
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.model, "gemini-web")
        self.assertEqual(resp.provider, "gemini")
        self.assertEqual(resp.elapsed, 4.5)
        self.assertIsNone(resp.error)
        self.assertTrue(resp.id.startswith("mocklm-"))
        self.assertEqual(resp.object, "chat.completion")

        # Verify dictionary serialization matches OpenAI schema
        d = resp.to_dict()
        self.assertEqual(d["object"], "chat.completion")
        self.assertEqual(d["choices"][0]["message"]["content"], "Mocked output response text.")
        self.assertEqual(d["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(d["choices"][0]["finish_reason"], "stop")
        self.assertEqual(d["_mocklm"]["provider"], "gemini")
        self.assertEqual(d["_mocklm"]["success"], True)

    def test_error_response(self):
        """Test response generation in case of failures."""
        resp = create_error_response("Timeout connecting to browser", "gemini", elapsed=12.0)
        self.assertFalse(resp.success)
        self.assertEqual(resp.error, "Timeout connecting to browser")
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.choices[0].message.content, "")
        self.assertEqual(resp.choices[0].finish_reason, "error")
        self.assertEqual(resp.elapsed, 12.0)

        # Dictionary serialization for errors
        d = resp.to_dict()
        self.assertEqual(d["_mocklm"]["success"], False)
        self.assertEqual(d["_mocklm"]["error"], "Timeout connecting to browser")


class TestPromptWrapper(unittest.TestCase):
    """Test the stateless prompt wrapping utility."""

    def test_stateless_wrap(self):
        """Verify wrapping query without a system prompt."""
        query = "Show me the code."
        wrapped = wrap_query(query, stateless=True)
        self.assertIn("[IMPORTANT INSTRUCTION]", wrapped)
        self.assertIn("Show me the code.", wrapped)
        self.assertIn("Treat this message as a completely STANDALONE request.", wrapped)

    def test_stateless_with_system_prompt(self):
        """Verify wrapping with system context and user query."""
        query = "Hello!"
        sys_prompt = "You are a math tutor."
        wrapped = wrap_query(query, system_prompt=sys_prompt, stateless=True)
        self.assertIn("[IMPORTANT INSTRUCTION]", wrapped)
        self.assertIn("[SYSTEM CONTEXT]\nYou are a math tutor.", wrapped)
        self.assertIn("[USER REQUEST]\nHello!", wrapped)

    def test_conversational_no_wrap(self):
        """Verify query is returned verbatim if stateless=False."""
        query = "Just conversational."
        wrapped = wrap_query(query, stateless=False)
        self.assertEqual(wrapped, query)

    def test_conversational_with_system_prompt(self):
        """Verify system prompt is prepended when stateless=False."""
        query = "Hello!"
        sys_prompt = "Be helpful."
        wrapped = wrap_query(query, system_prompt=sys_prompt, stateless=False)
        self.assertEqual(wrapped, "Be helpful.\n\nHello!")


class TestExceptions(unittest.TestCase):
    """Test custom exceptions inherit correctly and print meaningful errors."""

    def test_exceptions_inheritance(self):
        self.assertTrue(issubclass(BrowserNotReadyError, MockLLMError))
        self.assertTrue(issubclass(BrowserCrashedError, MockLLMError))
        self.assertTrue(issubclass(ResponseTimeoutError, MockLLMError))
        self.assertTrue(issubclass(SelectorNotFoundError, MockLLMError))
        self.assertTrue(issubclass(ProviderNotFoundError, MockLLMError))

    def test_selector_not_found_message(self):
        err = SelectorNotFoundError("submit button", [".btn-submit", "#send"])
        self.assertIn("submit button", str(err))
        self.assertIn(".btn-submit", str(err))


class TestProviders(unittest.TestCase):
    """Test the provider registry and supported providers."""

    def test_available_providers(self):
        self.assertIn("gemini", list_providers())

    def test_get_valid_provider(self):
        prov = get_provider("gemini")
        self.assertEqual(prov.name, "gemini")
        self.assertEqual(prov.base_url, "https://gemini.google.com/app")

    def test_get_invalid_provider_raises(self):
        with self.assertRaises(ProviderNotFoundError):
            get_provider("nonexistent_provider")


class TestMockLLMClient(unittest.TestCase):
    """Test the public client initialization and options forwarding."""

    def test_client_init(self):
        """Verify default properties on MockLLM."""
        llm = MockLLM(provider="gemini", headless=True, stateless=False, system_prompt="Sys")
        self.assertEqual(llm.provider_name, "gemini")
        self.assertEqual(llm._headless, True)
        self.assertEqual(llm._stateless, False)
        self.assertEqual(llm._system_prompt, "Sys")
        self.assertIsNone(llm._cdp_url)
        self.assertFalse(llm.is_ready)

    def test_client_init_with_cdp(self):
        """Verify cdp_url is captured correctly."""
        llm = MockLLM(provider="gemini", cdp_url="http://localhost:9222")
        self.assertEqual(llm._cdp_url, "http://localhost:9222")

    @patch("mocklm.client.BrowserManager")
    def test_client_async_start(self, mock_browser_manager_cls):
        """Verify astart() instantiates BrowserManager and triggers launch."""
        # Create an async mock context manager or mock instance
        mock_instance = MagicMock()
        mock_instance.launch = AsyncMock()
        mock_browser_manager_cls.return_value = mock_instance

        llm = MockLLM(provider="gemini", cdp_url="http://localhost:9222")
        
        # We run the async start method using unittest-isolated run
        import asyncio
        asyncio.run(llm.astart())

        # Verify BrowserManager was initialized with correct arguments
        mock_browser_manager_cls.assert_called_once_with(
            provider=llm._provider,
            headless=llm._headless,
            cdp_url="http://localhost:9222"
        )
        # Verify launch was called
        mock_instance.launch.assert_called_once()
        self.assertTrue(llm._started)

    @patch("mocklm.client.BrowserManager")
    def test_client_chat_not_ready_raises(self, mock_browser_manager_cls):
        """Verify chat() raises BrowserNotReadyError if client hasn't started."""
        llm = MockLLM(provider="gemini")
        with self.assertRaises(BrowserNotReadyError):
            llm.chat("Query")


if __name__ == "__main__":
    unittest.main()
