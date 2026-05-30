"""
MockLLM Gemini Provider
~~~~~~~~~~~~~~~~~~~~~~~
Selectors and configuration for Google Gemini (gemini.google.com).

NOTE: These selectors are based on the Gemini web UI as of mid-2025.
      If the UI changes, update the selectors here.
"""

from __future__ import annotations

from .base import BaseProvider
from . import register_provider


@register_provider
class GeminiProvider(BaseProvider):
    """Google Gemini web interface provider."""

    name = "gemini"
    base_url = "https://gemini.google.com/app"
    model_name = "gemini-web"

    def get_input_selectors(self) -> list[str]:
        return [
            # Gemini uses a rich-text contenteditable div
            '.ql-editor[contenteditable="true"]',
            'div[contenteditable="true"][aria-label*="prompt" i]',
            'div[contenteditable="true"][role="textbox"]',
            'div.ql-editor',
            'div[contenteditable="true"]',
        ]

    def get_submit_selectors(self) -> list[str]:
        return [
            'button[aria-label*="Send" i]',
            'button[aria-label*="Submit" i]',
            'button[data-testid*="send" i]',
            '.send-button',
            'button.send-button',
        ]

    def get_response_selectors(self) -> list[str]:
        return [
            # Direct custom element tags (highly robust against class name changes)
            'message-content',
            '.markdown',
            '.model-response-text',
            '.response-content',
            'div[class*="message-content" i]',
            'div[class*="model-response" i]',
            '[data-message-author-role="model" i]',
            # Specific combinations
            'message-content:last-of-type .markdown',
            '[data-message-author-role="model"]:last-of-type .markdown',
            '.conversation-container .model-response:last-child .text-content',
        ]

    def get_loading_selectors(self) -> list[str]:
        return [
            'button[aria-label*="Stop" i]',
            '.loading-indicator',
            '.response-loading',
            'mat-progress-bar',
            '.thinking-indicator',
        ]

    def get_done_selectors(self) -> list[str]:
        return [
            'button[aria-label*="Copy" i]',
            'button[aria-label*="Share" i]',
            '.response-actions button',
            'button[aria-label*="Good response" i]',
            'button[aria-label*="Bad response" i]',
        ]

    def get_new_chat_url(self) -> str:
        return "https://gemini.google.com/app"

    def get_input_method(self) -> str:
        # Use fast fill method (instant typing)
        return "fill"

    def get_submit_method(self) -> str:
        return "enter"

    def needs_enter_key(self) -> bool:
        return True

    def clean_response(self, raw_text: str) -> str:
        text = raw_text.strip()
        # Remove common Gemini UI artifacts that sometimes get scraped
        artifacts = [
            "volume_up",  # TTS button text
            "content_copy",  # Copy button text
            "thumb_up",
            "thumb_down",
            "Edit",
        ]
        for artifact in artifacts:
            text = text.replace(artifact, "")
        return text.strip()

    def get_dismiss_selectors(self) -> list[str]:
        return [
            # Gemini sometimes shows welcome/onboarding dialogs
            'button[aria-label*="Close" i]',
            'button[aria-label*="Got it" i]',
            'button[aria-label*="Dismiss" i]',
        ]
