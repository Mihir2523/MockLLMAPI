# 🧪 MockLLM — Browser-Based Free LLM API

> Turn free LLM web interfaces into a Python API. No API keys. No payments. Just your browser.

MockLLM automates your browser in the background to send prompts to LLM websites (Gemini, ChatGPT, Grok, etc.) and returns responses in **OpenAI-compatible format** — so you can drop it into any project that uses the OpenAI SDK with zero code changes.

---

## ⚠️ Disclaimer

This module is for **demo and educational purposes only**. Automating interactions with LLM web interfaces may violate the Terms of Service of some providers. Use responsibly.

---

## 🚀 Installation

```bash
# 1. Install the dependency
pip install playwright

# 2. Install browser binaries (one-time)
playwright install chromium
```

---

## 📋 Prerequisites

**You must be logged into the LLM provider in your browser before using MockLLM.**

MockLLM uses a persistent browser profile, so you only need to log in once:

1. Run `llm.start()` — a browser window will open
2. Log into your LLM account (Gemini, ChatGPT, etc.)
3. That's it! Future runs will reuse your login session

---

## 🎯 Quick Start

### Sync API

```python
from mocklm import MockLLM

# Initialize with your preferred provider
llm = MockLLM(provider="gemini")
llm.start()  # Opens browser (log in if first time)

# Send a query — returns OpenAI-compatible response
response = llm.chat("Explain quantum computing in 3 sentences")

# Access response just like OpenAI's API
print(response.choices[0].message.content)

# Or use the shortcut
print(response.text)

# Clean up
llm.close()
```

### Async API

```python
import asyncio
from mocklm import MockLLM

async def main():
    llm = MockLLM(provider="gemini")
    await llm.astart()

    response = await llm.achat("What is machine learning?")
    print(response.choices[0].message.content)

    await llm.aclose()

asyncio.run(main())
```

### Context Manager

```python
from mocklm import MockLLM

with MockLLM(provider="gemini") as llm:
    response = llm.chat("What is 2+2?")
    print(response.text)
# Browser closes automatically
```

---

## 🔄 OpenAI-Compatible Response Format

MockLLM returns responses in the exact same format as OpenAI's Chat Completion API:

```python
response = llm.chat("Hello!")

# The response object mirrors OpenAI's format:
response.id                          # "mocklm-abc123def456"
response.object                      # "chat.completion"
response.model                       # "gemini-web"
response.choices[0].message.role     # "assistant"
response.choices[0].message.content  # "Hello! How can I help you?"
response.choices[0].finish_reason    # "stop"
response.usage.prompt_tokens         # 0 (unavailable from browser)
response.usage.completion_tokens     # 0

# Extra MockLLM fields:
response.provider                    # "gemini"
response.elapsed                     # 3.2 (seconds)
response.success                     # True
response.text                        # Shortcut for choices[0].message.content

# Serialize to dict (for JSON)
response.to_dict()
```

---

## ⚙️ Configuration

```python
llm = MockLLM(
    provider="gemini",       # LLM provider name
    headless=False,          # False = visible browser (recommended)
    stateless=True,          # Wrap prompts to prevent context leakage
    system_prompt="You are a helpful coding assistant.",  # Optional
)
```

### Per-call overrides

```python
# Override system prompt for a single call
response = llm.chat(
    "Write a haiku",
    system_prompt="You are a poet.",
    stateless=True,
)
```

---

## 🔌 Supported Providers

| Provider | Name | Status |
|----------|------|--------|
| Google Gemini | `"gemini"` | ✅ Available |
| ChatGPT | `"chatgpt"` | 🔜 Coming soon |
| Grok | `"grok"` | 🔜 Coming soon |
| DeepSeek | `"deepseek"` | 🔜 Coming soon |

Check available providers:
```python
print(MockLLM.available_providers())  # ["gemini"]
```

---

## 🛠️ Adding a New Provider

Create a new file in `providers/` and implement the `BaseProvider` interface:

```python
# providers/chatgpt.py
from .base import BaseProvider
from . import register_provider

@register_provider
class ChatGPTProvider(BaseProvider):
    name = "chatgpt"
    base_url = "https://chatgpt.com"
    model_name = "chatgpt-web"

    def get_input_selectors(self) -> list[str]:
        return ['textarea[data-id="prompt-textarea"]', 'textarea']

    def get_submit_selectors(self) -> list[str]:
        return ['button[data-testid="send-button"]']

    # ... implement remaining abstract methods
```

Then add the import in `providers/__init__.py`:
```python
from . import chatgpt
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Browser is not started" | Call `llm.start()` before `llm.chat()` |
| "Provider not found" | Check `MockLLM.available_providers()` |
| "Selector not found" | The website UI may have changed — update provider selectors |
| Response timeout | The model may be slow — increase timeout or try again |
| Login not persisted | Make sure you're using the same user data directory |

---

## 📁 Project Structure

```
mocklm/
├── __init__.py            # Public exports
├── client.py              # MockLLM class (sync + async API)
├── browser_manager.py     # Playwright browser lifecycle
├── prompt_wrapper.py      # Stateless prompt wrapping
├── response.py            # OpenAI-compatible response objects
├── config.py              # Configuration constants
├── exceptions.py          # Custom exceptions
├── providers/
│   ├── __init__.py        # Provider registry
│   ├── base.py            # Abstract base provider
│   └── gemini.py          # Google Gemini provider
├── README.md
└── requirements.txt
```
