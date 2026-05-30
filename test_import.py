"""Quick import test for mocklm module."""
import sys
sys.path.insert(0, ".")

from mocklm import MockLLM, MockCompletionResponse, MockLLMError
from mocklm.response import create_response, create_error_response
from mocklm.prompt_wrapper import wrap_query
from mocklm.providers import list_providers, get_provider

print("=" * 50)
print("MockLLM Import Test")
print("=" * 50)

# 1. Basic imports
print(f"\n[OK] MockLLM class:        {MockLLM}")
print(f"[OK] Response class:       {MockCompletionResponse}")
print(f"[OK] Version:              {__import__('mocklm').__version__}")

# 2. Provider registry
providers = list_providers()
print(f"\n[OK] Available providers:  {providers}")
gemini = get_provider("gemini")
print(f"[OK] Gemini provider:      {gemini}")
print(f"     URL:                  {gemini.base_url}")
print(f"     Model:                {gemini.model_name}")

# 3. Response format (OpenAI-compatible)
resp = create_response(
    text="Hello! This is a test response.",
    provider="gemini",
    model="gemini-web",
    elapsed=2.5,
)
print(f"\n[OK] Response object:")
print(f"     .id:                  {resp.id}")
print(f"     .object:              {resp.object}")
print(f"     .model:               {resp.model}")
print(f"     .choices[0].message:  {resp.choices[0].message.to_dict()}")
print(f"     .text (shortcut):     {resp.text}")
print(f"     .to_dict() keys:      {list(resp.to_dict().keys())}")

# 4. Prompt wrapping
wrapped = wrap_query("What is Python?")
print(f"\n[OK] Prompt wrapping works ({len(wrapped)} chars)")

# 5. Error response
err = create_error_response("Timeout", "gemini")
print(f"[OK] Error response:       success={err.success}, error={err.error}")

# 6. MockLLM instantiation (no browser launch)
llm = MockLLM(provider="gemini")
print(f"\n[OK] MockLLM instance:     {llm}")
print(f"     .is_ready:            {llm.is_ready}")
print(f"     .provider_name:       {llm.provider_name}")
print(f"     .available_providers: {MockLLM.available_providers()}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
