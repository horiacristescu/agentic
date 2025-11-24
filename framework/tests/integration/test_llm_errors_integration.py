"""Integration tests for LLM error handling with real API calls.

These tests make real API calls that fail fast (no token consumption).
They validate that OpenRouter + OpenAI SDK behave as expected.
"""

import pytest

from agentic.framework.config import get_config
from agentic.framework.errors import AuthError, InvalidModelError
from agentic.framework.llm import LLM
from agentic.framework.messages import Message


def test_invalid_api_key_raises_auth_error():
    """Invalid API key should raise AuthError in real API call."""
    llm = LLM(model_name="openai/gpt-4", api_key="sk-invalid-key-12345")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Should raise AuthError before consuming tokens
    with pytest.raises(AuthError) as exc_info:
        llm.call(messages)

    # Verify error message mentions authentication
    assert "auth" in str(exc_info.value).lower() or "key" in str(exc_info.value).lower()


def test_invalid_model_raises_invalid_model_error():
    """Invalid model name should raise InvalidModelError in real API call."""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key found - set OPENROUTER_API_KEY in .env")

    # Use valid key but completely invalid model name
    llm = LLM(model_name="nonexistent/gpt-999-fake", api_key=config.api_key)
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Should raise InvalidModelError before consuming tokens
    with pytest.raises(InvalidModelError) as exc_info:
        llm.call(messages)

    # Verify error mentions model
    assert "model" in str(exc_info.value).lower() or "exist" in str(exc_info.value).lower()


if __name__ == "__main__":
    print("Testing real API error handling...")

    try:
        test_invalid_api_key_raises_auth_error()
        print("✓ Invalid API key → AuthError")
    except Exception as e:
        print(f"✗ Invalid API key test failed: {e}")

    try:
        test_invalid_model_raises_invalid_model_error()
        print("✓ Invalid model → InvalidModelError")
    except pytest.skip.Exception:
        print("⊘ Invalid model test skipped: No API key in .env")
    except Exception as e:
        print(f"✗ Invalid model test failed: {e}")
