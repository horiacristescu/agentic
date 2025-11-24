"""Contract tests validating OpenAI SDK + OpenRouter API behavior.

These tests call the OpenAI client DIRECTLY (not through our LLM wrapper)
to document and validate actual API behavior, error types, and edge cases.

Purpose: Ensure our assumptions about the API match reality.
If the API changes, these tests will catch it.
"""

import openai
import pytest

from agentic.framework.config import get_config


def test_openai_sdk_exceptions_exist():
    """Verify the OpenAI SDK has all exception types we depend on."""
    # These should exist in the SDK
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        PermissionDeniedError,
        RateLimitError,
    )

    # Just importing proves they exist
    assert AuthenticationError is not None
    assert BadRequestError is not None
    assert PermissionDeniedError is not None
    assert RateLimitError is not None
    assert InternalServerError is not None
    assert APIConnectionError is not None
    assert APITimeoutError is not None


def test_invalid_api_key_throws_authentication_error():
    """OpenRouter should throw AuthenticationError for invalid API key."""
    # Call raw OpenAI client with bad key
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key="sk-invalid-key-12345")

    with pytest.raises(openai.AuthenticationError) as exc_info:
        client.chat.completions.create(
            model="openai/gpt-4", messages=[{"role": "user", "content": "test"}]
        )

    print(f"\n✓ Invalid API key → {type(exc_info.value).__name__}")
    print(f"  Message: {str(exc_info.value)[:100]}")


def test_invalid_model_throws_bad_request_error():
    """OpenRouter should throw BadRequestError for nonexistent model."""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    # Call raw OpenAI client with invalid model
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=config.api_key)

    with pytest.raises(openai.BadRequestError) as exc_info:
        client.chat.completions.create(
            model="nonexistent/fake-model-999", messages=[{"role": "user", "content": "test"}]
        )

    print(f"\n✓ Invalid model → {type(exc_info.value).__name__}")
    print(f"  Message: {str(exc_info.value)[:100]}")


def test_malformed_message_throws_bad_request():
    """What does OpenAI throw for malformed message structure?"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=config.api_key)

    # Try sending a message with invalid role
    try:
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "invalid_role", "content": "test"}],  # type: ignore[dict-item]
        )
        pytest.fail(f"Expected BadRequestError but got success: {response}")
    except openai.BadRequestError as e:
        print("\n✓ Invalid role → BadRequestError")
        print(f"  Message: {str(e)[:100]}")
    except Exception as e:
        print(f"\n✗ Unexpected error type: {type(e).__name__}")
        print(f"  Message: {str(e)[:100]}")
        raise


def test_timeout_throws_api_timeout_error():
    """OpenAI SDK should throw timeout error when request times out."""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    # Set unreasonably low timeout
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config.api_key,
        timeout=0.001,  # 1 millisecond - guaranteed to timeout
    )

    try:
        response = client.chat.completions.create(
            model=config.model_name, messages=[{"role": "user", "content": "test"}]
        )
        pytest.fail(f"Expected timeout but got success: {response}")
    except openai.APITimeoutError as e:
        print("\n✓ Low timeout → APITimeoutError")
        print(f"  Message: {str(e)[:100]}")
    except Exception as e:
        print(f"\n✗ Unexpected error type: {type(e).__name__}")
        print(f"  Message: {str(e)[:100]}")
        raise


def test_context_length_exceeded():
    """OpenAI should throw error when context length is exceeded."""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    # gpt-3.5-turbo has 16K token limit
    # Send ~20K tokens worth of text to reliably exceed it (1 token ≈ 4 chars)
    # 20K tokens × 4 = 80K characters
    huge_text = "word " * 20000  # ~100K characters, safely over 16K token limit

    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=config.api_key)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # 16K context limit
            messages=[{"role": "user", "content": huge_text}],
        )
        pytest.fail(f"Expected context length error but got success: {response}")
    except openai.BadRequestError as e:
        # Should be BadRequestError with message about context length
        error_msg = str(e).lower()
        if "context" in error_msg or "length" in error_msg or "token" in error_msg:
            print("\n✓ Context length exceeded → BadRequestError")
            print(f"  Message: {str(e)[:100]}")
        else:
            print("\n✗ Got BadRequestError but not about context length")
            print(f"  Message: {str(e)[:200]}")
            raise
    except Exception as e:
        print(f"\n✗ Unexpected error type: {type(e).__name__}")
        print(f"  Message: {str(e)[:100]}")
        raise


def test_output_exceeds_max_tokens():
    """When LLM output exceeds max_tokens, detect truncation via finish_reason or token count."""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=config.api_key)

    max_tokens_limit = 20
    # Ask LLM to generate long output but limit tokens
    response = client.chat.completions.create(
        model=config.model_name,
        messages=[{"role": "user", "content": "Count from 1 to 100, one number per line."}],
        max_tokens=max_tokens_limit,  # Way too few for this task
    )

    # Different providers indicate truncation differently:
    # - OpenAI: finish_reason='length'
    # - xAI/Grok: finish_reason=None, but completion_tokens == max_tokens
    finish_reason = response.choices[0].finish_reason
    assert response.usage is not None, "Expected usage data in response"
    completion_tokens = response.usage.completion_tokens

    truncated = (finish_reason == "length") or (completion_tokens == max_tokens_limit)

    assert truncated, (
        f"Expected truncation (finish_reason='length' OR completion_tokens==max_tokens), "
        f"but got finish_reason='{finish_reason}' and {completion_tokens}/{max_tokens_limit} tokens"
    )

    print(
        f"\n✓ Output truncated at max_tokens (finish_reason={finish_reason}, tokens={completion_tokens})"
    )
    content = response.choices[0].message.content
    assert content is not None, "Expected content in response"
    print(f"  Content preview: {content[:50]}...")


def test_connection_error_after_retries():
    """Connection errors should raise APIConnectionError after retries exhausted."""
    # Use invalid URL to force connection error
    client = openai.OpenAI(
        base_url="https://invalid-url-that-does-not-exist-12345.com",
        api_key="test-key",
        max_retries=1,  # Low retries to make test fast
        timeout=2.0,  # Low timeout
    )

    try:
        client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "test"}]
        )
        pytest.fail("Expected APIConnectionError but call succeeded")
    except openai.APIConnectionError as e:
        print("\n✓ Connection failure → APIConnectionError")
        print(f"  Message: {str(e)[:100]}")
    except Exception as e:
        print(f"\n✗ Unexpected error type: {type(e).__name__}")
        print(f"  Message: {str(e)[:100]}")
        raise


"""
Note: OpenRouter does NOT strictly validate max_tokens parameter.
Both very large values (1000000) and invalid values (-1) are silently
accepted and the API returns successful responses. This differs from
direct OpenAI API behavior and cannot be reliably tested.

Note: Some transient errors cannot be reliably triggered in tests:
- RateLimitError: Requires actually hitting rate limits (expensive/slow)
- InternalServerError: Requires provider returning 5xx errors

These exception types are validated to exist (test_openai_sdk_exceptions_exist)
and we trust the SDK's retry logic to raise them after retries are exhausted.
The SDK uses exponential backoff and retries RateLimitError, InternalServerError,
and APIConnectionError automatically (configurable via max_retries parameter).
"""


if __name__ == "__main__":
    test_openai_sdk_exceptions_exist()
    print("✓ SDK exception types exist")

    test_invalid_api_key_throws_authentication_error()
    print("✓ Invalid API key → AuthenticationError")

    test_invalid_model_throws_bad_request_error()
    print("✓ Invalid model → BadRequestError")

    test_malformed_message_throws_bad_request()
    print("✓ Malformed message → BadRequestError")

    test_timeout_throws_api_timeout_error()
    print("✓ Timeout → APITimeoutError")

    test_context_length_exceeded()
    print("✓ Input context too long → BadRequestError")

    test_output_exceeds_max_tokens()
    print("✓ Output exceeds max_tokens → finish_reason='length'")

    test_connection_error_after_retries()
    print("✓ Connection error → APIConnectionError (after retries)")

    print("\nAll OpenAI API behavior tests passed!")
