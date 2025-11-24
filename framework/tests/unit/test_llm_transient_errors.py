"""Unit tests for LLM transient error handling (Category B).

These errors should be wrapped in TransientProviderError and raised.
The SDK already retried, so we raise with retry metadata.
"""

from unittest.mock import Mock, patch

import pytest

from agentic.framework.errors import TransientProviderError
from agentic.framework.llm import LLM
from agentic.framework.messages import Message


def test_rate_limit_error_raises_transient_provider_error():
    """RateLimitError after SDK retries should raise TransientProviderError."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    with patch.object(llm.model.chat.completions, "create") as mock_create:
        from openai import RateLimitError

        # SDK already retried and failed
        mock_create.side_effect = RateLimitError(
            message="Rate limit exceeded", response=Mock(status_code=429), body=None
        )

        # Should raise TransientProviderError (not raw RateLimitError)
        with pytest.raises(TransientProviderError) as exc_info:
            llm.call(messages)

        # Verify metadata is preserved
        error = exc_info.value
        assert "rate limit" in str(error).lower()
        assert error.error_type == "RateLimitError"


def test_internal_server_error_raises_transient_provider_error():
    """InternalServerError after SDK retries should raise TransientProviderError."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    with patch.object(llm.model.chat.completions, "create") as mock_create:
        from openai import InternalServerError

        # SDK already retried and failed
        mock_create.side_effect = InternalServerError(
            message="Internal server error", response=Mock(status_code=500), body=None
        )

        # Should raise TransientProviderError
        with pytest.raises(TransientProviderError) as exc_info:
            llm.call(messages)

        error = exc_info.value
        assert "internal server" in str(error).lower() or "500" in str(error)
        assert error.error_type == "InternalServerError"


def test_connection_error_raises_transient_provider_error():
    """APIConnectionError after SDK retries should raise TransientProviderError."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    with patch.object(llm.model.chat.completions, "create") as mock_create:
        from openai import APIConnectionError

        # SDK already retried and failed
        mock_create.side_effect = APIConnectionError(request=Mock())

        # Should raise TransientProviderError
        with pytest.raises(TransientProviderError) as exc_info:
            llm.call(messages)

        error = exc_info.value
        assert error.error_type == "APIConnectionError"


def test_timeout_error_raises_transient_provider_error():
    """APITimeoutError after SDK retries should raise TransientProviderError."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    with patch.object(llm.model.chat.completions, "create") as mock_create:
        from openai import APITimeoutError

        # SDK already retried and failed
        mock_create.side_effect = APITimeoutError(request=Mock())

        # Should raise TransientProviderError
        with pytest.raises(TransientProviderError) as exc_info:
            llm.call(messages)

        error = exc_info.value
        assert error.error_type == "APITimeoutError"


if __name__ == "__main__":
    test_rate_limit_error_raises_transient_provider_error()
    print("✓ RateLimitError → TransientProviderError")

    test_internal_server_error_raises_transient_provider_error()
    print("✓ InternalServerError → TransientProviderError")

    test_connection_error_raises_transient_provider_error()
    print("✓ APIConnectionError → TransientProviderError")

    test_timeout_error_raises_transient_provider_error()
    print("✓ APITimeoutError → TransientProviderError")

    print("\nAll transient error tests passed!")
