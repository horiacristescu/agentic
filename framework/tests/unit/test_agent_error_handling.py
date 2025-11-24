"""Unit tests for Agent error handling (Category A & B).

Category A errors (config/auth) should propagate up and crash fast.
Category B errors (transient) should also propagate up (caller decides retry).
"""

from unittest.mock import Mock, patch

import pytest

from agentic.framework.agents import Agent
from agentic.framework.errors import (
    AuthError,
    InvalidModelError,
    PermissionError,
    TransientProviderError,
)
from agentic.framework.llm import LLM


def test_auth_error_propagates_from_llm():
    """AuthError from LLM should propagate up (crash fast - Category A)"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    # Mock LLM to raise AuthError
    with patch.object(llm, "call") as mock_call:
        mock_call.side_effect = AuthError("Invalid API key")

        # Should raise AuthError (not catch it)
        with pytest.raises(AuthError) as exc_info:
            agent.run("Hello")

        assert "Invalid API key" in str(exc_info.value)


def test_invalid_model_error_propagates_from_llm():
    """InvalidModelError from LLM should propagate up (crash fast - Category A)"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    # Mock LLM to raise InvalidModelError
    with patch.object(llm, "call") as mock_call:
        mock_call.side_effect = InvalidModelError("Model 'fake-model' does not exist")

        # Should raise InvalidModelError (not catch it)
        with pytest.raises(InvalidModelError) as exc_info:
            agent.run("Hello")

        assert "fake-model" in str(exc_info.value)


def test_permission_error_propagates_from_llm():
    """PermissionError from LLM should propagate up (crash fast - Category A)"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    # Mock LLM to raise PermissionError
    with patch.object(llm, "call") as mock_call:
        mock_call.side_effect = PermissionError("No access to model")

        # Should raise PermissionError (not catch it)
        with pytest.raises(PermissionError) as exc_info:
            agent.run("Hello")

        assert "No access" in str(exc_info.value)


def test_transient_provider_error_propagates_from_llm():
    """TransientProviderError from LLM should propagate up (Category B - crash, caller retries)"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    # Mock LLM to raise TransientProviderError
    mock_last_error = Mock()
    with patch.object(llm, "call") as mock_call:
        mock_call.side_effect = TransientProviderError(
            message="Rate limit exceeded",
            attempt_count=3,
            last_error=mock_last_error,
            error_type="RateLimitError",
        )

        # Should raise TransientProviderError (not catch it)
        with pytest.raises(TransientProviderError) as exc_info:
            agent.run("Hello")

        assert "Rate limit" in str(exc_info.value)


def test_transient_error_metadata_accessible():
    """TransientProviderError should preserve metadata for caller inspection"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    # Mock LLM to raise TransientProviderError with metadata
    mock_last_error = Exception("Original timeout error")
    with patch.object(llm, "call") as mock_call:
        mock_call.side_effect = TransientProviderError(
            message="Request timed out after 3 attempts",
            attempt_count=3,
            last_error=mock_last_error,
            error_type="APITimeoutError",
        )

        with pytest.raises(TransientProviderError) as exc_info:
            agent.run("Hello")

        # Verify metadata is accessible
        error = exc_info.value
        assert error.error_type == "APITimeoutError"
        assert error.attempt_count == 3
        assert error.last_error == mock_last_error


if __name__ == "__main__":
    print("=== Category A: Config/Auth Errors ===")
    test_auth_error_propagates_from_llm()
    print("✓ AuthError propagates from LLM")

    test_invalid_model_error_propagates_from_llm()
    print("✓ InvalidModelError propagates from LLM")

    test_permission_error_propagates_from_llm()
    print("✓ PermissionError propagates from LLM")

    print("\n=== Category B: Transient Errors ===")
    test_transient_provider_error_propagates_from_llm()
    print("✓ TransientProviderError propagates from LLM")

    test_transient_error_metadata_accessible()
    print("✓ Transient error metadata accessible")

    print("\nAll Agent error handling tests passed!")
