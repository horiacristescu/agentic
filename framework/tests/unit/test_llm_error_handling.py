"""Unit tests for LLM error handling and OpenAI exception mapping."""

from unittest.mock import Mock, patch

import pytest

from agentic.framework.errors import (
    AuthError,
    InvalidModelError,
    MalformedResponseError,
    PermissionError,
)
from agentic.framework.llm import LLM
from agentic.framework.messages import Message


def test_authentication_error_raises_auth_error():
    """OpenAI AuthenticationError should raise our AuthError (Category A - crash fast)."""
    llm = LLM(model_name="gpt-4", api_key="invalid-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Mock the OpenAI client to raise AuthenticationError
    with patch.object(llm.model.chat.completions, "create") as mock_create:
        # Simulate OpenAI's AuthenticationError
        from openai import AuthenticationError

        mock_create.side_effect = AuthenticationError(
            message="Invalid API key", response=Mock(status_code=401), body=None
        )

        # Should raise AuthError, not return Message
        with pytest.raises(AuthError) as exc_info:
            llm.call(messages)

        assert "Invalid API key" in str(exc_info.value)


def test_bad_request_error_raises_invalid_model_error():
    """OpenAI BadRequestError for invalid model should raise InvalidModelError."""
    llm = LLM(model_name="gpt-99-nonexistent", api_key="valid-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Mock the OpenAI client to raise BadRequestError
    with patch.object(llm.model.chat.completions, "create") as mock_create:
        from openai import BadRequestError

        mock_create.side_effect = BadRequestError(
            message="The model `gpt-99-nonexistent` does not exist",
            response=Mock(status_code=400),
            body=None,
        )

        # Should raise InvalidModelError, not return Message
        with pytest.raises(InvalidModelError) as exc_info:
            llm.call(messages)

        assert "gpt-99-nonexistent" in str(exc_info.value) or "model" in str(exc_info.value).lower()


def test_permission_denied_error_raises_permission_error():
    """OpenAI PermissionDeniedError should raise PermissionError."""
    llm = LLM(model_name="gpt-4", api_key="limited-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Mock the OpenAI client to raise PermissionDeniedError
    with patch.object(llm.model.chat.completions, "create") as mock_create:
        from openai import PermissionDeniedError

        mock_create.side_effect = PermissionDeniedError(
            message="Insufficient permissions to access this model",
            response=Mock(status_code=403),
            body=None,
        )

        # Should raise PermissionError, not return Message
        with pytest.raises(PermissionError) as exc_info:
            llm.call(messages)

        assert "permission" in str(exc_info.value).lower()


def test_empty_choices_raises_malformed_response_error():
    """Provider returning empty choices array should raise MalformedResponseError."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Mock response with empty choices
    mock_response = Mock()
    mock_response.choices = []  # MALFORMED: no choices
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=0)

    with patch.object(llm.model.chat.completions, "create", return_value=mock_response):
        with pytest.raises(MalformedResponseError) as exc_info:
            llm.call(messages)

        assert "no choices" in str(exc_info.value).lower()
        assert "contract violation" in str(exc_info.value).lower()


def test_missing_usage_raises_malformed_response_error():
    """Provider returning response without usage data should raise MalformedResponseError."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Mock response with missing usage
    mock_response = Mock()
    mock_response.choices = [Mock(finish_reason="stop", message=Mock(content="test"))]
    mock_response.usage = None  # MALFORMED: missing usage

    with patch.object(llm.model.chat.completions, "create", return_value=mock_response):
        with pytest.raises(MalformedResponseError) as exc_info:
            llm.call(messages)

        assert "usage" in str(exc_info.value).lower()
        assert "contract violation" in str(exc_info.value).lower()


def test_missing_message_field_raises_malformed_response_error():
    """Provider returning choice without message field should raise MalformedResponseError."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Mock response with choice missing message field
    mock_response = Mock()
    mock_choice = Mock(spec=[])  # No attributes, missing 'message'
    mock_choice.finish_reason = "stop"
    # Explicitly delete 'message' if mock creates it
    if hasattr(mock_choice, "message"):
        delattr(mock_choice, "message")
    mock_response.choices = [mock_choice]  # MALFORMED: no message field
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)

    with patch.object(llm.model.chat.completions, "create", return_value=mock_response):
        with pytest.raises(MalformedResponseError) as exc_info:
            llm.call(messages)

        assert "message" in str(exc_info.value).lower()
        assert "contract violation" in str(exc_info.value).lower()


if __name__ == "__main__":
    test_authentication_error_raises_auth_error()
    print("✓ AuthenticationError → AuthError")

    test_bad_request_error_raises_invalid_model_error()
    print("✓ BadRequestError → InvalidModelError")

    test_permission_denied_error_raises_permission_error()
    print("✓ PermissionDeniedError → PermissionError")

    test_empty_choices_raises_malformed_response_error()
    print("✓ Empty choices → MalformedResponseError")

    test_missing_usage_raises_malformed_response_error()
    print("✓ Missing usage → MalformedResponseError")

    test_missing_message_field_raises_malformed_response_error()
    print("✓ Missing message field → MalformedResponseError")
