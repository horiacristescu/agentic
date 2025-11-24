"""Unit tests for LLM semantic error handling (Category C & D).

These errors should be returned as Message with error_code,
not raised as exceptions. The agent can reason about them.
"""

from unittest.mock import Mock, patch

from agentic.framework.llm import LLM
from agentic.framework.messages import Message


def test_content_filter_returns_rich_message():
    """When content is filtered, should return Message with error_code."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Mock response with content_filter finish_reason
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].finish_reason = "content_filter"
    mock_response.choices[0].message.content = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 0

    with patch.object(llm.model.chat.completions, "create", return_value=mock_response):
        result = llm.call(messages)

    # Should return Message with error_code (not raise!)
    assert isinstance(result, Message)
    assert result.error_code == "content_filter"
    assert result.role == "assistant"
    assert "filter" in result.content.lower() or "blocked" in result.content.lower()


def test_empty_response_returns_rich_message():
    """When API returns empty content, should return Message with error_code."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="test", timestamp=0.0)]

    # Mock response with empty content
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].message.content = None  # Empty!
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5

    with patch.object(llm.model.chat.completions, "create", return_value=mock_response):
        result = llm.call(messages)

    # Should return Message with error_code (not raise!)
    assert isinstance(result, Message)
    assert result.error_code == "empty_response"
    assert result.role == "assistant"
    assert "empty" in result.content.lower()


def test_normal_response_has_no_error_code():
    """Normal successful response should not have error_code."""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    messages = [Message(role="user", content="What is 2+2?", timestamp=0.0)]

    # Mock normal successful response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].message.content = "The answer is 4."
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 8

    with patch.object(llm.model.chat.completions, "create", return_value=mock_response):
        result = llm.call(messages)

    # Should return normal Message without error_code
    assert isinstance(result, Message)
    assert result.error_code is None
    assert result.role == "assistant"
    assert result.content == "The answer is 4."
    assert result.tokens_in == 10
    assert result.tokens_out == 8


if __name__ == "__main__":
    test_content_filter_returns_rich_message()
    print("✓ Content filter → Message with error_code")

    test_empty_response_returns_rich_message()
    print("✓ Empty response → Message with error_code")

    test_normal_response_has_no_error_code()
    print("✓ Normal response → no error_code")

    print("\nAll semantic error tests passed!")
