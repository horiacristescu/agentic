"""Unit tests for LLM JSON mode functionality.

Tests that the LLM class correctly passes response_format parameter
when json_mode is enabled, without making actual API calls.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from agentic.framework.llm import LLM
from agentic.framework.messages import ErrorCode, Message


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].finish_reason = "stop"
    response.choices[0].message = Mock()
    response.choices[0].message.content = '{"result": "test response"}'
    response.choices[0].message.tool_calls = None
    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.model = "test-model"
    return response


def test_json_mode_disabled_by_default():
    """Test that JSON mode is disabled by default."""
    with patch("agentic.framework.llm.openai.OpenAI") as mock_openai_class:
        llm = LLM(model_name="test-model", api_key="test-key")
        
        assert llm.json_mode is False


def test_json_mode_can_be_enabled():
    """Test that JSON mode can be enabled via constructor."""
    with patch("agentic.framework.llm.openai.OpenAI") as mock_openai_class:
        llm = LLM(model_name="test-model", api_key="test-key", json_mode=True)
        
        assert llm.json_mode is True


def test_json_mode_disabled_does_not_pass_response_format(mock_openai_response):
    """Test that response_format is NOT passed when json_mode=False."""
    with patch("agentic.framework.llm.openai.OpenAI") as mock_openai_class:
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response
        
        # Create LLM without JSON mode
        llm = LLM(model_name="test-model", api_key="test-key", json_mode=False)
        
        # Make a call
        messages = [Message(role="user", content="test", timestamp=time.time())]
        llm.call(messages)
        
        # Verify response_format was NOT passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "response_format" not in call_kwargs


def test_json_mode_enabled_passes_response_format(mock_openai_response):
    """Test that response_format is passed when json_mode=True."""
    with patch("agentic.framework.llm.openai.OpenAI") as mock_openai_class:
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response
        
        # Create LLM with JSON mode enabled
        llm = LLM(model_name="test-model", api_key="test-key", json_mode=True)
        
        # Make a call
        messages = [Message(role="user", content="test", timestamp=time.time())]
        llm.call(messages)
        
        # Verify response_format was passed with correct value
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_json_mode_preserves_other_parameters(mock_openai_response):
    """Test that enabling JSON mode doesn't affect other parameters."""
    with patch("agentic.framework.llm.openai.OpenAI") as mock_openai_class:
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response
        
        # Create LLM with custom parameters and JSON mode
        llm = LLM(
            model_name="test-model",
            api_key="test-key",
            temperature=0.7,
            max_tokens=2000,
            json_mode=True,
        )
        
        # Make a call
        messages = [Message(role="user", content="test", timestamp=time.time())]
        llm.call(messages)
        
        # Verify all parameters are passed correctly
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 2000
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_json_mode_with_multiple_calls(mock_openai_response):
    """Test that JSON mode setting persists across multiple calls."""
    with patch("agentic.framework.llm.openai.OpenAI") as mock_openai_class:
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response
        
        # Create LLM with JSON mode
        llm = LLM(model_name="test-model", api_key="test-key", json_mode=True)
        
        # Make multiple calls
        messages = [Message(role="user", content="test", timestamp=time.time())]
        llm.call(messages)
        llm.call(messages)
        llm.call(messages)
        
        # Verify all calls included response_format
        assert mock_client.chat.completions.create.call_count == 3
        for call in mock_client.chat.completions.create.call_args_list:
            call_kwargs = call[1]
            assert call_kwargs["response_format"] == {"type": "json_object"}


def test_json_mode_response_parsing(mock_openai_response):
    """Test that JSON mode doesn't break response parsing."""
    with patch("agentic.framework.llm.openai.OpenAI") as mock_openai_class:
        # Setup mock with JSON response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response
        
        # Create LLM with JSON mode
        llm = LLM(model_name="test-model", api_key="test-key", json_mode=True)
        
        # Make a call
        messages = [Message(role="user", content="test", timestamp=time.time())]
        response = llm.call(messages)
        
        # Verify response was parsed correctly
        assert response.role == "assistant"
        assert response.content == '{"result": "test response"}'
        assert response.tokens_in == 10
        assert response.tokens_out == 20
        assert response.error_code is None


def test_json_mode_with_empty_response():
    """Test that JSON mode handles empty responses correctly."""
    with patch("agentic.framework.llm.openai.OpenAI") as mock_openai_class:
        # Setup mock with empty content
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = ""  # Empty!
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 1
        mock_response.model = "test-model"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create LLM with JSON mode
        llm = LLM(model_name="test-model", api_key="test-key", json_mode=True)
        
        # Make a call
        messages = [Message(role="user", content="test", timestamp=time.time())]
        response = llm.call(messages)
        
        # Should detect empty response
        assert response.error_code == ErrorCode.EMPTY_RESPONSE


def test_json_mode_with_content_filter():
    """Test that JSON mode handles content filter correctly."""
    with patch("agentic.framework.llm.openai.OpenAI") as mock_openai_class:
        # Setup mock with content_filter finish reason
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "content_filter"
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Filtered content"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "test-model"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create LLM with JSON mode
        llm = LLM(model_name="test-model", api_key="test-key", json_mode=True)
        
        # Make a call
        messages = [Message(role="user", content="test", timestamp=time.time())]
        response = llm.call(messages)
        
        # Should detect content filter
        assert response.error_code == ErrorCode.CONTENT_FILTER

