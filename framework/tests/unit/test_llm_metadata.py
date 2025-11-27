"""Tests for LLM response metadata attachment for debugging."""

from unittest.mock import Mock

from agentic.framework.llm import LLM
from agentic.framework.messages import ErrorCode, Message


class TestLLMResponseMetadata:
    """Test that LLM attaches raw response metadata to Messages"""

    def test_successful_response_has_metadata(self):
        """Normal successful response includes raw_content, finish_reason, model, usage"""
        llm = LLM(model_name="test-model", api_key="test-key")

        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[
            0
        ].message.content = (
            '{"reasoning": "test", "tool_calls": null, "result": "answer", "is_finished": true}'
        )
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.model = "actual-model-name"

        llm.model.chat.completions.create = Mock(return_value=mock_response)

        # Call LLM
        messages = [Message(role="user", content="test", timestamp=0.0)]
        result = llm.call(messages)

        # Check metadata exists
        assert result.metadata is not None
        assert "raw_content" in result.metadata
        assert "finish_reason" in result.metadata
        assert "model" in result.metadata
        assert "usage" in result.metadata

    def test_metadata_has_correct_values(self):
        """Metadata contains correct raw response values"""
        llm = LLM(model_name="test-model", api_key="test-key")

        raw_content = '  {"reasoning": "test"}  '  # With whitespace

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = raw_content
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.model = "gpt-4o-mini"

        llm.model.chat.completions.create = Mock(return_value=mock_response)

        messages = [Message(role="user", content="test", timestamp=0.0)]
        result = llm.call(messages)

        # Check values
        assert result.metadata["raw_content"] == raw_content  # Original with whitespace
        assert result.metadata["finish_reason"] == "stop"
        assert result.metadata["model"] == "gpt-4o-mini"
        assert result.metadata["usage"]["prompt_tokens"] == 100
        assert result.metadata["usage"]["completion_tokens"] == 50
        assert result.metadata["usage"]["total_tokens"] == 150

    def test_raw_content_differs_from_cleaned_content(self):
        """raw_content preserves original, content has cleaned version"""
        llm = LLM(model_name="test-model", api_key="test-key")

        # Response with markdown wrapping
        raw_with_markdown = '```json\n{"reasoning": "test", "tool_calls": null, "result": "done", "is_finished": true}\n```'

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = raw_with_markdown
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "test"

        llm.model.chat.completions.create = Mock(return_value=mock_response)

        messages = [Message(role="user", content="test", timestamp=0.0)]
        result = llm.call(messages)

        # raw_content has markdown
        assert result.metadata["raw_content"] == raw_with_markdown
        # content is cleaned
        assert "```" not in result.content
        assert result.content.strip().startswith("{")

    def test_content_filter_response_has_metadata(self):
        """Content filter responses include metadata"""
        llm = LLM(model_name="test-model", api_key="test-key")

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "content_filter"
        mock_response.choices[0].message.content = "Inappropriate content"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 50
        mock_response.model = "filter-model"

        llm.model.chat.completions.create = Mock(return_value=mock_response)

        messages = [Message(role="user", content="test", timestamp=0.0)]
        result = llm.call(messages)

        # Should have error code
        assert result.error_code == ErrorCode.CONTENT_FILTER

        # Should also have metadata
        assert result.metadata is not None
        assert result.metadata["raw_content"] == "Inappropriate content"
        assert result.metadata["finish_reason"] == "content_filter"
        assert result.metadata["model"] == "filter-model"

    def test_metadata_usage_matches_message_tokens(self):
        """Usage in metadata matches tokens_in/tokens_out on Message"""
        llm = LLM(model_name="test-model", api_key="test-key")

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[
            0
        ].message.content = (
            '{"reasoning": "x", "tool_calls": null, "result": "y", "is_finished": true}'
        )
        mock_response.usage.prompt_tokens = 123
        mock_response.usage.completion_tokens = 456
        mock_response.usage.total_tokens = 579
        mock_response.model = "test"

        llm.model.chat.completions.create = Mock(return_value=mock_response)

        messages = [Message(role="user", content="test", timestamp=0.0)]
        result = llm.call(messages)

        # Tokens should match between Message fields and metadata
        assert result.tokens_in == 123
        assert result.tokens_out == 456
        assert result.metadata["usage"]["prompt_tokens"] == 123
        assert result.metadata["usage"]["completion_tokens"] == 456
        assert result.metadata["usage"]["total_tokens"] == 579

    def test_finish_reason_length_preserved(self):
        """finish_reason='length' (truncated) is preserved in metadata"""
        llm = LLM(model_name="test-model", api_key="test-key")

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "length"  # Hit max_tokens
        mock_response.choices[0].message.content = '{"reasoning": "incomplete'  # Truncated
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 1000  # Max tokens
        mock_response.usage.total_tokens = 1010
        mock_response.model = "test"

        llm.model.chat.completions.create = Mock(return_value=mock_response)

        messages = [Message(role="user", content="test", timestamp=0.0)]
        result = llm.call(messages)

        # finish_reason tells us it was truncated
        assert result.metadata["finish_reason"] == "length"

    def test_model_name_captured_even_if_different_from_requested(self):
        """metadata.model captures actual model, which may differ from requested"""
        llm = LLM(model_name="gpt-4", api_key="test-key")  # Request gpt-4

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[
            0
        ].message.content = (
            '{"reasoning": "test", "tool_calls": null, "result": "x", "is_finished": true}'
        )
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-4-0314"  # Provider returns specific version

        llm.model.chat.completions.create = Mock(return_value=mock_response)

        messages = [Message(role="user", content="test", timestamp=0.0)]
        result = llm.call(messages)

        # Captures actual model used, not requested
        assert result.metadata["model"] == "gpt-4-0314"

    def test_raw_content_preserved_during_protocol_conversion(self):
        """raw_content preserves original even when tool_calls protocol converts content"""
        llm = LLM(model_name="test-model", api_key="test-key")

        # Mock native OpenAI tool_calls format (Grok, GPT-4 style)
        original_reasoning = "I'll use the calculator"

        mock_tool_call = Mock()
        mock_tool_call.id = "call_abc"
        mock_tool_call.function.name = "calculator"
        mock_tool_call.function.arguments = '{"x": 5, "y": 3}'

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.choices[0].message.content = original_reasoning
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 30
        mock_response.usage.total_tokens = 80
        mock_response.model = "grok-fast"

        llm.model.chat.completions.create = Mock(return_value=mock_response)

        messages = [Message(role="user", content="test", timestamp=0.0)]
        result = llm.call(messages)

        # raw_content should preserve the original reasoning text
        assert result.metadata["raw_content"] == original_reasoning

        # content should be converted to our JSON format
        import json

        parsed = json.loads(result.content)
        assert "tool_calls" in parsed
        assert parsed["tool_calls"][0]["tool"] == "calculator"
