"""Unit tests for Agent handling of LLM semantic errors (Category C/D).

Category C/D errors (semantic/output) should be added to conversation history
so the agent can see them and reason about recovery.
"""

from unittest.mock import patch

from agentic.framework.agents import Agent
from agentic.framework.llm import LLM
from agentic.framework.messages import ErrorCode, Message, ResultStatus


def test_content_filter_error_added_to_conversation():
    """When LLM returns content_filter error, Agent should add it to conversation and continue"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    # Mock LLM to return content_filter error on first call, then success
    with patch.object(llm, "call") as mock_call:
        # First call: content filter error
        mock_call.return_value = Message(
            role="assistant",
            content="Content was blocked by safety filters",
            error_code=ErrorCode.CONTENT_FILTER,
            timestamp=0.0,
            tokens_in=10,
            tokens_out=5,
        )

        # Agent should not crash, should add error to conversation
        result = agent.run("Test input")

        # Should fail gracefully (can't proceed with filtered content)
        assert result.status in [ResultStatus.ERROR, ResultStatus.MAX_TURNS_REACHED]

        # Error should be in conversation history
        error_messages = [msg for msg in agent.messages if msg.error_code == ErrorCode.CONTENT_FILTER]
        assert len(error_messages) > 0


def test_empty_response_error_added_to_conversation():
    """When LLM returns empty_response error, Agent should add it to conversation"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        # LLM returns empty response error
        mock_call.return_value = Message(
            role="assistant",
            content="Received empty response from API",
            error_code=ErrorCode.EMPTY_RESPONSE,
            timestamp=0.0,
            tokens_in=10,
            tokens_out=0,
        )

        result = agent.run("Test input")

        # Should fail gracefully
        assert result.status in [ResultStatus.ERROR, ResultStatus.MAX_TURNS_REACHED]

        # Error should be in conversation
        error_messages = [msg for msg in agent.messages if msg.error_code == ErrorCode.EMPTY_RESPONSE]
        assert len(error_messages) > 0


def test_semantic_error_does_not_crash_agent():
    """Semantic errors should not crash - agent continues trying (up to max_turns)"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=3)

    with patch.object(llm, "call") as mock_call:
        # Always return content filter error
        mock_call.return_value = Message(
            role="assistant",
            content="Content was blocked",
            error_code=ErrorCode.CONTENT_FILTER,
            timestamp=0.0,
            tokens_in=10,
            tokens_out=5,
        )

        # Should not raise exception - just hit max turns
        result = agent.run("Test input")

        # Should reach max turns, not crash
        assert result.status == ResultStatus.MAX_TURNS_REACHED
        assert agent.turn_count == 3


if __name__ == "__main__":
    test_content_filter_error_added_to_conversation()
    print("✓ Content filter error added to conversation")

    test_empty_response_error_added_to_conversation()
    print("✓ Empty response error added to conversation")

    test_semantic_error_does_not_crash_agent()
    print("✓ Semantic errors don't crash agent")

    print("\nAll Agent semantic error tests passed!")
