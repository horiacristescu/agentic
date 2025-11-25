"""Unit tests for Agent basic behavior.

Tests multi-turn reasoning, max_turns enforcement, and conversation continuation.
"""

import json
from unittest.mock import patch

from agentic.framework.agents import Agent
from agentic.framework.llm import LLM
from agentic.framework.messages import Message, ResultStatus


def test_multi_turn_reasoning():
    """Agent should use multiple turns when LLM doesn't finish immediately"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        # Turn 1: Agent reasons but doesn't finish
        turn1_response = json.dumps(
            {
                "reasoning": "I need to think about this more.",
                "tool_calls": None,
                "result": None,
                "is_finished": False,
            }
        )

        # Turn 2: Agent reasons more but still not done
        turn2_response = json.dumps(
            {
                "reasoning": "Getting closer to the answer.",
                "tool_calls": None,
                "result": None,
                "is_finished": False,
            }
        )

        # Turn 3: Agent finishes
        turn3_response = json.dumps(
            {
                "reasoning": "Now I have the answer.",
                "tool_calls": None,
                "result": "The final answer is 42",
                "is_finished": True,
            }
        )

        mock_call.side_effect = [
            Message(
                role="assistant", content=turn1_response, timestamp=0.0, tokens_in=10, tokens_out=20
            ),
            Message(
                role="assistant", content=turn2_response, timestamp=0.0, tokens_in=10, tokens_out=20
            ),
            Message(
                role="assistant", content=turn3_response, timestamp=0.0, tokens_in=10, tokens_out=20
            ),
        ]

        result = agent.run("What is the answer?")

        # Should succeed after 3 turns
        assert result.status == ResultStatus.SUCCESS
        assert result.value == "The final answer is 42"
        assert agent.turn_count == 3

        # Should have called LLM 3 times
        assert mock_call.call_count == 3


def test_max_turns_enforcement():
    """Agent should stop at max_turns and return MAX_TURNS_REACHED status"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=3)

    with patch.object(llm, "call") as mock_call:
        # Mock LLM to never finish (is_finished always False)
        never_finish_response = json.dumps(
            {
                "reasoning": "Still thinking...",
                "tool_calls": None,
                "result": None,
                "is_finished": False,
            }
        )

        mock_call.return_value = Message(
            role="assistant",
            content=never_finish_response,
            timestamp=0.0,
            tokens_in=10,
            tokens_out=20,
        )

        result = agent.run("Impossible task")

        # Should hit max turns
        assert result.status == ResultStatus.MAX_TURNS_REACHED
        assert agent.turn_count == 3
        assert "[MAX_TURNS]" in result.value

        # Should have tried max_turns times
        assert mock_call.call_count == 3


def test_conversation_continuation_with_reset_false():
    """Agent should continue conversation when reset=False"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        # First run - agent responds
        first_response = json.dumps(
            {
                "reasoning": "User asked about weather",
                "tool_calls": None,
                "result": "The weather is sunny",
                "is_finished": True,
            }
        )

        # Second run - agent continues conversation
        second_response = json.dumps(
            {
                "reasoning": "User asked a follow-up",
                "tool_calls": None,
                "result": "Temperature is 22C",
                "is_finished": True,
            }
        )

        mock_call.side_effect = [
            Message(
                role="assistant", content=first_response, timestamp=0.0, tokens_in=10, tokens_out=20
            ),
            Message(
                role="assistant",
                content=second_response,
                timestamp=0.0,
                tokens_in=15,
                tokens_out=15,
            ),
        ]

        # First query
        result1 = agent.run("What's the weather?")
        assert result1.status == ResultStatus.SUCCESS
        assert result1.value == "The weather is sunny"
        assert agent.turn_count == 1
        messages_after_first = len(agent.messages)

        # Second query with reset=False (continue conversation)
        result2 = agent.run("What's the temperature?", reset=False)
        assert result2.status == ResultStatus.SUCCESS
        assert result2.value == "Temperature is 22C"
        assert agent.turn_count == 2  # Accumulated

        # Messages should accumulate (not reset)
        assert len(agent.messages) > messages_after_first

        # Should have both user queries in history
        user_messages = [msg for msg in agent.messages if msg.role == "user"]
        assert len(user_messages) == 2


def test_conversation_reset_with_reset_true():
    """Agent should reset conversation when reset=True (default)"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        success_response = json.dumps(
            {
                "reasoning": "Answering the question",
                "tool_calls": None,
                "result": "Answer",
                "is_finished": True,
            }
        )

        mock_call.return_value = Message(
            role="assistant",
            content=success_response,
            timestamp=0.0,
            tokens_in=10,
            tokens_out=20,
        )

        # First query
        agent.run("First question")
        first_turn_count = agent.turn_count
        first_message_count = len(agent.messages)

        # Second query with reset=True (default)
        agent.run("Second question")

        # Turn count should reset
        assert agent.turn_count == 1

        # Messages should only contain second conversation
        user_messages = [msg for msg in agent.messages if msg.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].content == "Second question"


if __name__ == "__main__":
    print("=== Agent Basic Behavior Tests ===\n")

    test_multi_turn_reasoning()
    print("✓ Multi-turn reasoning works")

    test_max_turns_enforcement()
    print("✓ Max turns enforcement works")

    test_conversation_continuation_with_reset_false()
    print("✓ Conversation continuation (reset=False) works")

    test_conversation_reset_with_reset_true()
    print("✓ Conversation reset (reset=True) works")

    print("\nAll Agent behavior tests passed!")
