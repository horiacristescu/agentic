"""Unit tests for Agent state management and token tracking.

Tests that tokens accumulate correctly, trajectory is preserved,
and state persists across reset=False runs.
"""

import json
from unittest.mock import patch

from pydantic import BaseModel

from agentic.framework.agents import Agent
from agentic.framework.llm import LLM
from agentic.framework.messages import Message, ResultStatus, ToolCall
from agentic.framework.tools import create_tool


def test_tokens_accumulate_across_turns():
    """Tokens should accumulate across multiple turns"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        # Turn 1: 10 in, 20 out = 30 total
        turn1 = json.dumps(
            {
                "reasoning": "First turn",
                "tool_calls": None,
                "result": None,
                "is_finished": False,
            }
        )

        # Turn 2: 15 in, 25 out = 40 total
        turn2 = json.dumps(
            {
                "reasoning": "Second turn",
                "tool_calls": None,
                "result": None,
                "is_finished": False,
            }
        )

        # Turn 3: 12 in, 18 out = 30 total
        turn3 = json.dumps(
            {
                "reasoning": "Done",
                "tool_calls": None,
                "result": "Answer",
                "is_finished": True,
            }
        )

        mock_call.side_effect = [
            Message(role="assistant", content=turn1, timestamp=0.0, tokens_in=10, tokens_out=20),
            Message(role="assistant", content=turn2, timestamp=0.0, tokens_in=15, tokens_out=25),
            Message(role="assistant", content=turn3, timestamp=0.0, tokens_in=12, tokens_out=18),
        ]

        result = agent.run("Test")

        # Tokens should accumulate: 30 + 40 + 30 = 100
        assert agent.tokens_used == 100
        assert result.status == ResultStatus.SUCCESS


def test_tokens_returned_in_result_metadata():
    """Final Result should include token count in metadata"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        response = json.dumps(
            {
                "reasoning": "Answer",
                "tool_calls": None,
                "result": "Done",
                "is_finished": True,
            }
        )

        mock_call.return_value = Message(
            role="assistant",
            content=response,
            timestamp=0.0,
            tokens_in=50,
            tokens_out=75,
        )

        result = agent.run("Test")

        # Metadata should include tokens
        assert result.metadata is not None
        assert "tokens" in result.metadata
        assert result.metadata["tokens"] == 125  # 50 + 75

        # Should also include turns
        assert "turns" in result.metadata
        assert result.metadata["turns"] == 1


def test_messages_contains_full_trajectory():
    """Agent.messages should contain complete trajectory including all messages"""

    class AddToolSchema(BaseModel):
        """Adds two numbers"""

        x: int
        y: int

        def execute(self) -> str:
            return str(self.x + self.y)

    llm = LLM(model_name="gpt-4", api_key="test-key")
    add_tool = create_tool(AddToolSchema, dependencies={})
    add_tool.name = "add"
    add_tool.description = "Adds two numbers"
    agent = Agent(llm=llm, tools=[add_tool], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        # Turn 1: Call tool
        turn1 = json.dumps(
            {
                "reasoning": "Need to add",
                "tool_calls": [{"id": "call_1", "tool": "add", "args": {"x": 2, "y": 3}}],
                "result": None,
                "is_finished": False,
            }
        )

        # Turn 2: Finish
        turn2 = json.dumps(
            {
                "reasoning": "Got result",
                "tool_calls": None,
                "result": "The answer is 5",
                "is_finished": True,
            }
        )

        mock_call.side_effect = [
            Message(
                role="assistant",
                content=turn1,
                timestamp=0.0,
                tokens_in=10,
                tokens_out=20,
                tool_calls=[ToolCall(id="call_1", tool="add", args={"x": 2, "y": 3})],
            ),
            Message(role="assistant", content=turn2, timestamp=0.0, tokens_in=10, tokens_out=20),
        ]

        agent.run("What is 2 + 3?")

        # Trajectory should include:
        # 1. System prompt
        # 2. User query
        # 3. Assistant response (turn 1 with tool call)
        # 4. Tool result
        # 5. Assistant response (turn 2 with final answer)
        assert len(agent.messages) >= 5

        # Check message types
        roles = [msg.role for msg in agent.messages]
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert roles[2] == "assistant"
        assert roles[3] == "tool"
        assert roles[4] == "assistant"


def test_trajectory_includes_tool_calls_and_results():
    """Trajectory should include both tool calls and their results with IDs"""

    class MultiplyToolSchema(BaseModel):
        """Multiplies two numbers"""

        x: int
        y: int

        def execute(self) -> str:
            return str(self.x * self.y)

    llm = LLM(model_name="gpt-4", api_key="test-key")
    multiply_tool = create_tool(MultiplyToolSchema, dependencies={})
    multiply_tool.name = "multiply"
    multiply_tool.description = "Multiplies two numbers"
    agent = Agent(llm=llm, tools=[multiply_tool], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        turn1 = json.dumps(
            {
                "reasoning": "Need to multiply",
                "tool_calls": [{"id": "call_abc", "tool": "multiply", "args": {"x": 7, "y": 6}}],
                "result": None,
                "is_finished": False,
            }
        )

        turn2 = json.dumps(
            {
                "reasoning": "Done",
                "tool_calls": None,
                "result": "The answer is 42",
                "is_finished": True,
            }
        )

        mock_call.side_effect = [
            Message(
                role="assistant",
                content=turn1,
                timestamp=0.0,
                tokens_in=10,
                tokens_out=20,
                tool_calls=[ToolCall(id="call_abc", tool="multiply", args={"x": 7, "y": 6})],
            ),
            Message(role="assistant", content=turn2, timestamp=0.0, tokens_in=10, tokens_out=20),
        ]

        agent.run("What is 7 * 6?")

        # Find the assistant message with tool call
        assistant_msgs_with_tools = [
            msg for msg in agent.messages if msg.role == "assistant" and msg.tool_calls is not None
        ]
        assert len(assistant_msgs_with_tools) == 1
        tool_calls = assistant_msgs_with_tools[0].tool_calls
        assert tool_calls is not None
        assert tool_calls[0].id == "call_abc"

        # Find the tool result message
        tool_msgs = [msg for msg in agent.messages if msg.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "call_abc"
        assert tool_msgs[0].content == "42"


def test_state_persists_across_reset_false_runs():
    """Messages and turn count should persist when reset=False"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    agent = Agent(llm=llm, tools=[], max_turns=10)

    with patch.object(llm, "call") as mock_call:
        response_template = {
            "reasoning": "Answering",
            "tool_calls": None,
            "result": "Response",
            "is_finished": True,
        }

        mock_call.return_value = Message(
            role="assistant",
            content=json.dumps(response_template),
            timestamp=0.0,
            tokens_in=10,
            tokens_out=20,
        )

        # Run 1
        agent.run("First question")
        messages_after_run1 = len(agent.messages)
        tokens_after_run1 = agent.tokens_used

        # Run 2 with reset=False
        agent.run("Second question", reset=False)

        # Messages should accumulate
        assert len(agent.messages) > messages_after_run1

        # Tokens should accumulate
        assert agent.tokens_used > tokens_after_run1

        # Turn count should accumulate
        assert agent.turn_count == 2

        # Run 3 with reset=True (default)
        agent.run("Third question")

        # Should reset
        assert agent.turn_count == 1
        assert agent.tokens_used == 30  # Just the last run


if __name__ == "__main__":
    print("=== Token Tracking Tests ===\n")

    test_tokens_accumulate_across_turns()
    print("✓ Tokens accumulate across turns")

    test_tokens_returned_in_result_metadata()
    print("✓ Tokens returned in Result metadata")

    print("\n=== State Management Tests ===\n")

    test_messages_contains_full_trajectory()
    print("✓ Messages contains full trajectory")

    test_trajectory_includes_tool_calls_and_results()
    print("✓ Trajectory includes tool calls and results")

    test_state_persists_across_reset_false_runs()
    print("✓ State persists across reset=False runs")

    print("\nAll state management tests passed!")
