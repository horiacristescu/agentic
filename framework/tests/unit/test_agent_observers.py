"""Unit tests for Agent observer notifications.

Tests that all observer methods are called correctly with the right parameters.
"""

import json
from unittest.mock import Mock, patch

from pydantic import BaseModel

from agentic.framework.agents import Agent
from agentic.framework.llm import LLM
from agentic.framework.messages import Message, ResultStatus, ToolCall
from agentic.framework.tools import create_tool


def test_on_turn_start_called_correctly():
    """Observer's on_turn_start should be called at the start of each turn"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    mock_observer = Mock()
    agent = Agent(llm=llm, tools=[], observers=[mock_observer], max_turns=3)

    with patch.object(llm, "call") as mock_call:
        # Agent runs for 2 turns
        turn1 = json.dumps(
            {
                "reasoning": "First turn",
                "tool_calls": None,
                "result": None,
                "is_finished": False,
            }
        )
        turn2 = json.dumps(
            {
                "reasoning": "Second turn - done",
                "tool_calls": None,
                "result": "Final answer",
                "is_finished": True,
            }
        )

        mock_call.side_effect = [
            Message(role="assistant", content=turn1, timestamp=0.0, tokens_in=10, tokens_out=20),
            Message(role="assistant", content=turn2, timestamp=0.0, tokens_in=10, tokens_out=20),
        ]

        agent.run("Test")

        # Should have been called twice (once per turn)
        assert mock_observer.on_turn_start.call_count == 2

        # First call: turn=1
        first_call = mock_observer.on_turn_start.call_args_list[0]
        assert first_call.kwargs["turn"] == 1
        assert "messages" in first_call.kwargs

        # Second call: turn=2
        second_call = mock_observer.on_turn_start.call_args_list[1]
        assert second_call.kwargs["turn"] == 2


def test_on_llm_response_called_with_message():
    """Observer's on_llm_response should be called with Message after LLM responds"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    mock_observer = Mock()
    agent = Agent(llm=llm, tools=[], observers=[mock_observer], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        response = json.dumps(
            {
                "reasoning": "Simple answer",
                "tool_calls": None,
                "result": "Done",
                "is_finished": True,
            }
        )

        mock_call.return_value = Message(
            role="assistant",
            content=response,
            timestamp=0.0,
            tokens_in=10,
            tokens_out=20,
        )

        agent.run("Test")

        # Should have been called once
        assert mock_observer.on_llm_response.call_count == 1

        # Should receive turn number and response Message
        call_kwargs = mock_observer.on_llm_response.call_args.kwargs
        assert call_kwargs["turn"] == 1
        assert isinstance(call_kwargs["response"], Message)
        assert call_kwargs["response"].role == "assistant"


def test_on_tool_execution_called_with_results():
    """Observer's on_tool_execution should be called for each tool execution"""

    # Create a simple test tool schema
    class AddToolSchema(BaseModel):
        """Adds two numbers"""

        x: int
        y: int

        def execute(self) -> str:
            return str(self.x + self.y)

    llm = LLM(model_name="gpt-4", api_key="test-key")
    mock_observer = Mock()
    add_tool = create_tool(AddToolSchema, dependencies={})
    add_tool.name = "add"
    add_tool.description = "Adds two numbers"
    agent = Agent(llm=llm, tools=[add_tool], observers=[mock_observer], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        # Turn 1: Call tool
        turn1 = json.dumps(
            {
                "reasoning": "Need to add numbers",
                "tool_calls": [{"id": "call_1", "tool": "add", "args": {"x": 5, "y": 3}}],
                "result": None,
                "is_finished": False,
            }
        )

        # Turn 2: Finish
        turn2 = json.dumps(
            {
                "reasoning": "Got the result",
                "tool_calls": None,
                "result": "The answer is 8",
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
                tool_calls=[ToolCall(id="call_1", tool="add", args={"x": 5, "y": 3})],
            ),
            Message(role="assistant", content=turn2, timestamp=0.0, tokens_in=10, tokens_out=20),
        ]

        agent.run("What is 5 + 3?")

        # Should have been called once for the tool execution
        assert mock_observer.on_tool_execution.call_count == 1

        # Verify parameters
        call_kwargs = mock_observer.on_tool_execution.call_args.kwargs
        assert call_kwargs["turn"] == 1
        assert call_kwargs["tool_name"] == "add"
        assert isinstance(call_kwargs["result"], Message)
        assert call_kwargs["result"].role == "tool"


def test_on_finish_called_on_completion():
    """Observer's on_finish should be called when agent completes successfully"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    mock_observer = Mock()
    agent = Agent(llm=llm, tools=[], observers=[mock_observer], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        response = json.dumps(
            {
                "reasoning": "Answering",
                "tool_calls": None,
                "result": "Final answer",
                "is_finished": True,
            }
        )

        mock_call.return_value = Message(
            role="assistant",
            content=response,
            timestamp=0.0,
            tokens_in=10,
            tokens_out=20,
        )

        agent.run("Test")

        # Should have been called once
        assert mock_observer.on_finish.call_count == 1

        # Verify parameters
        call_kwargs = mock_observer.on_finish.call_args.kwargs
        assert "final_result" in call_kwargs
        assert "all_messages" in call_kwargs
        assert isinstance(call_kwargs["final_result"], Message)
        assert isinstance(call_kwargs["all_messages"], list)


def test_on_error_called_on_failures():
    """Observer's on_error should be called when errors occur"""
    llm = LLM(model_name="gpt-4", api_key="test-key")
    mock_observer = Mock()
    agent = Agent(llm=llm, tools=[], observers=[mock_observer], max_turns=3)

    with patch.object(llm, "call") as mock_call:
        # Mock LLM to never finish (triggers max_turns error)
        response = json.dumps(
            {
                "reasoning": "Still working",
                "tool_calls": None,
                "result": None,
                "is_finished": False,
            }
        )

        mock_call.return_value = Message(
            role="assistant",
            content=response,
            timestamp=0.0,
            tokens_in=10,
            tokens_out=20,
        )

        agent.run("Test")

        # Should have been called when max_turns reached
        assert mock_observer.on_error.call_count >= 1

        # Last call should be for max_turns
        last_call = mock_observer.on_error.call_args_list[-1]
        assert "Max turns" in last_call.kwargs["error"] or "MAX_TURNS" in last_call.kwargs["error"]


def test_observer_exceptions_dont_crash_agent():
    """If an observer raises an exception, agent should continue (errors suppressed)"""

    # Create observer that raises exceptions
    class CrashingObserver:
        def on_turn_start(self, **kwargs):
            raise RuntimeError("Observer crashed!")

        def on_llm_response(self, **kwargs):
            raise ValueError("Another crash!")

        def on_finish(self, **kwargs):
            raise Exception("Final crash!")

    llm = LLM(model_name="gpt-4", api_key="test-key")
    crashing_observer = CrashingObserver()
    agent = Agent(llm=llm, tools=[], observers=[crashing_observer], max_turns=5)

    with patch.object(llm, "call") as mock_call:
        response = json.dumps(
            {
                "reasoning": "Simple answer",
                "tool_calls": None,
                "result": "Done",
                "is_finished": True,
            }
        )

        mock_call.return_value = Message(
            role="assistant",
            content=response,
            timestamp=0.0,
            tokens_in=10,
            tokens_out=20,
        )

        # Agent should complete successfully despite observer crashes
        result = agent.run("Test")

        assert result.status == ResultStatus.SUCCESS
        assert result.value == "Done"


if __name__ == "__main__":
    print("=== Agent Observer Notification Tests ===\n")

    test_on_turn_start_called_correctly()
    print("✓ on_turn_start called correctly")

    test_on_llm_response_called_with_message()
    print("✓ on_llm_response called with Message")

    test_on_tool_execution_called_with_results()
    print("✓ on_tool_execution called with results")

    test_on_finish_called_on_completion()
    print("✓ on_finish called on completion")

    test_on_error_called_on_failures()
    print("✓ on_error called on failures")

    test_observer_exceptions_dont_crash_agent()
    print("✓ Observer exceptions don't crash agent")

    print("\nAll Observer notification tests passed!")
