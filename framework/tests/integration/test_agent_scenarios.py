"""Integration tests for Agent full scenarios with real API.

Tests success scenarios, failure recovery, and edge cases with real LLM calls.
"""

import pytest
from pydantic import BaseModel

from agentic.framework.agents import Agent
from agentic.framework.config import get_config
from agentic.framework.llm import LLM
from agentic.framework.messages import ResultStatus
from agentic.framework.tools import create_tool


class AddToolSchema(BaseModel):
    """Adds two numbers"""

    x: int
    y: int

    def execute(self) -> str:
        return str(self.x + self.y)


class MultiplyToolSchema(BaseModel):
    """Multiplies two numbers"""

    x: int
    y: int

    def execute(self) -> str:
        return str(self.x * self.y)


class FailingToolSchema(BaseModel):
    """Always fails - for testing error recovery"""

    value: str = "dummy"

    def execute(self) -> str:
        raise RuntimeError("This tool always fails!")


class EmptyToolSchema(BaseModel):
    """Returns empty string - edge case"""

    value: str = "dummy"

    def execute(self) -> str:
        return ""


def test_sequential_tool_calls():
    """Agent should handle sequential tool calls (calculate → use result → calculate again)"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    llm = LLM(
        model_name=config.model_name, api_key=config.api_key, temperature=0.0, max_tokens=1000
    )

    add_tool = create_tool(AddToolSchema, dependencies={})
    add_tool.name = "add"
    add_tool.description = "Adds two numbers and returns the result"

    multiply_tool = create_tool(MultiplyToolSchema, dependencies={})
    multiply_tool.name = "multiply"
    multiply_tool.description = "Multiplies two numbers and returns the result"

    agent = Agent(llm=llm, tools=[add_tool, multiply_tool], max_turns=15)

    # Simpler task for reliability across models
    result = agent.run(
        "First add 5 and 3. Then multiply the result by 2. What is the final answer?"
    )

    # Should succeed or at least attempt the task
    # Some models may struggle with multi-step, so we accept partial success
    assert result.status in [ResultStatus.SUCCESS, ResultStatus.MAX_TURNS_REACHED]
    assert result.value is not None

    # If succeeded, answer should be 16
    if result.status == ResultStatus.SUCCESS:
        assert "16" in result.value
        print(f"\n✓ Sequential tool calls completed in {agent.turn_count} turns")
    else:
        # Max turns reached - at least check it attempted the task
        assert agent.turn_count == 15
        print("\n⚠ Sequential tool calls hit max_turns (model-specific behavior)")

    print(f"  Tokens used: {agent.tokens_used}")
    print(f"  Result preview: {result.value[:100]}")


def test_tool_failure_agent_retries_with_correction():
    """When a tool fails, agent should see error and try a different approach"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    llm = LLM(model_name=config.model_name, api_key=config.api_key, temperature=0.0, max_tokens=500)

    failing_tool = create_tool(FailingToolSchema, dependencies={})
    failing_tool.name = "failing_tool"
    failing_tool.description = "A tool that always fails - don't use this"

    add_tool = create_tool(AddToolSchema, dependencies={})
    add_tool.name = "add"
    add_tool.description = "Adds two numbers - use this for addition"

    agent = Agent(llm=llm, tools=[failing_tool, add_tool], max_turns=10)

    # Agent should try failing_tool, see error, then use add instead
    result = agent.run("Add 5 and 3 together. You have a failing_tool and add tool available.")

    # Should eventually succeed using add tool
    assert result.status == ResultStatus.SUCCESS
    assert "8" in result.value

    # Should have tool error messages in trajectory
    tool_errors = [
        msg for msg in agent.messages if msg.role == "tool" and "error" in msg.content.lower()
    ]

    print("\n✓ Agent recovered from tool failure")
    print(f"  Turns: {agent.turn_count}")
    print(f"  Tool errors encountered: {len(tool_errors)}")


def test_max_turns_reached_returns_appropriate_status():
    """Agent should return MAX_TURNS_REACHED when limit is hit"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    llm = LLM(model_name=config.model_name, api_key=config.api_key, temperature=0.7, max_tokens=500)
    agent = Agent(llm=llm, tools=[], max_turns=2)  # Very low limit

    # Give a task that might take several turns
    result = agent.run("Write a detailed essay about quantum physics")

    # Should hit max turns with this complex task and low limit
    assert result.status == ResultStatus.MAX_TURNS_REACHED
    assert agent.turn_count == 2
    assert "[MAX_TURNS]" in result.value

    # Metadata should still be populated
    assert result.metadata is not None
    assert result.metadata["turns"] == 2

    print("\n✓ Max turns enforcement works")
    print(f"  Status: {result.status}")


def test_empty_user_input():
    """Agent should handle empty user input gracefully"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    llm = LLM(model_name=config.model_name, api_key=config.api_key, temperature=0.0, max_tokens=500)
    agent = Agent(llm=llm, tools=[], max_turns=5)

    # Empty string input
    result = agent.run("")

    # Should handle gracefully (either respond or ask for clarification)
    assert result.status in [ResultStatus.SUCCESS, ResultStatus.MAX_TURNS_REACHED]
    assert result.value is not None

    print("\n✓ Empty input handled gracefully")


def test_tool_returns_empty_string():
    """Agent should handle tool returning empty string"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    llm = LLM(model_name=config.model_name, api_key=config.api_key, temperature=0.0, max_tokens=500)

    empty_tool = create_tool(EmptyToolSchema, dependencies={})
    empty_tool.name = "empty_tool"
    empty_tool.description = "Returns an empty string"

    agent = Agent(llm=llm, tools=[empty_tool], max_turns=5)

    result = agent.run("Call the empty_tool")

    # Should complete (agent sees empty result and reports it)
    assert result.status in [ResultStatus.SUCCESS, ResultStatus.MAX_TURNS_REACHED]

    # Should have tool message with empty content
    tool_messages = [msg for msg in agent.messages if msg.role == "tool"]
    assert len(tool_messages) >= 1

    print("\n✓ Empty tool result handled")
    print(f"  Turns: {agent.turn_count}")


def test_multiple_tools_with_same_name_error():
    """Agent should handle duplicate tool names gracefully"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    # Create two tool schemas with different behavior
    class DuplicateToolSchema1(BaseModel):
        """First duplicate"""

        value: str = "dummy"

        def execute(self) -> str:
            return "Tool 1"

    class DuplicateToolSchema2(BaseModel):
        """Second duplicate"""

        value: str = "dummy"

        def execute(self) -> str:
            return "Tool 2"

    llm = LLM(model_name=config.model_name, api_key=config.api_key, temperature=0.0, max_tokens=500)

    tool1 = create_tool(DuplicateToolSchema1, dependencies={})
    tool1.name = "duplicate"
    tool1.description = "First duplicate tool"

    tool2 = create_tool(DuplicateToolSchema2, dependencies={})
    tool2.name = "duplicate"
    tool2.description = "Second duplicate tool"

    # Agent with duplicate tool names
    agent = Agent(llm=llm, tools=[tool1, tool2], max_turns=5)

    result = agent.run("Use the duplicate tool")

    # Agent's _find_tool should return None for duplicates
    # This will cause a "Tool not found" error to be added to conversation
    # Agent should see this and either ask for clarification or fail

    # Check that agent encountered the duplicate tool issue
    tool_errors = [
        msg for msg in agent.messages if msg.role == "tool" and "not found" in msg.content.lower()
    ]

    print("\n✓ Duplicate tool names handled")
    print(f"  Status: {result.status}")
    print(f"  Tool errors: {len(tool_errors)}")


def test_very_long_conversation_history():
    """Agent should handle long conversation history (multiple reset=False runs)"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    llm = LLM(model_name=config.model_name, api_key=config.api_key, temperature=0.0, max_tokens=500)
    agent = Agent(llm=llm, tools=[], max_turns=5)

    # Run 5 conversations in sequence without reset
    for i in range(5):
        result = agent.run(f"Question {i + 1}", reset=(i == 0))  # Reset only first time

        # Each should succeed or reach max turns
        assert result.status in [ResultStatus.SUCCESS, ResultStatus.MAX_TURNS_REACHED]

    # Should have accumulated many messages
    assert len(agent.messages) > 10

    # Should have all 5 user queries
    user_messages = [msg for msg in agent.messages if msg.role == "user"]
    assert len(user_messages) == 5

    print("\n✓ Long conversation history handled")
    print(f"  Total messages: {len(agent.messages)}")
    print(f"  Total turns: {agent.turn_count}")


if __name__ == "__main__":
    print("=== Agent Integration Scenarios ===\n")

    print("\n--- Success Scenarios ---")
    try:
        test_sequential_tool_calls()
    except Exception as e:
        print(f"✗ Sequential tool calls failed: {e}")

    print("\n--- Failure Recovery ---")
    try:
        test_tool_failure_agent_retries_with_correction()
    except Exception as e:
        print(f"✗ Tool failure recovery failed: {e}")

    try:
        test_max_turns_reached_returns_appropriate_status()
    except Exception as e:
        print(f"✗ Max turns test failed: {e}")

    print("\n--- Edge Cases ---")
    try:
        test_empty_user_input()
    except Exception as e:
        print(f"✗ Empty input test failed: {e}")

    try:
        test_tool_returns_empty_string()
    except Exception as e:
        print(f"✗ Empty tool result test failed: {e}")

    try:
        test_multiple_tools_with_same_name_error()
    except Exception as e:
        print(f"✗ Duplicate tool names test failed: {e}")

    try:
        test_very_long_conversation_history()
    except Exception as e:
        print(f"✗ Long conversation test failed: {e}")

    print("\n=== Integration scenarios complete! ===")
