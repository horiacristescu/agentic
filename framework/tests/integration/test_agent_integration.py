"""Integration tests for Agent with real LLM calls.

These tests validate Agent behavior against real OpenRouter/OpenAI API.
They ensure our error handling works end-to-end, not just in mocked scenarios.
"""

import pytest

from agentic.agents.calculator.tools import CalculatorTool
from agentic.framework.agents import Agent
from agentic.framework.config import get_config
from agentic.framework.errors import InvalidModelError
from agentic.framework.llm import LLM
from agentic.framework.messages import ResultStatus
from agentic.framework.tools import create_tool


def test_agent_runs_simple_task_successfully():
    """Agent should successfully run a simple task with real API"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    # Create agent with calculator tool
    llm = LLM(model_name=config.model_name, api_key=config.api_key, temperature=0.0, max_tokens=500)
    calculator_tool = create_tool(CalculatorTool, dependencies={})
    agent = Agent(llm=llm, tools=[calculator_tool], max_turns=5)

    # Run simple calculation task
    result = agent.run("What is 15 + 27?")

    # Should succeed
    assert result.status == ResultStatus.SUCCESS
    assert result.value is not None, "Expected non-None value in successful result"
    assert "42" in result.value  # Answer should contain 42
    assert agent.turn_count >= 1  # Should have run at least one turn

    print("\n✓ Agent successfully ran simple task with real API")
    print(f"  Turns: {agent.turn_count}")
    print(f"  Tokens used: {agent.tokens_used}")
    print(f"  Result: {result.value[:100]}")


def test_agent_handles_invalid_model_error():
    """Agent should crash fast with InvalidModelError for bad model name"""
    try:
        config = get_config()
    except Exception:
        pytest.skip("No API key - set OPENROUTER_API_KEY in .env")

    # Create agent with invalid model
    llm = LLM(model_name="nonexistent/fake-model-999", api_key=config.api_key)
    agent = Agent(llm=llm, tools=[], max_turns=5)

    # Should raise InvalidModelError (Category A - crash fast)
    with pytest.raises(InvalidModelError) as exc_info:
        agent.run("Hello")

    assert "nonexistent/fake-model" in str(exc_info.value) or "not" in str(exc_info.value).lower()

    print("\n✓ Agent correctly raised InvalidModelError for bad model")
    print(f"  Error: {str(exc_info.value)[:100]}")


if __name__ == "__main__":
    print("=== Agent Integration Tests ===\n")

    try:
        test_agent_runs_simple_task_successfully()
    except Exception as e:
        print(f"✗ Simple task test failed: {e}")

    try:
        test_agent_handles_invalid_model_error()
    except Exception as e:
        print(f"✗ Invalid model test failed: {e}")

    print("\nAgent integration tests complete!")
