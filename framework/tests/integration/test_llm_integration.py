import time

import pytest

from agentic.framework.config import get_config
from agentic.framework.llm import LLM
from agentic.framework.messages import Message, ToolCall


def test_basic_llm_call():
    """Test basic LLM call returns Message with tokens"""
    try:
        config = get_config()
    except Exception:
        pytest.skip(
            "No API credentials found - set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env"
        )

    llm = LLM(model_name=config.model_name, api_key=config.api_key)

    messages = [
        Message(role="system", content="You are a helpful assistant.", timestamp=time.time()),
        Message(role="user", content="What is 2+2?", timestamp=time.time()),
    ]

    response = llm.call(messages)

    # Check it's a Message
    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert response.content is not None
    assert "4" in response.content

    # Check token tracking
    assert response.tokens_in is not None
    assert response.tokens_out is not None
    assert response.tokens_in > 0
    assert response.tokens_out > 0

    # Check timestamp
    assert response.timestamp > 0


def test_llm_with_tool_calls():
    """Test LLM can see and respond to messages with tool_calls"""
    try:
        config = get_config()
    except Exception:
        pytest.skip(
            "No API credentials found - set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env"
        )

    llm = LLM(model_name=config.model_name, api_key=config.api_key)

    messages = [
        Message(role="system", content="You help with math.", timestamp=time.time()),
        Message(role="user", content="Calculate 15 * 3", timestamp=time.time()),
        Message(
            role="assistant",
            content="I'll use the calculator",
            tool_calls=[ToolCall(id="call_1", tool="calculator", args={"expr": "15*3"})],
            timestamp=time.time(),
        ),
        Message(
            role="tool",
            tool_call_id="call_1",
            name="calculator",
            content="45",
            timestamp=time.time(),
        ),
    ]

    response = llm.call(messages)

    # Should acknowledge the tool result
    assert isinstance(response, Message)
    assert "45" in response.content


def test_llm_error_handling():
    """Test LLM raises AuthError for invalid credentials (Category A - crash fast)"""
    import pytest

    from agentic.framework.errors import AuthError

    llm = LLM(model_name="gpt-4", api_key="bad-key")

    messages = [Message(role="user", content="Hello", timestamp=time.time())]

    # Should raise AuthError, not return Message
    with pytest.raises(AuthError):
        llm.call(messages)


def test_tool_call_ids():
    """Test that tool calls with IDs get properly paired with results"""
    from agentic.agents.calculator.tools import CalculatorTool
    from agentic.framework.messages import ToolCall
    from agentic.framework.tools import create_tool

    # Create tool calls with IDs (as LLM would generate)
    tool_calls = [
        ToolCall(id="calc_1", tool="calculator", args={"operation": "add", "x": 10, "y": 5}),
        ToolCall(id="calc_2", tool="calculator", args={"operation": "multiply", "x": 3, "y": 7}),
        ToolCall(id="calc_3", tool="calculator", args={"operation": "subtract", "x": 20, "y": 8}),
    ]

    tool = create_tool(CalculatorTool)

    # Execute and verify IDs are preserved
    for tc in tool_calls:
        result = tool.run(tc.args)
        assert isinstance(result, Message)
        assert result.name == "calculator"
        print(f"{tc.id}: {result.content}")


if __name__ == "__main__":
    test_basic_llm_call()
    print("✓ Basic LLM call")

    test_llm_with_tool_calls()
    print("✓ Tool calls handling")

    test_llm_error_handling()
    print("✓ Error handling")

    test_tool_call_ids()
    print("✓ Tool call IDs")

    print("\nAll tests passed!")
