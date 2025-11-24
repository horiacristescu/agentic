from pydantic import BaseModel

from agentic.framework.messages import ErrorCode, Message
from agentic.framework.tools import create_tool


class WeatherTool(BaseModel):
    """Get weather for a city"""

    city: str

    def execute(self, api_key: str) -> str:
        # This expects api_key dependency
        return f"Weather in {self.city}: sunny (using key {api_key})"


def test_tool_with_correct_dependencies():
    """Should work when dependencies match"""
    tool = create_tool(WeatherTool, dependencies={"api_key": "test_key_123"})
    result = tool.run({"city": "NYC"})

    # Now returns Message
    assert isinstance(result, Message)
    assert result.role == "tool"
    assert result.name == "weather"
    assert "sunny" in result.content
    assert "test_key_123" in result.content
    assert result.error_code is None


def test_tool_with_wrong_dependencies():
    """Should return error Message when dependencies don't match"""
    tool = create_tool(WeatherTool, dependencies={"wrong_key": "oops"})
    result = tool.run({"city": "NYC"})

    # Should return error Message, not raise
    assert isinstance(result, Message)
    assert result.error_code == ErrorCode.EXECUTION_ERROR
    assert "dependency mismatch" in result.content


def test_tool_with_no_dependencies():
    """Tools that don't need dependencies should work fine"""
    from agentic.agents.calculator.tools import CalculatorTool

    tool = create_tool(CalculatorTool, dependencies={})
    result = tool.run({"operation": "add", "x": 5, "y": 3})

    assert isinstance(result, Message)
    assert result.content == "8.0"  # Calculator returns float
    assert result.error_code is None


def test_tool_with_invalid_arguments():
    """Invalid arguments should return error Message, not raise"""
    from agentic.agents.calculator.tools import CalculatorTool

    tool = create_tool(CalculatorTool, dependencies={})
    # Missing required field 'operation'
    result = tool.run({"x": 5, "y": 3})

    # Should return error Message (not crash!)
    assert isinstance(result, Message)
    assert result.error_code == ErrorCode.VALIDATION_ERROR
    assert "validation" in result.content.lower() or "required" in result.content.lower()


def test_tool_execution_raises_exception():
    """Exceptions during tool execution should return error Message"""

    class BrokenTool(BaseModel):
        """A tool that always crashes"""

        value: int

        def execute(self) -> str:
            raise RuntimeError("Something went wrong!")

    tool = create_tool(BrokenTool, dependencies={})
    result = tool.run({"value": 42})

    # Should return error Message (not crash!)
    assert isinstance(result, Message)
    assert result.error_code == ErrorCode.EXECUTION_ERROR
    assert "Something went wrong!" in result.content


if __name__ == "__main__":
    test_tool_with_correct_dependencies()
    print("✓ Tool with correct dependencies")

    test_tool_with_wrong_dependencies()
    print("✓ Tool with wrong dependencies")

    test_tool_with_no_dependencies()
    print("✓ Tool with no dependencies")

    test_tool_with_invalid_arguments()
    print("✓ Tool with invalid arguments")

    test_tool_execution_raises_exception()
    print("✓ Tool execution exception")
