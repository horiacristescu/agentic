"""Tests for tool error message quality.

Validates that error messages contain the right information for agents to recover.
"""

from pydantic import BaseModel

from agentic.agents.calculator.tools import CalculatorTool
from agentic.framework.messages import Message
from agentic.framework.tools import create_tool


def test_missing_field_error_is_clear():
    """Missing field error should mention field name and 'required'"""
    tool = create_tool(CalculatorTool, dependencies={})
    result = tool.run({"x": 5, "y": 3})  # Missing 'operation'

    assert isinstance(result, Message)
    assert result.error_code == "validation_error"
    # Should mention the tool name
    assert "calculator" in result.content.lower()
    # Should mention the field name
    assert "operation" in result.content.lower()
    # Should say it's required
    assert "required" in result.content.lower()


def test_wrong_type_error_mentions_field_and_type():
    """Wrong type error should mention field name and expected type"""
    tool = create_tool(CalculatorTool, dependencies={})
    result = tool.run({"operation": "add", "x": "not_a_number", "y": 3})

    assert isinstance(result, Message)
    assert result.error_code == "validation_error"
    # Should mention the tool name
    assert "calculator" in result.content.lower()
    # Should mention the field name
    assert "x" in result.content.lower()
    # Should mention expected type (integer in this case)
    assert "integer" in result.content.lower() or "number" in result.content.lower()


def test_invalid_enum_shows_valid_options():
    """Invalid enum error should show what values are valid"""
    tool = create_tool(CalculatorTool, dependencies={})
    result = tool.run({"operation": "power", "x": 2, "y": 3})

    assert isinstance(result, Message)
    assert result.error_code == "validation_error"
    # Should mention the field name
    assert "operation" in result.content.lower()
    # Should show valid options
    assert "add" in result.content.lower()
    assert "subtract" in result.content.lower()
    assert "multiply" in result.content.lower()
    assert "divide" in result.content.lower()


def test_multiple_errors_lists_all_problems():
    """When multiple fields are wrong, should list all of them"""
    tool = create_tool(CalculatorTool, dependencies={})
    result = tool.run({"operation": "invalid", "x": "bad", "y": "also_bad"})

    assert isinstance(result, Message)
    assert result.error_code == "validation_error"
    # Should mention all three problematic fields
    assert "operation" in result.content.lower()
    assert "'x'" in result.content.lower()
    assert "'y'" in result.content.lower()
    # Should show valid operations
    assert "add" in result.content.lower() or "subtract" in result.content.lower()


def test_empty_input_lists_all_required_fields():
    """Empty input should list all required fields"""
    tool = create_tool(CalculatorTool, dependencies={})
    result = tool.run({})

    assert isinstance(result, Message)
    assert result.error_code == "validation_error"
    # Should mention all required fields
    assert "operation" in result.content.lower()
    assert "'x'" in result.content.lower()
    assert "'y'" in result.content.lower()
    # Should say they're required
    assert "required" in result.content.lower()


def test_dependency_mismatch_mentions_tool_and_issue():
    """Dependency errors should mention tool name and what's wrong"""
    tool = create_tool(CalculatorTool, dependencies={"wrong_dep": "value"})
    result = tool.run({"operation": "add", "x": 5, "y": 3})

    assert isinstance(result, Message)
    assert result.error_code == "execution_error"
    # Should mention the tool name
    assert "calculator" in result.content.lower()
    # Should indicate dependency issue
    assert "dependency" in result.content.lower() or "unexpected" in result.content.lower()


def test_execution_error_includes_exception_message():
    """Generic execution errors should include the actual error message"""

    class BrokenTool(BaseModel):
        """Tool that crashes with custom message"""

        value: int

        def execute(self) -> str:
            raise ValueError("Custom error: database connection failed")

    tool = create_tool(BrokenTool, dependencies={})
    result = tool.run({"value": 42})

    assert isinstance(result, Message)
    assert result.error_code == "execution_error"
    # Should include the custom error message
    assert "Custom error" in result.content or "database connection failed" in result.content


def test_success_has_no_error_code():
    """Successful execution should not have error_code"""
    tool = create_tool(CalculatorTool, dependencies={})
    result = tool.run({"operation": "add", "x": 5, "y": 3})

    assert isinstance(result, Message)
    assert result.error_code is None
    assert result.content == "8"  # Integer result (calculator returns int when no decimal)


if __name__ == "__main__":
    test_missing_field_error_is_clear()
    print("✓ Missing field error is clear")

    test_wrong_type_error_mentions_field_and_type()
    print("✓ Wrong type error mentions field and type")

    test_invalid_enum_shows_valid_options()
    print("✓ Invalid enum shows valid options")

    test_multiple_errors_lists_all_problems()
    print("✓ Multiple errors lists all problems")

    test_empty_input_lists_all_required_fields()
    print("✓ Empty input lists all required fields")

    test_dependency_mismatch_mentions_tool_and_issue()
    print("✓ Dependency mismatch mentions tool and issue")

    test_execution_error_includes_exception_message()
    print("✓ Execution error includes exception message")

    test_success_has_no_error_code()
    print("✓ Success has no error code")

    print("\nAll error message quality tests passed!")
