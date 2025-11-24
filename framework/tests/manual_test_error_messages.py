"""Manual test to review error messages that the agent will see.

Run this to verify error messages are clear and actionable for LLM agents.
"""

from framework.tools import create_tool
from tools.calculator_tool import CalculatorTool


def print_separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    tool = create_tool(CalculatorTool, dependencies={})

    print_separator("SCENARIO 1: Missing Required Field")
    print("Agent forgets to include 'operation' field")
    print("\nInput: {'x': 5, 'y': 3}")
    result = tool.run({"x": 5, "y": 3})
    print(f"\nError Code: {result.error_code}")
    print(f"Message to Agent:\n{result.content}\n")

    print_separator("SCENARIO 2: Wrong Type for Field")
    print("Agent provides string instead of integer")
    print("\nInput: {'operation': 'add', 'x': 'five', 'y': 3}")
    result = tool.run({"operation": "add", "x": "five", "y": 3})
    print(f"\nError Code: {result.error_code}")
    print(f"Message to Agent:\n{result.content}\n")

    print_separator("SCENARIO 3: Invalid Enum Value")
    print("Agent uses operation that doesn't exist")
    print("\nInput: {'operation': 'power', 'x': 2, 'y': 3}")
    result = tool.run({"operation": "power", "x": 2, "y": 3})
    print(f"\nError Code: {result.error_code}")
    print(f"Message to Agent:\n{result.content}\n")

    print_separator("SCENARIO 4: Multiple Validation Errors")
    print("Agent makes multiple mistakes at once")
    print("\nInput: {'operation': 'invalid', 'x': 'bad', 'y': 'also_bad'}")
    result = tool.run({"operation": "invalid", "x": "bad", "y": "also_bad"})
    print(f"\nError Code: {result.error_code}")
    print(f"Message to Agent:\n{result.content}\n")

    print_separator("SCENARIO 5: Extra Unexpected Field")
    print("Agent includes field that doesn't exist")
    print("\nInput: {'operation': 'add', 'x': 5, 'y': 3, 'z': 10}")
    result = tool.run({"operation": "add", "x": 5, "y": 3, "z": 10})
    print(f"\nError Code: {result.error_code}")
    print(f"Message to Agent:\n{result.content}\n")

    print_separator("SCENARIO 6: Empty Input")
    print("Agent provides no arguments")
    print("\nInput: {}")
    result = tool.run({})
    print(f"\nError Code: {result.error_code}")
    print(f"Message to Agent:\n{result.content}\n")

    print_separator("SCENARIO 7: Wrong Dependencies (Internal Error)")
    print("Tool configured incorrectly (not agent's fault)")
    tool_bad_deps = create_tool(CalculatorTool, dependencies={"wrong_key": "value"})
    print("\nInput: {'operation': 'add', 'x': 5, 'y': 3}")
    result = tool_bad_deps.run({"operation": "add", "x": 5, "y": 3})
    print(f"\nError Code: {result.error_code}")
    print(f"Message to Agent:\n{result.content}\n")

    print_separator("SCENARIO 8: Successful Call (for comparison)")
    print("Agent does everything correctly")
    print("\nInput: {'operation': 'add', 'x': 5, 'y': 3}")
    result = tool.run({"operation": "add", "x": 5, "y": 3})
    print(f"\nError Code: {result.error_code}")
    print(f"Message to Agent:\n{result.content}\n")

    print("=" * 70)
    print("\nREVIEW QUESTIONS:")
    print("1. Can the agent understand what went wrong?")
    print("2. Does the message tell the agent how to fix it?")
    print("3. Are the messages concise enough for context window?")
    print("4. Do they avoid technical jargon the agent doesn't need?")
    print("=" * 70)


if __name__ == "__main__":
    main()
