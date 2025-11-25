"""Comprehensive tests for LLM response parsing.

This documents every absurd response format we've encountered from various models
and ensures our parser can handle them all gracefully.
"""

import json

from agentic.framework.llm import LLM


def clean_response(response: str) -> str:
    """Test helper: directly call the parser without creating a full agent."""
    llm = LLM(model_name="mock", api_key="fake")
    return llm._clean_markdown_response(response)


# Valid baseline JSON (what we expect)
VALID_JSON = json.dumps(
    {
        "reasoning": "I need to solve this problem",
        "tool_calls": None,
        "result": "The answer is 42",
        "is_finished": True,
    }
)


class TestCleanJSONBaseline:
    """Test that clean, well-formatted JSON parses correctly."""

    def test_perfect_json(self):
        """Baseline: Clean JSON with no extras."""
        result = clean_response(VALID_JSON)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_json_with_whitespace(self):
        """JSON with surrounding whitespace."""
        result = clean_response(f"\n\n  {VALID_JSON}  \n\n")
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"


class TestMarkdownCodeBlocks:
    """Test extraction from markdown code blocks (common in GPT-4, Claude)."""

    def test_json_in_markdown_block(self):
        """JSON wrapped in ```json ... ```"""
        markdown = f"```json\n{VALID_JSON}\n```"
        result = clean_response(markdown)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_json_in_plain_code_block(self):
        """JSON wrapped in ``` ... ``` (no language tag)."""
        markdown = f"```\n{VALID_JSON}\n```"
        result = clean_response(markdown)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_json_with_explanation_before(self):
        """Text before the code block."""
        markdown = f"Here's my response:\n```json\n{VALID_JSON}\n```"
        result = clean_response(markdown)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_json_with_explanation_after(self):
        """Text after the code block (should be ignored)."""
        markdown = f"```json\n{VALID_JSON}\n```\nThat's my final answer."
        result = clean_response(markdown)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"


class TestCommonPreambles:
    """Test stripping of common preambles models add before JSON."""

    def test_assistant_prefix(self):
        """Grok failure case: '\n\nAssistant: '"""
        response = f"\n\nAssistant: {VALID_JSON}"
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_here_is_the_json(self):
        """Models saying 'Here is the JSON:'"""
        response = f"Here is the JSON:\n{VALID_JSON}"
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_here_are_the_results(self):
        """Variation: 'Here are the results:'"""
        response = f"Here are the results:\n{VALID_JSON}"
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_response_colon(self):
        """Simple 'Response:' prefix."""
        response = f"Response: {VALID_JSON}"
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_output_prefix(self):
        """'Output:' prefix."""
        response = f"Output:\n{VALID_JSON}"
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"


class TestAnthropicXMLFormat:
    """Test Anthropic's <function_calls> XML format."""

    def test_anthropic_function_calls(self):
        """Anthropic's XML wrapper for tool calls."""
        response = """
I need to use tools.
<function_calls>
[
  {"id": "call_1", "tool": "calculator", "args": {"x": 2, "y": 3}}
]
</function_calls>
"""
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["reasoning"] == "I need to use tools."
        assert parsed["tool_calls"][0]["tool"] == "calculator"

    def test_anthropic_empty_reasoning(self):
        """Anthropic XML with no reasoning text before it."""
        response = """
<function_calls>
[
  {"id": "call_1", "tool": "calculator", "args": {"x": 2, "y": 3}}
]</function_calls>
"""
        result = clean_response(response)
        parsed = json.loads(result)
        # Should add default reasoning
        assert "Calling tools" in parsed["reasoning"]


class TestVerboseXMLToolCallFormat:
    """Test verbose XML tool call format (<tool_call><function=name>...)."""

    def test_single_tool_call(self):
        """Single tool call with parameters."""
        response = """
<tool_call>
<function=calculator>
<parameter=operation>
add
</parameter>
<parameter=x>
5022
</parameter>
<parameter=y>
11075
</parameter>
</function>
</tool_call>
"""
        result = clean_response(response)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["tool"] == "calculator"
        assert parsed["tool_calls"][0]["args"]["operation"] == "add"
        assert parsed["tool_calls"][0]["args"]["x"] == 5022
        assert parsed["tool_calls"][0]["args"]["y"] == 11075
        assert parsed["tool_calls"][0]["id"] == "call_1"

    def test_multiple_tool_calls(self):
        """Multiple tool calls in sequence."""
        response = """
<tool_call>
<function=calculator>
<parameter=operation>add</parameter>
<parameter=x>5022</parameter>
<parameter=y>11075</parameter>
</function>
</tool_call>
<tool_call>
<function=calculator>
<parameter=operation>add</parameter>
<parameter=x>14610</parameter>
<parameter=y>8264</parameter>
</function>
</tool_call>
"""
        result = clean_response(response)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 2
        assert parsed["tool_calls"][0]["tool"] == "calculator"
        assert parsed["tool_calls"][0]["args"]["x"] == 5022
        assert parsed["tool_calls"][1]["tool"] == "calculator"
        assert parsed["tool_calls"][1]["args"]["x"] == 14610
        assert parsed["tool_calls"][0]["id"] == "call_1"
        assert parsed["tool_calls"][1]["id"] == "call_2"

    def test_with_reasoning_text(self):
        """XML format with reasoning text before tool calls."""
        response = """
I need to calculate the sum of these numbers.
<tool_call>
<function=calculator>
<parameter=operation>add</parameter>
<parameter=x>5</parameter>
<parameter=y>3</parameter>
</function>
</tool_call>
"""
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["reasoning"] == "I need to calculate the sum of these numbers."
        assert len(parsed["tool_calls"]) == 1

    def test_numeric_parameter_conversion(self):
        """Parameters should be converted to numbers when possible."""
        response = """
<tool_call>
<function=calculator>
<parameter=operation>multiply</parameter>
<parameter=x>10</parameter>
<parameter=y>3.5</parameter>
</function>
</tool_call>
"""
        result = clean_response(response)
        parsed = json.loads(result)
        args = parsed["tool_calls"][0]["args"]
        assert args["x"] == 10  # int
        assert args["y"] == 3.5  # float
        assert args["operation"] == "multiply"  # string

    def test_empty_reasoning(self):
        """XML format with no reasoning text."""
        response = """
<tool_call>
<function=calculator>
<parameter=operation>add</parameter>
<parameter=x>1</parameter>
<parameter=y>2</parameter>
</function>
</tool_call>
"""
        result = clean_response(response)
        parsed = json.loads(result)
        # Should add default reasoning
        assert "Calling tools" in parsed["reasoning"]
        assert len(parsed["tool_calls"]) == 1


class TestMalformedJSON:
    """Test handling of common JSON malformations."""

    def test_trailing_comma_in_array(self):
        """Trailing comma in JSON array (common mistake)."""
        malformed = (
            '{"reasoning": "test", "tool_calls": [1, 2,], "result": null, "is_finished": true}'
        )
        # This will fail - JSON doesn't allow trailing commas
        # But we should at least extract the string
        result = clean_response(malformed)
        assert result.startswith("{")

    def test_json_with_trailing_text(self):
        """Valid JSON followed by garbage."""
        response = f"{VALID_JSON}\n\nHope that helps!"
        result = clean_response(response)
        # Should extract just the valid JSON part
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_multiple_json_objects(self):
        """Two JSON objects (should take the first)."""
        response = '{"first": "value"}\n{"second": "value"}'
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["first"] == "value"
        assert "second" not in parsed


class TestEdgeCases:
    """Test bizarre edge cases we've encountered."""

    def test_json_in_middle_of_prose(self):
        """JSON buried in conversational text."""
        response = f"""
Let me think about this problem.

After careful consideration, here's my response:

{VALID_JSON}

I hope this helps!
"""
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_nested_braces_in_strings(self):
        """JSON with { } characters inside strings (tests brace balancing)."""
        tricky_json = json.dumps(
            {
                "reasoning": "The formula is {{x + y}}",
                "tool_calls": None,
                "result": "Nested braces: {{{",
                "is_finished": True,
            }
        )
        result = clean_response(tricky_json)
        parsed = json.loads(result)
        assert "{{x + y}}" in parsed["reasoning"]

    def test_escaped_quotes_in_strings(self):
        """JSON with escaped quotes (tests string tracking)."""
        tricky_json = json.dumps(
            {
                "reasoning": 'She said "Hello"',
                "tool_calls": None,
                "result": 'Escaped: "test"',
                "is_finished": True,
            }
        )
        result = clean_response(tricky_json)
        parsed = json.loads(result)
        assert '"Hello"' in parsed["reasoning"]

    def test_multiline_strings(self):
        """JSON with multiline string values."""
        multiline_json = json.dumps(
            {
                "reasoning": "Step 1\nStep 2\nStep 3",
                "tool_calls": None,
                "result": "Line 1\nLine 2",
                "is_finished": True,
            }
        )
        result = clean_response(multiline_json)
        parsed = json.loads(result)
        assert "Step 1\nStep 2\nStep 3" in parsed["reasoning"]

    def test_empty_response(self):
        """Empty string response (model failure)."""
        result = clean_response("")
        assert result == ""  # Should return as-is, will fail parsing later

    def test_only_preamble_no_json(self):
        """Just 'Assistant:' with no actual JSON (Grok's failure)."""
        result = clean_response("\n\nAssistant: ")
        # Should strip preamble, leaving empty/whitespace
        assert "{" not in result


class TestRealWorldFailures:
    """Document actual failures we've seen from specific models."""

    def test_grok_assistant_only(self):
        """Grok-code-fast-1: Returned only '\n\nAssistant: ' on turn 2."""
        response = "\n\nAssistant: "
        result = clean_response(response)
        # Should strip to empty/whitespace
        assert result.strip() == ""

    def test_gemini_verbose_preamble(self):
        """Gemini sometimes adds verbose explanations."""
        response = f"""
Okay, I understand. I will now provide my response in the required JSON format.

Here is the JSON response:

{VALID_JSON}

This response includes my reasoning and the final result.
"""
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"

    def test_claude_conversational_wrapper(self):
        """Claude adding conversational wrappers."""
        response = f"""
I'll help you with that. Let me structure my response:

```json
{VALID_JSON}
```

Does this answer your question?
"""
        result = clean_response(response)
        parsed = json.loads(result)
        assert parsed["result"] == "The answer is 42"
