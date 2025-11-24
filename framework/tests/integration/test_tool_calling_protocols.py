"""
Test tool calling protocol compatibility across models.

This test documents which models use which protocol:
- Native OpenAI format (tool_calls in response)
- JSON-in-content format (tool_calls in JSON string)

When adding new models, update MODEL_PROTOCOLS registry.
"""

import json
import time

import pytest

from agentic.framework.config import get_config
from agentic.framework.llm import LLM
from agentic.framework.messages import Message

# Model Protocol Registry
# This documents which protocol each model natively uses
# Updated as we discover model capabilities
MODEL_PROTOCOLS = {
    "google/gemini-2.5-flash": "JSON_IN_CONTENT",
    "google/gemini-2.0-flash-lite-001": "JSON_IN_CONTENT",
    "x-ai/grok-code-fast-1": "NATIVE_OPENAI",
    "x-ai/grok-4.1-fast": "NATIVE_OPENAI",
    "anthropic/claude-haiku-4.5": "JSON_IN_CONTENT",  # Via OpenRouter
    "qwen/qwen3-coder-30b-a3b-instruct": "JSON_IN_CONTENT",
}


def detect_tool_calling_protocol(model_name: str, api_key: str) -> str:
    """
    Detect which tool calling protocol a model uses.

    After protocol conversion in llm.py, all models should return JSON_IN_CONTENT.
    This test verifies the conversion works correctly.

    Returns:
        - "JSON_IN_CONTENT": Successfully returns JSON with tool_calls
        - "ERROR: ...": Something went wrong
    """
    llm = LLM(model_name=model_name, api_key=api_key, temperature=0.0, max_tokens=200)

    # Test with a simple tool-requiring task
    messages = [
        Message(
            role="system",
            content="""You are a helpful agent that can use tools.

Tool Name: get_weather
Tool Description: Get current weather for a location
Tool Arguments: {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}

Respond with JSON: {"reasoning": "...", "tool_calls": [{"id": "call_1", "tool": "get_weather", "args": {...}}], "result": null, "is_finished": false}""",
            timestamp=time.time(),
        ),
        Message(role="user", content="What's the weather in London?", timestamp=time.time()),
    ]

    try:
        response = llm.call(messages)

        # After protocol conversion, should always be JSON
        try:
            parsed = json.loads(response.content)
            if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                return "JSON_IN_CONTENT"
            return f"UNEXPECTED_FORMAT: {response.content[:100]}"
        except json.JSONDecodeError as e:
            return f"JSON_PARSE_ERROR: {str(e)}"

    except Exception as e:
        return f"ERROR: {str(e)[:100]}"


@pytest.mark.parametrize("model_name", MODEL_PROTOCOLS.keys())
def test_protocol_conversion_works(model_name):
    """
    Verify that protocol conversion works for all models.

    After conversion, all models should return JSON_IN_CONTENT format,
    regardless of their native protocol.
    """
    config = get_config()
    result = detect_tool_calling_protocol(model_name, config.api_key)

    assert result == "JSON_IN_CONTENT", (
        f"Model {model_name} failed protocol conversion. "
        f"Native protocol: {MODEL_PROTOCOLS[model_name]}, "
        f"Result: {result}"
    )


def test_native_openai_models_documented():
    """
    Document which models use native OpenAI format.

    This helps understand which models benefit from protocol conversion.
    """
    native_models = [m for m, p in MODEL_PROTOCOLS.items() if p == "NATIVE_OPENAI"]
    json_models = [m for m, p in MODEL_PROTOCOLS.items() if p == "JSON_IN_CONTENT"]

    print(f"\n{'=' * 70}")
    print("Tool Calling Protocol Distribution")
    print(f"{'=' * 70}")
    print(f"\nNative OpenAI Format ({len(native_models)} models):")
    for m in native_models:
        print(f"  - {m}")
    print(f"\nJSON-in-Content Format ({len(json_models)} models):")
    for m in json_models:
        print(f"  - {m}")
    print(f"{'=' * 70}\n")

    # This test always passes - it's just documentation
    assert len(native_models) + len(json_models) == len(MODEL_PROTOCOLS)


if __name__ == "__main__":
    """Manual protocol detection for updating the registry."""
    try:
        config = get_config()
    except Exception:
        print("❌ Error: No API key found. Set OPENROUTER_API_KEY in .env")
        exit(1)

    print("\n" + "=" * 70)
    print("Tool Calling Protocol Detection")
    print("=" * 70)

    for model_name in MODEL_PROTOCOLS:
        print(f"\nTesting: {model_name}")
        native_protocol = MODEL_PROTOCOLS[model_name]
        detected = detect_tool_calling_protocol(model_name, config.api_key)

        print(f"  Native Protocol: {native_protocol}")
        print(f"  After Conversion: {detected}")

        if detected == "JSON_IN_CONTENT":
            print("  ✅ Conversion successful")
        else:
            print(f"  ❌ Conversion failed: {detected}")

    print("\n" + "=" * 70)
