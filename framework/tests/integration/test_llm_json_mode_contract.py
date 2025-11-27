"""Integration tests for LLM JSON mode with real API calls.

Tests that JSON mode works correctly across different model providers
via OpenRouter. These are contract tests that verify the feature works
in production with real models.

Run with: pytest -m integration framework/tests/integration/test_llm_json_mode_contract.py
"""

import json
import os
import time

import pytest

from agentic.framework.llm import LLM
from agentic.framework.messages import ErrorCode, Message

# Models to test JSON mode with
MODELS_TO_TEST = [
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-exp:free",
]


@pytest.fixture
def api_key():
    """Get OpenRouter API key from environment."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture
def simple_json_request():
    """Create a simple request that should return valid JSON."""
    return [
        Message(
            role="user",
            content=(
                'Respond with valid JSON in this format: {"answer": "your response"}. '
                "What is 2+2?"
            ),
            timestamp=time.time(),
        )
    ]


@pytest.fixture
def structured_json_request():
    """Create a request for a more complex JSON structure."""
    return [
        Message(
            role="user",
            content=(
                "Respond with valid JSON containing:\n"
                '{"calculation": {"operation": "addition", "operands": [2, 2], "result": 4}, '
                '"explanation": "your explanation"}\n'
                "Calculate 2+2 and explain."
            ),
            timestamp=time.time(),
        )
    ]


@pytest.mark.integration
@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_json_mode_returns_valid_json(api_key, model_name, simple_json_request):
    """Test that JSON mode returns valid JSON for each model."""
    # Create LLM with JSON mode enabled
    llm = LLM(
        model_name=model_name,
        api_key=api_key,
        temperature=0.0,
        max_tokens=500,
        json_mode=True,
    )
    
    # Make API call
    response = llm.call(simple_json_request)
    
    # Should not have errors
    assert response.error_code is None, f"Got error: {response.error_code}"
    
    # Response should be valid JSON
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError as e:
        pytest.fail(f"Model {model_name} with JSON mode returned invalid JSON: {e}\nContent: {response.content}")
    
    # Should be a dictionary (object)
    assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"
    
    # Should have some content
    assert len(parsed) > 0, "JSON object is empty"


@pytest.mark.integration
@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_json_mode_vs_normal_mode(api_key, model_name, simple_json_request):
    """Compare JSON mode vs normal mode for robustness."""
    # Test with JSON mode enabled
    llm_json = LLM(
        model_name=model_name,
        api_key=api_key,
        temperature=0.0,
        max_tokens=500,
        json_mode=True,
    )
    
    response_json = llm_json.call(simple_json_request)
    
    # Test without JSON mode
    llm_normal = LLM(
        model_name=model_name,
        api_key=api_key,
        temperature=0.0,
        max_tokens=500,
        json_mode=False,
    )
    
    response_normal = llm_normal.call(simple_json_request)
    
    # Both should work
    assert response_json.error_code is None
    assert response_normal.error_code is None
    
    # JSON mode response MUST be valid JSON
    try:
        parsed_json = json.loads(response_json.content)
        assert isinstance(parsed_json, dict)
    except json.JSONDecodeError:
        pytest.fail(f"JSON mode response is not valid JSON: {response_json.content}")
    
    # Normal mode might have markdown wrappers or other formatting
    # Just check it returned something
    assert len(response_normal.content) > 0


@pytest.mark.integration
@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_json_mode_with_complex_structure(api_key, model_name, structured_json_request):
    """Test JSON mode with a more complex structure."""
    llm = LLM(
        model_name=model_name,
        api_key=api_key,
        temperature=0.0,
        max_tokens=500,
        json_mode=True,
    )
    
    response = llm.call(structured_json_request)
    
    # Should not have errors
    assert response.error_code is None
    
    # Should be valid JSON
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON from {model_name}: {e}\nContent: {response.content}")
    
    # Should be an object
    assert isinstance(parsed, dict)
    
    # Should have expected structure (flexible - models may interpret differently)
    # Just verify it's not empty and has some key-value pairs
    assert len(parsed) > 0


@pytest.mark.integration
def test_json_mode_prevents_markdown_wrappers(api_key):
    """Test that JSON mode prevents markdown code block wrappers.
    
    Without JSON mode, models often wrap JSON in ```json...``` blocks.
    With JSON mode, they should return raw JSON.
    """
    # Use Claude Haiku as it's known to add markdown wrappers
    model_name = "anthropic/claude-haiku-4.5"
    
    request = [
        Message(
            role="user",
            content='Return JSON with this structure: {"numbers": [1, 2, 3], "sum": 6}',
            timestamp=time.time(),
        )
    ]
    
    # With JSON mode
    llm_json = LLM(
        model_name=model_name,
        api_key=api_key,
        temperature=0.0,
        max_tokens=300,
        json_mode=True,
    )
    
    response_json = llm_json.call(request)
    
    # Should be raw JSON, not wrapped in markdown
    content = response_json.content.strip()
    assert not content.startswith("```"), "JSON mode should not return markdown wrappers"
    assert content.startswith("{"), "Should start with { (raw JSON object)"
    assert content.endswith("}"), "Should end with } (raw JSON object)"
    
    # Should parse cleanly
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


@pytest.mark.integration
def test_json_mode_with_multiple_exchanges(api_key):
    """Test that JSON mode works in multi-turn conversations."""
    model_name = "deepseek/deepseek-chat"
    
    llm = LLM(
        model_name=model_name,
        api_key=api_key,
        temperature=0.0,
        max_tokens=300,
        json_mode=True,
    )
    
    # First turn
    messages = [
        Message(
            role="user",
            content='Respond with JSON: {"step": 1, "action": "start"}',
            timestamp=time.time(),
        )
    ]
    
    response1 = llm.call(messages)
    assert response1.error_code is None
    parsed1 = json.loads(response1.content)
    assert isinstance(parsed1, dict)
    
    # Second turn - add assistant response and new user message
    messages.append(
        Message(
            role="assistant",
            content=response1.content,
            timestamp=time.time(),
        )
    )
    messages.append(
        Message(
            role="user",
            content='Respond with JSON: {"step": 2, "action": "continue"}',
            timestamp=time.time(),
        )
    )
    
    response2 = llm.call(messages)
    assert response2.error_code is None
    parsed2 = json.loads(response2.content)
    assert isinstance(parsed2, dict)


@pytest.mark.integration
def test_json_mode_with_long_output(api_key):
    """Test JSON mode with a request that generates longer JSON output."""
    model_name = "deepseek/deepseek-chat"
    
    llm = LLM(
        model_name=model_name,
        api_key=api_key,
        temperature=0.0,
        max_tokens=1000,
        json_mode=True,
    )
    
    messages = [
        Message(
            role="user",
            content=(
                "Generate a JSON array of 10 numbers from 1 to 10, each with its square. "
                'Format: {"numbers": [{"value": 1, "square": 1}, ...]}'
            ),
            timestamp=time.time(),
        )
    ]
    
    response = llm.call(messages)
    assert response.error_code is None
    
    # Should be valid JSON
    parsed = json.loads(response.content)
    assert isinstance(parsed, dict)
    
    # Should have some reasonable structure
    assert len(parsed) > 0


@pytest.mark.integration
@pytest.mark.parametrize("model_name", MODELS_TO_TEST[:2])  # Test with first 2 models to save API calls
def test_json_mode_handles_unicode(api_key, model_name):
    """Test that JSON mode correctly handles Unicode characters."""
    llm = LLM(
        model_name=model_name,
        api_key=api_key,
        temperature=0.0,
        max_tokens=300,
        json_mode=True,
    )
    
    messages = [
        Message(
            role="user",
            content='Respond with JSON containing emoji: {"greeting": "Hello 👋", "language": "English"}',
            timestamp=time.time(),
        )
    ]
    
    response = llm.call(messages)
    assert response.error_code is None
    
    # Should parse as valid JSON
    parsed = json.loads(response.content)
    assert isinstance(parsed, dict)


if __name__ == "__main__":
    # Allow running tests directly for quick iteration
    print("Run with: pytest -m integration framework/tests/integration/test_llm_json_mode_contract.py -v")

