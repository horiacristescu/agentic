#!/usr/bin/env python3
"""Manual test script to see multi-model execution with ConsoleTracer.

Run this to watch agents execute with full visibility into each step.
"""

from agents.calculator.tools import CalculatorTool
from framework.agents import Agent
from framework.config import get_config
from framework.llm import LLM
from framework.tools import create_tool
from observers.console_tracer import ConsoleTracer

# Models to test
MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash-lite-001",
    "x-ai/grok-4.1-fast",
    "anthropic/claude-haiku-4.5",
]


def test_model(model_name: str, api_key: str):
    """Test a single model with full console tracing."""
    print("\n" + "=" * 80)
    print(f"TESTING: {model_name}")
    print("=" * 80)

    # Create agent with calculator tool and verbose console tracer
    llm = LLM(model_name=model_name, api_key=api_key, temperature=0.0, max_tokens=500)
    calculator_tool = create_tool(CalculatorTool, dependencies={})
    agent = Agent(
        llm=llm,
        tools=[calculator_tool],
        max_turns=5,
        observers=[ConsoleTracer(verbose=True)]  # Enable full tracing
    )

    # Run simple task
    task = "What is 15% of 80? Use the calculator tool."
    print(f"\n📋 TASK: {task}\n")

    try:
        result = agent.run(task)

        print("\n" + "-" * 80)
        print("FINAL RESULT:")
        print(f"  Status: {result.status}")
        print(f"  Turns: {agent.turn_count}")
        print(f"  Tokens: {agent.tokens_used}")
        print(f"  Answer: {result.value}")
        print("-" * 80)

        return True

    except Exception as e:
        print("\n" + "!" * 80)
        print(f"ERROR: {e}")
        print("!" * 80)
        return False


def main():
    """Run tests on all models."""
    try:
        config = get_config()
        api_key = config.api_key
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Make sure OPENROUTER_API_KEY is set in .env")
        return

    print("🚀 Multi-Model Agent Test with Console Tracing")
    print(f"Testing {len(MODELS)} models...\n")

    results = {}
    for model_name in MODELS:
        success = test_model(model_name, api_key)
        results[model_name] = "✓ PASS" if success else "✗ FAIL"

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for model, status in results.items():
        print(f"{model:<50} {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()

