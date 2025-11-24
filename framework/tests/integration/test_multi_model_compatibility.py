"""
Multi-Model Compatibility Smoke Tests.
Tests that agents complete simple tasks successfully across different LLM models.
This validates that our framework works reliably across providers without crashes.
"""

import re
import time

from pydantic import BaseModel, Field

from agentic.framework.agents import Agent
from agentic.framework.config import get_config
from agentic.framework.llm import LLM
from agentic.framework.messages import ResultStatus
from agentic.framework.tools import create_tool
from agentic.observers.console_tracer import ConsoleTracer
from agentic.tools.calculator_tool import CalculatorTool

# Mock filesystem structure for deterministic testing
MOCK_FILESYSTEM = {
    "framework": {
        "agents.py": 18104,
        "config.py": 1851,
        "errors.py": 4541,
        "llm.py": 5427,
        "messages.py": 2945,
        "observers.py": 991,
        "tools.py": 2963,
        "tests": {
            "README.md": 181,
            "manual_test_error_messages.py": 3480,
            "integration": {
                "__init__.py": 63,
                "test_agent_integration.py": 2753,
                "test_llm_errors_integration.py": 2269,
                "test_llm_integration.py": 3923,
                "test_multi_model_compatibility.py": 7152,
                "test_openai_contract.py": 9907,
            },
            "unit": {
                "__init__.py": 59,
                "test_agent_error_handling.py": 4703,
                "test_agent_semantic_errors.py": 3690,
                "test_error_classification.py": 4574,
                "test_llm_error_handling.py": 6288,
                "test_llm_semantic_errors.py": 3545,
                "test_llm_transient_errors.py": 4225,
                "test_response_parsing.py": 9700,
                "test_tool_error_messages.py": 5847,
                "test_tool_unit.py": 3255,
            },
        },
    }
}


class MockListDirectoryTool(BaseModel):
    """Mock directory listing tool that returns deterministic results from MOCK_FILESYSTEM."""

    path: str = Field(description="Path to list (relative to root)")
    show_hidden: bool | None = Field(default=None, description="Show hidden files")

    def execute(self, dependencies: dict | None = None) -> str:
        """List directory contents from mock filesystem."""
        # Navigate through the mock filesystem
        parts = [p for p in self.path.split("/") if p]
        current = MOCK_FILESYSTEM

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return f"Error: Path '{self.path}' not found"

        if not isinstance(current, dict):
            return f"Error: '{self.path}' is not a directory"

        # Format output like the real ListDirectoryTool
        lines = [f"Contents of '{self.path}':"]
        for name, value in sorted(current.items()):
            if isinstance(value, dict):
                num_items = len(value)
                lines.append(f" {name}/ (directory, {num_items} items)")
            else:
                lines.append(f" {name} (file, {value:,} bytes)")

        return "\n".join(lines)


POLICY = """
Explore the 'framework' directory and find all test files
(files starting with 'test_' and ending with '.py').

The listdirectory tool shows file sizes. Add up the sizes of all test files and
report the total in bytes. Be terse and efficient, don't recapitulate, you have
access to past messages. Use tools to help you.

TWO PHASES - Complete Phase 1 before Phase 2:

PHASE 1 - EXPLORE (find ALL test files first):
  - Use listdirectory to explore each directory recursively
  - Avoid __pycache__ directories.
  - When you find subdirectories, list ALL of them in parallel
  - Traverse the tree by intuition in an efficient manner

PHASE 2 - CALCULATE (only after exploration complete):
  - Filter: filename STARTS WITH "test_" AND ENDS WITH ".py", skip files that do not start with "test_"!
  - Use calculator tool to sum file sizes (avoid arithmetic errors)
  - Calculate subtotals per directory, then sum all subtotals
  - To compute the sum more efficiently, sum up pairs of files in parallel
  - So by doing multiple calls to calculate in parallel you use log(n) steps to compute the sum
  - Be careful to quote the file sizes correctly.
"""

# Test models
MODELS_TO_TEST = [
    "google/gemini-2.5-flash",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "x-ai/grok-code-fast-1",
    "anthropic/claude-haiku-4.5",
]

# Expected answer: sum of all test_*.py files in MOCK_FILESYSTEM
# integration: 2753 + 2269 + 3923 + 7152 + 9907 = 26004
# unit: 4703 + 3690 + 4574 + 6288 + 3545 + 4225 + 9700 + 5847 + 3255 = 45827
# Total: 71831 bytes
EXPECTED_ANSWER = 71831


def run_model_test(model_name: str, api_key: str) -> dict:
    """Test a single model with file system navigation task.

    Returns dict with test results.
    """
    print(f"Testing: {model_name}")

    try:
        # Create agent with mock file navigation and calculator tools
        llm = LLM(model_name=model_name, api_key=api_key, temperature=0.0, max_tokens=1500)

        # Set up tools
        list_tool = create_tool(MockListDirectoryTool, dependencies={})
        calc_tool = create_tool(CalculatorTool, dependencies={})

        agent = Agent(
            llm=llm,
            tools=[list_tool, calc_tool],
            max_turns=15,
            observers=[ConsoleTracer(verbose=True)],
        )

        # Task: Find all test files in framework/ and sum their sizes
        start_time = time.time()
        result = agent.run(POLICY)
        duration = time.time() - start_time

        # Verify results - extract the number from agent's response
        result_text = result.value or ""

        # Extract numbers (with or without comma formatting)
        # Match numbers like "62065" or "62,065"
        numbers_raw = re.findall(r"\b\d[\d,]*\b", result_text)
        # Strip commas and filter for numbers with 4+ digits
        numbers = [n.replace(",", "") for n in numbers_raw if len(n.replace(",", "")) >= 4]

        # Check if the expected answer appears in the result
        success = (
            result.status == ResultStatus.SUCCESS
            and str(EXPECTED_ANSWER) in numbers
            and agent.turn_count >= 3  # Should need multiple turns
            and agent.turn_count < 15
        )

        if success:
            print(f"\n✅ {model_name}: PASSED")
            print(
                f"   Turns: {agent.turn_count}, Tokens: {agent.tokens_used}, Duration: {duration:.2f}s"
            )
            print(f"   Expected: {EXPECTED_ANSWER} bytes, Found: {numbers}")
            print(f"   Result: {result_text[:150]}")
        else:
            print(f"\n❌ {model_name}: FAILED")
            print(f"   Status: {result.status}")
            print(f"   Turns: {agent.turn_count}/{15}, Duration: {duration:.2f}s")
            print(f"   Expected: {EXPECTED_ANSWER} bytes, Found: {numbers}")
            print(f"   Result: {result_text[:200]}")

        return {
            "model": model_name,
            "success": success,
            "turns": agent.turn_count,
            "tokens": agent.tokens_used,
            "duration": duration,
            "result": result_text[:100],
            "error": None,
        }

    except Exception as e:
        print(f"\n❌ {model_name}: CRASHED")
        print(f"   Error: {e}")
        return {
            "model": model_name,
            "success": False,
            "turns": 0,
            "tokens": 0,
            "duration": 0.0,
            "result": "",
            "error": str(e),
        }


def run_all_tests():
    """Run tests on all models and print summary."""
    try:
        config = get_config()
        api_key = config.api_key
    except Exception as e:
        print("❌ Error: No API key found. Set OPENROUTER_API_KEY in .env")
        print(f"   {e}")
        return

    print("\n" + "=" * 70)
    print("MULTI-MODEL COMPATIBILITY TEST")
    print("=" * 70)
    print(f"Testing {len(MODELS_TO_TEST)} models with file system task")
    print("Task: Find all test files in framework/tests and sum their sizes")
    print(f"Expected answer: {EXPECTED_ANSWER:,} bytes")
    print("=" * 70)

    results = []
    for model_name in MODELS_TO_TEST:
        result = run_model_test(model_name, api_key)
        results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<40} {'Status':<10} {'Turns':<8} {'Tokens':<10} {'Time':<8}")
    print("-" * 70)

    for r in results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        duration_str = f"{r['duration']:.2f}s"
        error_note = f" ({r['error'][:30]}...)" if r["error"] else ""
        print(
            f"{r['model']:<40} {status:<10} {r['turns']:<8} {r['tokens']:<10} {duration_str:<8}{error_note}"
        )

    print("=" * 70)

    passed = sum(1 for r in results if r["success"])
    print(f"\nResult: {passed}/{len(results)} models passed")

    return results


if __name__ == "__main__":
    run_all_tests()
