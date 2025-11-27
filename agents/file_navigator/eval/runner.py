"""Evaluation Runner - Execute agent with mock filesystem and validate results.

This module combines:
- Mock filesystem tools (simple JSON loader)
- Agent orchestration (run with mock tools)
- Validation orchestration (run validators and print results)
"""

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from agentic.agents.file_navigator.eval.validator import ToolCallValidator, get_ground_truth
from agentic.framework.agents import Agent
from agentic.framework.config import get_config
from agentic.framework.llm import LLM
from agentic.framework.messages import Result
from agentic.framework.tools import create_tool
from agentic.observers.console_tracer import ConsoleTracer
from agentic.tools.calculator_tool import CalculatorTool

# ============================================================================
# Mock Filesystem Tools
# ============================================================================


def load_filesystem(name: str) -> dict:
    """Load a filesystem definition from scenarios/ directory.

    Args:
        name: Scenario name (e.g., 'basic') or path to JSON file

    Returns:
        Dictionary representing the filesystem structure
    """
    if "/" in name or name.endswith(".json"):
        fs_path = Path(name)
    else:
        scenarios_dir = Path(__file__).parent.parent / "scenarios"
        fs_path = scenarios_dir / f"{name}.json"

    if not fs_path.exists():
        raise FileNotFoundError(f"Filesystem definition not found: {fs_path}")

    with open(fs_path) as f:
        return json.load(f)


def load_prompt(name: str) -> str:
    """Load a prompt from prompts/ directory.

    Args:
        name: Prompt name (e.g., 'find_test_files') or path to prompt file

    Returns:
        The prompt text
    """
    if "/" in name or name.endswith(".txt"):
        prompt_path = Path(name)
    else:
        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompt_path = prompts_dir / f"{name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    with open(prompt_path) as f:
        return f.read()


class MockListDirectoryTool(BaseModel):
    """Mock directory listing tool that returns deterministic results from a JSON filesystem."""

    path: str = Field(description="Path to list (relative to root)")
    show_hidden: bool | None = Field(default=None, description="Show hidden files")

    def execute(self, filesystem: dict) -> str:
        """List directory contents from mock filesystem."""
        # Navigate through the mock filesystem
        parts = [p for p in self.path.split("/") if p]
        current = filesystem

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


# ============================================================================
# Agent Orchestration
# ============================================================================


def run_agent(
    model_name: str | None = None,
    filesystem_name: str = "basic",
    prompt_name: str = "find_test_files",
    save_checkpoint: str | None = None,
    json_mode: bool = False,
    verbose: bool = True,
) -> tuple[Result, Agent]:
    """Run file navigator agent with mock filesystem.

    Args:
        model_name: LLM model to use (default from config)
        filesystem_name: Name of filesystem scenario to load
        prompt_name: Name of prompt to use
        save_checkpoint: Optional path to save checkpoint after run
        json_mode: If True, use API-level JSON mode
        verbose: If True, show console output

    Returns:
        (Result, Agent) tuple - Agent contains messages for validation
    """
    # Load filesystem and prompt
    filesystem = load_filesystem(filesystem_name)
    prompt = load_prompt(prompt_name)

    # LLM init
    agent_config = get_config()
    llm = LLM(
        model_name=model_name or agent_config.model_name,
        api_key=agent_config.api_key,
        temperature=agent_config.temperature,
        max_tokens=1500,  # Override config - need sufficient tokens for complete responses
        json_mode=json_mode,  # Enable API-level JSON mode if requested
    )

    # Agent init with mock tools
    agent = Agent(
        llm=llm,
        tools=[
            create_tool(MockListDirectoryTool, dependencies={"filesystem": filesystem}),
            create_tool(CalculatorTool),
        ],
        max_turns=15,  # Match integration test
        observers=[ConsoleTracer(verbose=verbose)],
    )

    # Run task
    result = agent.run(prompt, auto_checkpoint=save_checkpoint)

    # Display result
    if verbose:
        print(f"\n{'=' * 70}")
        print("FINAL RESULT:")
        print(result.value or f"[{result.status.value.upper()}]")
        print(f"Status: {result.status}")
        print(f"Turns: {agent.turn_count}")
        print(f"Tokens: {agent.tokens_used}")
        print(f"{'=' * 70}\n")

    return result, agent


# ============================================================================
# Validation Orchestration
# ============================================================================


def validate_and_print(
    messages: list,
    filesystem: dict,
    final_result: Result,
) -> dict:
    """Validate agent trace and print results.

    Args:
        messages: Message history from agent
        filesystem: The mock filesystem used
        final_result: Result from agent.run()

    Returns:
        Validation results dict
    """
    # Get ground truth
    ground_truth = get_ground_truth(filesystem)

    # Validate
    print(f"{'=' * 70}")
    print("Running Validation")
    print(f"{'=' * 70}\n")

    validator = ToolCallValidator(filesystem, initial_path="framework")

    # Process trace chronologically
    turn = 0
    for i, msg in enumerate(messages):
        if msg.role == "assistant":
            turn += 1
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    validator.validate_tool_call(tc, turn)

        elif msg.role == "tool" and msg.tool_call_id:
            # Find corresponding tool call and process result
            for prev_msg in reversed(messages[:i]):
                if prev_msg.tool_calls:
                    for tc in prev_msg.tool_calls:
                        if tc.id == msg.tool_call_id:
                            validator.process_tool_result(tc, msg)
                            break

    # Check results
    answer_check = validator.check_final_answer(final_result, ground_truth["expected_answer"])
    completeness_check = validator.check_completeness(ground_truth["test_file_sizes"])
    trace_summary = validator.get_summary()

    # Print results
    print("Ground Truth:")
    print(f"  Expected answer: {ground_truth['expected_answer']}")
    print(f"  Number of test files: {ground_truth['num_test_files']}")

    print("\nMetrics:")
    metrics = trace_summary["metrics"]
    print(f"  Total tool calls: {metrics['total_tool_calls']}")
    print(f"  - listdirectory: {metrics['listdir_calls']}")
    print(f"  - calculator: {metrics['calculator_calls']}")

    print(f"\n{'=' * 70}")
    print("Validation Results")
    print(f"{'=' * 70}\n")

    # Answer
    if answer_check["passed"]:
        print(f"✓ PASS - Final Answer: {answer_check['final_answer']}")
    else:
        print("✗ FAIL - Final Answer")
        for issue in answer_check["issues"]:
            print(f"     {issue.get('message', issue['type'])}")

    # Completeness
    if completeness_check["passed"]:
        print("✓ PASS - Task Completeness")
    else:
        print("✗ FAIL - Task Completeness")
        for issue in completeness_check["issues"]:
            print(f"     {issue.get('message', issue['type'])}")

    # Trace
    if trace_summary["passed"]:
        print("✓ PASS - Trace Validation (no hallucinations)")
    else:
        print("✗ FAIL - Trace Validation")
        print(f"     {len(trace_summary['violations'])} violation(s)")
        for v in trace_summary["violations"][:3]:
            print(f"     - Turn {v.get('turn')}: {v.get('message', v['type'])}")

    # Overall
    all_passed = answer_check["passed"] and completeness_check["passed"] and trace_summary["passed"]

    print(f"\n{'=' * 70}")
    if all_passed:
        print("Result: ✅ ALL CHECKS PASSED")
    else:
        checks = sum(
            [answer_check["passed"], completeness_check["passed"], trace_summary["passed"]]
        )
        print(f"Result: ❌ FAILED ({checks}/3 checks passed)")
    print(f"{'=' * 70}\n")

    return {
        "passed": all_passed,
        "answer": answer_check,
        "completeness": completeness_check,
        "trace": trace_summary,
    }


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run evaluation from command line."""
    # Parse args: python runner.py [model_name] [checkpoint_path] [--json-mode]
    model = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    checkpoint = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    json_mode = "--json-mode" in sys.argv

    print(f"\n{'=' * 70}")
    print("File Navigator Agent Evaluation")
    print(f"  Model: {model or 'default'}")
    print(f"  JSON Mode: {'enabled' if json_mode else 'disabled'}")
    print(f"{'=' * 70}\n")

    # Run agent
    result, agent = run_agent(
        model_name=model,
        save_checkpoint=checkpoint,
        json_mode=json_mode,
    )

    # Validate
    filesystem = load_filesystem("basic")
    validation = validate_and_print(agent.messages, filesystem, result)

    # Exit code
    sys.exit(0 if validation["passed"] else 1)


if __name__ == "__main__":
    main()
