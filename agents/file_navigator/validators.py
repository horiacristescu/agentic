"""Deterministic validators for file navigation task evaluation."""

import json
import re
from dataclasses import dataclass

from agentic.agents.file_navigator.expected_outputs import get_ground_truth
from agentic.framework.messages import Message


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str
    details: dict | None = None


def extract_tool_calls(messages: list[Message]) -> list[dict]:
    """Extract all tool calls from message history in order.

    Returns list of dicts with: {tool, args, result, turn_index}
    """
    tool_calls = []

    for i, msg in enumerate(messages):
        if msg.role == "assistant" and msg.content:
            try:
                parsed = json.loads(msg.content)
                if parsed.get("tool_calls"):
                    for tc in parsed["tool_calls"]:
                        # Find corresponding result in next messages
                        result = None
                        if i + 1 < len(messages) and messages[i + 1].role == "tool":
                            result = messages[i + 1].content

                        tool_calls.append(
                            {
                                "tool": tc.get("tool"),
                                "args": tc.get("args", {}),
                                "result": result,
                                "turn_index": i,
                            }
                        )
            except (json.JSONDecodeError, KeyError):
                continue

    return tool_calls


def extract_final_answer(messages: list[Message]) -> str | None:
    """Extract the final answer from the last assistant message."""
    for msg in reversed(messages):
        if msg.role == "assistant":
            try:
                parsed = json.loads(msg.content)
                if parsed.get("is_finished") and parsed.get("result"):
                    return parsed["result"]
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def check_used_calculator(messages: list[Message]) -> ValidationResult:
    """Verify agent used calculator tool (no manual arithmetic)."""
    tool_calls = extract_tool_calls(messages)
    calc_calls = [tc for tc in tool_calls if tc["tool"] == "calculator"]

    if not calc_calls:
        return ValidationResult(
            name="used_calculator",
            passed=False,
            message="Agent did not use calculator tool",
            details={"calculator_calls": 0},
        )

    return ValidationResult(
        name="used_calculator",
        passed=True,
        message=f"Agent used calculator {len(calc_calls)} time(s)",
        details={"calculator_calls": len(calc_calls)},
    )


def check_correct_answer(messages: list[Message], expected: int) -> ValidationResult:
    """Check if final answer matches expected value."""
    final_answer = extract_final_answer(messages)

    if not final_answer:
        return ValidationResult(
            name="correct_answer",
            passed=False,
            message="No final answer found in messages",
            details={"expected": expected, "actual": None},
        )

    # Extract number from answer (handle comma formatting)
    numbers = re.findall(r"\b\d[\d,]*\b", final_answer)
    if not numbers:
        return ValidationResult(
            name="correct_answer",
            passed=False,
            message="Could not extract numeric answer",
            details={"expected": expected, "actual": final_answer},
        )

    # Try to match expected value
    for num_str in numbers:
        num = int(num_str.replace(",", ""))
        if num == expected:
            return ValidationResult(
                name="correct_answer",
                passed=True,
                message=f"Correct answer: {expected}",
                details={"expected": expected, "actual": num},
            )

    # Found numbers but none match
    actual_nums = [int(n.replace(",", "")) for n in numbers]
    return ValidationResult(
        name="correct_answer",
        passed=False,
        message=f"Wrong answer. Expected {expected}, found {actual_nums}",
        details={"expected": expected, "actual": actual_nums},
    )


def check_used_only_valid_values(
    messages: list[Message], valid_file_sizes: set[int]
) -> ValidationResult:
    """Verify calculator only used actual file sizes or prior calculator results.

    This catches:
    - Using hallucinated/typo'd file sizes
    - Using file sizes before seeing them in listdirectory
    - Inventing intermediate sums
    """
    tool_calls = extract_tool_calls(messages)

    # Build set of valid values as we go (file sizes + calculator outputs)
    valid_values = valid_file_sizes.copy()
    invalid_uses = []

    for tc in tool_calls:
        if tc["tool"] == "calculator":
            # Check each operand
            args = tc.get("args", {})
            for arg_name, arg_value in args.items():
                if isinstance(arg_value, int | float) and int(arg_value) not in valid_values:
                    invalid_uses.append(
                        {
                            "value": arg_value,
                            "arg": arg_name,
                            "turn": tc["turn_index"],
                            "valid_at_time": sorted(valid_values)[:10],  # Sample for debugging
                        }
                    )

            # Add calculator result to valid values
            if tc["result"]:
                try:
                    result_num = int(tc["result"])
                    valid_values.add(result_num)
                except (ValueError, TypeError):
                    pass

    if invalid_uses:
        return ValidationResult(
            name="used_only_valid_values",
            passed=False,
            message=f"Found {len(invalid_uses)} invalid value(s) used in calculator",
            details={"invalid_uses": invalid_uses},
        )

    return ValidationResult(
        name="used_only_valid_values",
        passed=True,
        message="All calculator values were valid (file sizes or prior results)",
        details={
            "total_calculator_calls": len([tc for tc in tool_calls if tc["tool"] == "calculator"])
        },
    )


def check_used_correct_test_files(
    messages: list[Message], expected_sizes: set[int]
) -> ValidationResult:
    """Verify which test file sizes were actually used vs expected.

    Detects:
    - Including wrong files (non-test files)
    - Missing test files
    """
    tool_calls = extract_tool_calls(messages)

    # Extract file sizes from listdirectory results
    seen_sizes = set()
    for tc in tool_calls:
        if tc["tool"] == "mocklistdirectory" and tc["result"]:
            # Extract numbers from directory listing
            numbers = re.findall(r"\(file, ([\d,]+) bytes\)", tc["result"])
            seen_sizes.update(int(n.replace(",", "")) for n in numbers)

    # Extract sizes used in calculator
    used_sizes = set()
    for tc in tool_calls:
        if tc["tool"] == "calculator":
            args = tc.get("args", {})
            for arg_value in args.values():
                if isinstance(arg_value, int | float):
                    val = int(arg_value)
                    # Only count if it's a potential file size (not an intermediate sum)
                    if val in seen_sizes:
                        used_sizes.add(val)

    # Check for missing and extra files
    missing = expected_sizes - used_sizes
    extra = used_sizes - expected_sizes

    if missing or extra:
        issues = []
        if missing:
            issues.append(f"missing {len(missing)} test file(s)")
        if extra:
            issues.append(f"included {len(extra)} wrong file(s)")

        return ValidationResult(
            name="used_correct_test_files",
            passed=False,
            message=f"File filtering error: {', '.join(issues)}",
            details={
                "expected_count": len(expected_sizes),
                "used_count": len(used_sizes),
                "missing_sizes": sorted(missing),
                "extra_sizes": sorted(extra),
            },
        )

    return ValidationResult(
        name="used_correct_test_files",
        passed=True,
        message=f"Correctly used all {len(expected_sizes)} test file sizes",
        details={"test_files_used": len(used_sizes)},
    )


def check_explored_required_paths(
    messages: list[Message], required_paths: list[str]
) -> ValidationResult:
    """Verify agent explored all directories containing test files."""
    tool_calls = extract_tool_calls(messages)
    list_calls = [tc for tc in tool_calls if tc["tool"] == "mocklistdirectory"]

    explored_paths = {tc["args"].get("path", "") for tc in list_calls}

    # Check which required paths were NOT explored
    missing_paths = [p for p in required_paths if p not in explored_paths]

    if missing_paths:
        return ValidationResult(
            name="explored_required_paths",
            passed=False,
            message=f"Did not explore {len(missing_paths)} required path(s)",
            details={
                "explored": sorted(explored_paths),
                "required": required_paths,
                "missing": missing_paths,
            },
        )

    return ValidationResult(
        name="explored_required_paths",
        passed=True,
        message=f"Explored all {len(required_paths)} required paths",
        details={"paths_explored": len(explored_paths)},
    )


def compute_efficiency_metrics(messages: list[Message]) -> dict:
    """Compute efficiency metrics for the agent execution.

    Returns:
        dict with tool call counts and turn statistics
    """
    tool_calls = extract_tool_calls(messages)

    by_tool = {}
    for tc in tool_calls:
        tool_name = tc["tool"]
        by_tool[tool_name] = by_tool.get(tool_name, 0) + 1

    return {
        "total_tool_calls": len(tool_calls),
        "listdirectory_calls": by_tool.get("mocklistdirectory", 0),
        "calculator_calls": by_tool.get("calculator", 0),
        "assistant_turns": len([m for m in messages if m.role == "assistant"]),
        "total_turns": len(messages),
        "tools_breakdown": by_tool,
    }


def validate_trace(
    messages: list[Message],
    filesystem: dict,
) -> dict:
    """Run all validators on a message trace.

    Args:
        messages: Message history from agent execution
        filesystem: The filesystem dict used in the task

    Returns:
        dict with validation results, metrics, and summary
    """
    # Extract ground truth from filesystem
    ground_truth = get_ground_truth(filesystem)

    # Run validators
    validators = [
        lambda: check_used_calculator(messages),
        lambda: check_correct_answer(messages, ground_truth["expected_answer"]),
        lambda: check_used_only_valid_values(messages, ground_truth["test_file_sizes"]),
        lambda: check_used_correct_test_files(messages, ground_truth["test_file_sizes"]),
        lambda: check_explored_required_paths(messages, ground_truth["required_paths"]),
    ]

    results = [v() for v in validators]

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    # Compute efficiency metrics
    metrics = compute_efficiency_metrics(messages)

    return {
        "summary": {
            "passed": passed,
            "total": total,
            "all_passed": passed == total,
        },
        "ground_truth": {
            "expected_answer": ground_truth["expected_answer"],
            "num_test_files": ground_truth["num_test_files"],
            "required_paths": ground_truth["required_paths"],
        },
        "metrics": metrics,
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "details": r.details,
            }
            for r in results
        ],
    }
