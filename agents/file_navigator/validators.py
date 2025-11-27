"""Deterministic validators for file navigation task evaluation."""

import json
import re
from dataclasses import dataclass
from typing import Any

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
        if msg.role == "assistant" and msg.tool_calls:
            # Use the pre-parsed tool_calls from the message
            for tc in msg.tool_calls:
                # Handle both ToolCall objects and dicts
                if isinstance(tc, dict):
                    tool_name = tc.get("tool")
                    args = tc.get("args", {})
                    call_id = tc.get("id")
                else:
                    # Pydantic ToolCall object
                    tool_name = tc.tool
                    args = tc.args
                    call_id = tc.id

                # Find corresponding result by matching tool_call_id
                result = None
                for j in range(i + 1, len(messages)):
                    if messages[j].role == "tool" and messages[j].tool_call_id == call_id:
                        result = messages[j].content
                        break
                    # Stop searching when we hit the next assistant message
                    if messages[j].role == "assistant":
                        break

                tool_calls.append({
                    "tool": tool_name,
                    "args": args,
                    "result": result,
                    "turn_index": i,
                })

    return tool_calls


def extract_final_answer(messages: list[Message]) -> str | None:
    """Extract the final answer from the last assistant message or metadata."""
    # Check last few assistant messages for answers
    assistant_messages = [msg for msg in messages if msg.role == "assistant"]
    
    for msg in reversed(assistant_messages[-3:]):  # Check last 3 assistant messages
        # First check metadata for is_finished flag and result
        if msg.metadata:
            if msg.metadata.get("is_finished") and msg.metadata.get("result"):
                return str(msg.metadata["result"])
        
        # Then try parsing content as JSON
        if msg.content:
            try:
                parsed = json.loads(msg.content)
                if parsed.get("is_finished") and parsed.get("result"):
                    return parsed["result"]
            except (json.JSONDecodeError, KeyError):
                pass
    
    # Fallback: extract any substantial number from last few assistant messages
    # This handles cases where agent sends non-JSON final response
    for msg in reversed(assistant_messages[-3:]):
        if msg.content:
            # Use more flexible regex to catch numbers with or without commas
            numbers = re.findall(r"\d[\d,]+", msg.content)
            # Filter for substantial numbers (4+ digits when commas removed)
            substantial = [n for n in numbers if len(n.replace(",", "")) >= 4]
            if substantial:
                # Return the last substantial number found
                return substantial[-1].replace(",", "")
    
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


def check_correct_answer(messages: list[Message], expected: int, final_result: Any = None) -> ValidationResult:
    """Check if final answer matches expected value."""
    # First try to get answer from Result object
    final_answer = None
    if final_result and hasattr(final_result, 'value') and final_result.value:
        final_answer = str(final_result.value)
    
    # Fallback to extracting from messages
    if not final_answer:
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

    # Group tool calls by turn_index to handle parallel calls correctly
    calls_by_turn = {}
    for tc in tool_calls:
        turn = tc["turn_index"]
        if turn not in calls_by_turn:
            calls_by_turn[turn] = []
        calls_by_turn[turn].append(tc)

    # Build set of valid values as we go (file sizes + calculator outputs)
    valid_values = valid_file_sizes.copy()
    invalid_uses = []

    # Process turns in order
    for turn in sorted(calls_by_turn.keys()):
        turn_calc_results = []
        
        # First, validate all calculator calls in this turn
        for tc in calls_by_turn[turn]:
            if tc["tool"] == "calculator":
                args = tc.get("args", {})
                for arg_name, arg_value in args.items():
                    if isinstance(arg_value, int | float) and int(arg_value) not in valid_values:
                        invalid_uses.append({
                            "value": arg_value,
                            "arg": arg_name,
                            "turn": turn,
                            "valid_at_time": sorted(valid_values)[:20],  # More samples for debugging
                        })
                
                # Collect this turn's results
                if tc["result"]:
                    try:
                        result_num = int(tc["result"])
                        turn_calc_results.append(result_num)
                    except (ValueError, TypeError):
                        pass
        
        # Then add ALL results from this turn to valid values
        # (so next turn can use any of them)
        valid_values.update(turn_calc_results)

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
    final_result: Any = None,
) -> dict:
    """Run all validators on a message trace.

    Args:
        messages: Message history from agent execution
        filesystem: The filesystem dict used in the task
        final_result: Optional Result object with final answer

    Returns:
        dict with validation results, metrics, and summary
    """
    # Extract ground truth from filesystem
    ground_truth = get_ground_truth(filesystem)

    # Run validators (pass final_result for answer checking)
    validators = [
        lambda: check_used_calculator(messages),
        lambda: check_correct_answer(messages, ground_truth["expected_answer"], final_result),
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
