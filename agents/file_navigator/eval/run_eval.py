"""Run evaluation: execute agent + validate trace with new validator."""

import sys

from agentic.agents.file_navigator.eval.ground_truth import get_ground_truth
from agentic.agents.file_navigator.eval.mock_tools import load_filesystem
from agentic.agents.file_navigator.eval.orchestrator import run_eval
from agentic.agents.file_navigator.eval.validator import validate_trace


def run_and_validate(
    model_name: str | None = None,
    filesystem_name: str = "basic",
    prompt_name: str = "v1_find_test_files",
    save_trace: str | None = None,
    json_mode: bool = False,
) -> dict:
    """Run agent and validate the trace with state-based validator.

    Args:
        model_name: LLM model to use
        filesystem_name: Filesystem scenario to test
        prompt_name: Prompt version to use
        save_trace: Optional path to save checkpoint
        json_mode: If True, use API-level JSON mode

    Returns:
        dict with run info and validation results
    """
    print(f"\n{'=' * 70}")
    print("Running Evaluation")
    print(f"  Model: {model_name or 'default'}")
    print(f"  Filesystem: {filesystem_name}")
    print(f"  Prompt: {prompt_name}")
    print(f"  JSON Mode: {'enabled' if json_mode else 'disabled'}")
    print(f"{'=' * 70}\n")

    # Load filesystem for validation
    filesystem = load_filesystem(filesystem_name)
    ground_truth = get_ground_truth(filesystem)

    # Run agent
    result, agent = run_eval(
        model_name=model_name,
        filesystem_name=filesystem_name,
        prompt_name=prompt_name,
        save_checkpoint=save_trace,
        verbose=True,
        json_mode=json_mode,
    )

    # Validate trace with new state-based validator
    print(f"\n{'=' * 70}")
    print("Running Validation")
    print(f"{'=' * 70}\n")

    validation = validate_trace(
        messages=agent.messages,
        filesystem=filesystem,
        expected_answer=ground_truth["expected_answer"],
        expected_test_file_sizes=ground_truth["test_file_sizes"],
        final_result=result,
    )

    # Print results
    print_validation_results(validation, ground_truth)

    return {
        "run": {
            "model": agent.llm.model_name,
            "status": result.status.value,
            "turns": agent.turn_count,
            "tokens": agent.tokens_used,
        },
        "validation": validation,
    }


def print_validation_results(validation: dict, ground_truth: dict):
    """Print validation results in a clear format."""

    # Print ground truth
    print("Ground Truth:")
    print(f"  Expected answer: {ground_truth['expected_answer']}")
    print(f"  Number of test files: {ground_truth['num_test_files']}")
    print(f"  Required paths: {len(ground_truth['required_paths'])}")

    # Print metrics
    print(f"\nEfficiency Metrics:")
    metrics = validation["metrics"]
    print(f"  Total tool calls: {metrics['total_tool_calls']}")
    print(f"  - listdirectory: {metrics['listdir_calls']}")
    print(f"  - calculator: {metrics['calculator_calls']}")

    # Print validation results
    print(f"\n{'=' * 70}")
    print("Validation Results")
    print(f"{'=' * 70}\n")

    # Answer check
    answer = validation["answer"]
    if answer["passed"]:
        print(f"✓ PASS - Final Answer")
        print(f"     Correct answer: {answer['final_answer']}")
    else:
        print(f"✗ FAIL - Final Answer")
        for issue in answer["issues"]:
            print(f"     {issue.get('message', issue['type'])}")
            if "expected" in issue:
                print(f"     Expected: {issue['expected']}, Got: {issue.get('actual')}")

    # Completeness check
    completeness = validation["completeness"]
    if completeness["passed"]:
        print(f"\n✓ PASS - Task Completeness")
        print(f"     All required paths explored")
        print(f"     All test files found")
    else:
        print(f"\n✗ FAIL - Task Completeness")
        for issue in completeness["issues"]:
            print(f"     {issue.get('message', issue['type'])}")
            if issue.get("missing_calls"):
                print(f"     Missing: {issue['missing_calls']}")
            if issue.get("missing_sizes"):
                print(f"     Missing file sizes: {issue['missing_sizes']}")
            if issue.get("extra_sizes"):
                print(f"     Extra file sizes: {issue['extra_sizes']}")

    # Trace validation
    trace = validation["trace_validation"]
    if trace["passed"]:
        print(f"\n✓ PASS - Trace Validation")
        print(f"     No hallucinations or invalid tool calls")
    else:
        print(f"\n✗ FAIL - Trace Validation")
        print(f"     Found {len(trace['violations'])} violation(s)")
        for v in trace["violations"][:5]:  # Show first 5
            print(f"     - Turn {v.get('turn')}: {v.get('message', v['type'])}")

    # Warnings
    if trace.get("warnings"):
        print(f"\n⚠️  Warnings ({len(trace['warnings'])})")
        for w in trace["warnings"][:3]:  # Show first 3
            print(f"     - Turn {w.get('turn')}: {w.get('message', w['type'])}")

    # Overall result
    print(f"\n{'=' * 70}")
    if validation["passed"]:
        print("Result: ✅ ALL CHECKS PASSED")
    else:
        checks_passed = sum([
            validation["answer"]["passed"],
            validation["completeness"]["passed"],
            validation["trace_validation"]["passed"],
        ])
        print(f"Result: ❌ FAILED ({checks_passed}/3 checks passed)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    # Usage: python run_eval.py [model_name] [checkpoint_path] [--json-mode]
    model = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    checkpoint = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    json_mode = "--json-mode" in sys.argv

    eval_result = run_and_validate(
        model_name=model,
        save_trace=checkpoint,
        json_mode=json_mode,
    )

    # Print final pass/fail indicator
    if eval_result["validation"]["passed"]:
        print("✅ TASK PASSED")
    else:
        print("❌ TASK FAILED")
