"""Run evaluation: execute agent + validate trace."""

import json
import sys

from agentic.agents.file_navigator.eval_agent import run_eval
from agentic.agents.file_navigator.mock_tools import load_filesystem
from agentic.agents.file_navigator.validators import validate_trace


def run_and_validate(
    model_name: str | None = None,
    filesystem_name: str = "basic",
    prompt_name: str = "v1_find_test_files",
    save_trace: str | None = None,
    json_mode: bool = False,
) -> dict:
    """Run agent and validate the trace.

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

    # Run agent
    result, agent = run_eval(
        model_name=model_name,
        filesystem_name=filesystem_name,
        prompt_name=prompt_name,
        save_checkpoint=save_trace,
        verbose=True,
        json_mode=json_mode,
    )

    # Validate trace
    print(f"\n{'=' * 70}")
    print("Running Validators")
    print(f"{'=' * 70}\n")

    validation = validate_trace(agent.messages, filesystem, final_result=result)

    # Print ground truth info
    gt = validation["ground_truth"]
    print("Ground Truth:")
    print(f"  Expected answer: {gt['expected_answer']}")
    print(f"  Number of test files: {gt['num_test_files']}")
    print(f"  Required paths: {len(gt['required_paths'])}")
    print()

    # Print efficiency metrics
    metrics = validation["metrics"]
    print("Efficiency Metrics:")
    print(f"  Total tool calls: {metrics['total_tool_calls']}")
    print(f"  - listdirectory: {metrics['listdirectory_calls']}")
    print(f"  - calculator: {metrics['calculator_calls']}")
    print(f"  Assistant turns: {metrics['assistant_turns']}")
    print()

    # Print validation results
    for check in validation["checks"]:
        status = "✓ PASS" if check["passed"] else "✗ FAIL"
        print(f"{status} - {check['name']}")
        print(f"     {check['message']}")
        if check["details"] and not check["passed"]:
            print(f"     Details: {json.dumps(check['details'], indent=6)}")
        print()

    # Summary
    summary = validation["summary"]
    print(f"{'=' * 70}")
    print(f"Validation Summary: {summary['passed']}/{summary['total']} checks passed")
    print(f"Result: {'SUCCESS' if summary['all_passed'] else 'FAILED'}")
    print(f"{'=' * 70}\n")

    return {
        "model": model_name,
        "filesystem": filesystem_name,
        "prompt": prompt_name,
        "result": {
            "status": result.status.value,
            "value": result.value,
            "turns": agent.turn_count,
            "tokens": agent.tokens_used,
        },
        "validation": validation,
    }


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
    if eval_result["validation"]["summary"]["all_passed"]:
        print("✅ TASK PASSED")
    else:
        print("❌ TASK FAILED")
