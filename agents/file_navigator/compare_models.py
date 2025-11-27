"""Compare multiple models with and without JSON mode."""

import json
import time
from pathlib import Path

from agentic.agents.file_navigator.run_eval import run_and_validate

# Models to test
MODELS = [
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
    "google/gemini-2.0-flash-exp:free",
]


def run_comparison(models: list[str], output_file: str = "model_comparison.json"):
    """Run evaluation on multiple models with and without JSON mode.

    Args:
        models: List of model names to test
        output_file: Path to save results JSON
    """
    results = []

    for model in models:
        print(f"\n{'=' * 80}")
        print(f"Testing: {model}")
        print(f"{'=' * 80}\n")

        for json_mode in [False, True]:
            mode_str = "WITH JSON mode" if json_mode else "WITHOUT JSON mode"
            print(f"\n🔬 Testing {mode_str}...")

            try:
                start = time.time()
                eval_result = run_and_validate(
                    model_name=model,
                    json_mode=json_mode,
                )
                duration = time.time() - start

                result = {
                    "model": model,
                    "json_mode": json_mode,
                    "success": eval_result["validation"]["summary"]["all_passed"],
                    "passed_checks": eval_result["validation"]["summary"]["passed"],
                    "total_checks": eval_result["validation"]["summary"]["total"],
                    "turns": eval_result["run"]["turns"],
                    "tokens": eval_result["run"]["tokens"],
                    "duration_seconds": round(duration, 2),
                    "status": "success",
                }

                # Add failed check details
                failed_checks = [
                    check["name"]
                    for check in eval_result["validation"]["checks"]
                    if not check["passed"]
                ]
                result["failed_checks"] = failed_checks

                print(f"✅ Completed in {result['turns']} turns")
                print(f"   Passed: {result['passed_checks']}/{result['total_checks']}")
                if failed_checks:
                    print(f"   Failed: {', '.join(failed_checks)}")

            except Exception as e:
                result = {
                    "model": model,
                    "json_mode": json_mode,
                    "success": False,
                    "error": str(e),
                    "status": "error",
                }
                print(f"❌ Error: {e}")

            results.append(result)

            # Brief pause between runs
            time.sleep(2)

    # Save results
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}\n")

    # Print summary table
    print_summary_table(results)

    return results


def print_summary_table(results: list[dict]):
    """Print a formatted summary table of results."""
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    print(
        f"{'Model':<35} {'JSON Mode':<12} {'Status':<10} {'Turns':<8} {'Passed':<10} {'Failed Checks':<30}"
    )
    print("-" * 120)

    for r in results:
        if r["status"] == "error":
            print(
                f"{r['model']:<35} {'Yes' if r['json_mode'] else 'No':<12} {'ERROR':<10} {'-':<8} {'-':<10} {r.get('error', '')[:30]}"
            )
        else:
            json_mode_str = "✅ Yes" if r["json_mode"] else "❌ No"
            status_str = "✅ PASS" if r["success"] else "❌ FAIL"
            passed_str = f"{r['passed_checks']}/{r['total_checks']}"
            failed_str = ", ".join(r.get("failed_checks", []))[:30]

            print(
                f"{r['model']:<35} {json_mode_str:<12} {status_str:<10} {r['turns']:<8} {passed_str:<10} {failed_str:<30}"
            )

    print("=" * 120 + "\n")


if __name__ == "__main__":
    import sys

    # Allow specifying models via command line
    if len(sys.argv) > 1:
        models_to_test = sys.argv[1:]
    else:
        models_to_test = MODELS

    print(f"\n🧪 Running model comparison on {len(models_to_test)} model(s)")
    print(f"   Models: {', '.join(models_to_test)}\n")

    run_comparison(models_to_test)
