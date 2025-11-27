"""Test suite for file_navigator validators.

This test uses a synthetic trace that represents a successful agent execution,
ensuring all validators correctly identify proper agent behavior.
"""

import time

from agentic.framework.messages import Message, Result, ResultStatus, ToolCall
from agentic.agents.file_navigator.expected_outputs import get_ground_truth
from agentic.agents.file_navigator.mock_tools import load_filesystem
from agentic.agents.file_navigator.validators import validate_trace


def create_successful_trace() -> tuple[list[Message], Result]:
    """Create a synthetic trace of a successful agent execution.
    
    Simulates an agent that:
    - Lists framework/ and finds tests/ subdirectory
    - Lists tests/ and finds integration/ and unit/ subdirectories
    - Lists both subdirectories in parallel
    - Finds 14 test files
    - Calculates their sum correctly using parallel addition (log n steps)
    - Returns final answer 71831
    """
    messages = []
    
    # Turn 1: List framework/
    messages.append(Message(
        role="user",
        content="Explore the 'framework' directory and find all test files...",
        timestamp=time.time(),
    ))
    
    messages.append(Message(
        role="assistant",
        content='{"reasoning": "Start by exploring framework/", "tool_calls": [...], "is_finished": false}',
        tool_calls=[
            ToolCall(
                id="call_1",
                tool="mocklistdirectory",
                args={"path": "framework"},
            )
        ],
        timestamp=time.time(),
    ))
    
    messages.append(Message(
        role="tool",
        content="tests/ (directory)",
        tool_call_id="call_1",
        timestamp=time.time(),
    ))
    
    # Turn 2: List tests/
    messages.append(Message(
        role="assistant",
        content='{"reasoning": "Explore tests subdirectory", "tool_calls": [...], "is_finished": false}',
        tool_calls=[
            ToolCall(
                id="call_2",
                tool="mocklistdirectory",
                args={"path": "framework/tests"},
            )
        ],
        timestamp=time.time(),
    ))
    
    messages.append(Message(
        role="tool",
        content="integration/ (directory)\nunit/ (directory)",
        tool_call_id="call_2",
        timestamp=time.time(),
    ))
    
    # Turn 3: List both subdirectories in parallel
    messages.append(Message(
        role="assistant",
        content='{"reasoning": "List both subdirs in parallel", "tool_calls": [...], "is_finished": false}',
        tool_calls=[
            ToolCall(
                id="call_3a",
                tool="mocklistdirectory",
                args={"path": "framework/tests/integration"},
            ),
            ToolCall(
                id="call_3b",
                tool="mocklistdirectory",
                args={"path": "framework/tests/unit"},
            ),
        ],
        timestamp=time.time(),
    ))
    
    messages.append(Message(
        role="tool",
        content=(
            "test_agent_integration.py (file, 2753 bytes)\n"
            "test_llm_errors_integration.py (file, 2269 bytes)\n"
            "test_llm_integration.py (file, 3923 bytes)\n"
            "test_multi_model_compatibility.py (file, 7152 bytes)\n"
            "test_openai_contract.py (file, 9907 bytes)"
        ),
        tool_call_id="call_3a",
        timestamp=time.time(),
    ))
    
    messages.append(Message(
        role="tool",
        content=(
            "test_agent_error_handling.py (file, 4703 bytes)\n"
            "test_agent_semantic_errors.py (file, 3690 bytes)\n"
            "test_error_classification.py (file, 4574 bytes)\n"
            "test_llm_error_handling.py (file, 6288 bytes)\n"
            "test_llm_semantic_errors.py (file, 3545 bytes)\n"
            "test_llm_transient_errors.py (file, 4225 bytes)\n"
            "test_response_parsing.py (file, 9700 bytes)\n"
            "test_tool_error_messages.py (file, 5847 bytes)\n"
            "test_tool_unit.py (file, 3255 bytes)"
        ),
        tool_call_id="call_3b",
        timestamp=time.time(),
    ))
    
    # Turn 4: First round of parallel additions (7 calls: 14 → 7 sums)
    messages.append(Message(
        role="assistant",
        content='{"reasoning": "Sum pairs in parallel (14 files → 7 sums)", "tool_calls": [...], "is_finished": false}',
        tool_calls=[
            ToolCall(id="call_4a", tool="calculator", args={"operation": "add", "x": 2753, "y": 2269}),
            ToolCall(id="call_4b", tool="calculator", args={"operation": "add", "x": 3923, "y": 7152}),
            ToolCall(id="call_4c", tool="calculator", args={"operation": "add", "x": 9907, "y": 4703}),
            ToolCall(id="call_4d", tool="calculator", args={"operation": "add", "x": 3690, "y": 4574}),
            ToolCall(id="call_4e", tool="calculator", args={"operation": "add", "x": 6288, "y": 3545}),
            ToolCall(id="call_4f", tool="calculator", args={"operation": "add", "x": 4225, "y": 9700}),
            ToolCall(id="call_4g", tool="calculator", args={"operation": "add", "x": 5847, "y": 3255}),
        ],
        timestamp=time.time(),
    ))
    
    # Add results for all 7 parallel calls
    for call_id, result in [
        ("call_4a", "5022"), ("call_4b", "11075"), ("call_4c", "14610"),
        ("call_4d", "8264"), ("call_4e", "9833"), ("call_4f", "13925"), ("call_4g", "9102"),
    ]:
        messages.append(Message(
            role="tool",
            content=result,
            tool_call_id=call_id,
            timestamp=time.time(),
        ))
    
    # Turn 5: Second round (7 → 3 sums, with one leftover)
    messages.append(Message(
        role="assistant",
        content='{"reasoning": "Sum pairs (7 sums → 3 sums + 1 leftover)", "tool_calls": [...], "is_finished": false}',
        tool_calls=[
            ToolCall(id="call_5a", tool="calculator", args={"operation": "add", "x": 5022, "y": 11075}),
            ToolCall(id="call_5b", tool="calculator", args={"operation": "add", "x": 14610, "y": 8264}),
            ToolCall(id="call_5c", tool="calculator", args={"operation": "add", "x": 9833, "y": 13925}),
            # 9102 is leftover, carried to next round
        ],
        timestamp=time.time(),
    ))
    
    for call_id, result in [("call_5a", "16097"), ("call_5b", "22874"), ("call_5c", "23758")]:
        messages.append(Message(
            role="tool",
            content=result,
            tool_call_id=call_id,
            timestamp=time.time(),
        ))
    
    # Turn 6: Third round (4 → 2 sums)
    messages.append(Message(
        role="assistant",
        content='{"reasoning": "Sum pairs (4 values → 2 sums)", "tool_calls": [...], "is_finished": false}',
        tool_calls=[
            ToolCall(id="call_6a", tool="calculator", args={"operation": "add", "x": 16097, "y": 22874}),
            ToolCall(id="call_6b", tool="calculator", args={"operation": "add", "x": 23758, "y": 9102}),
        ],
        timestamp=time.time(),
    ))
    
    for call_id, result in [("call_6a", "38971"), ("call_6b", "32860")]:
        messages.append(Message(
            role="tool",
            content=result,
            tool_call_id=call_id,
            timestamp=time.time(),
        ))
    
    # Turn 7: Final sum
    messages.append(Message(
        role="assistant",
        content='{"reasoning": "Final sum", "tool_calls": [...], "is_finished": false}',
        tool_calls=[
            ToolCall(id="call_7", tool="calculator", args={"operation": "add", "x": 38971, "y": 32860}),
        ],
        timestamp=time.time(),
    ))
    
    messages.append(Message(
        role="tool",
        content="71831",
        tool_call_id="call_7",
        timestamp=time.time(),
    ))
    
    # Turn 8: Final response
    messages.append(Message(
        role="assistant",
        content='{"reasoning": "Task complete", "is_finished": true, "result": "71831"}',
        timestamp=time.time(),
        metadata={
            "reasoning": "Task complete",
            "is_finished": True,
            "result": "71831",
        },
    ))
    
    # Create Result object matching the final answer
    result = Result(
        value="71831",
        status=ResultStatus.SUCCESS,
        metadata={"turn_count": 8},
    )
    
    return messages, result


def test_all_validators_pass_on_successful_trace():
    """Test that all validators pass on a correctly executed agent trace."""
    # Load filesystem
    filesystem = load_filesystem("basic")
    
    # Create successful trace
    messages, final_result = create_successful_trace()
    
    # Run validation
    results = validate_trace(messages, filesystem, final_result)
    
    # All validators should pass
    assert results["summary"]["all_passed"], (
        f"Expected all validators to pass, but got failures:\n"
        + "\n".join(
            f"  - {check['name']}: {check['message']}"
            for check in results["checks"]
            if not check["passed"]
        )
    )
    
    # Verify all checks ran
    check_names = {check["name"] for check in results["checks"]}
    assert "used_calculator" in check_names
    assert "correct_answer" in check_names
    assert "used_only_valid_values" in check_names
    assert "used_correct_test_files" in check_names
    assert "explored_required_paths" in check_names
    
    # Verify efficiency metrics exist
    assert "metrics" in results
    assert results["metrics"]["total_tool_calls"] == 17  # 1 + 1 + 2 + 7 + 3 + 2 + 1
    assert results["metrics"]["assistant_turns"] == 8


def test_validator_catches_hallucinated_value():
    """Test that validator catches when agent uses hallucinated file sizes."""
    # Load filesystem
    filesystem = load_filesystem("basic")
    
    # Create trace with a hallucinated value
    messages = [
        Message(
            role="user",
            content="Find test files...",
            timestamp=time.time(),
        ),
        Message(
            role="assistant",
            content='{"reasoning": "Using hallucinated value", "tool_calls": [...]}',
            tool_calls=[
                ToolCall(
                    id="call_1",
                    tool="calculator",
                    args={"operation": "add", "x": 99999, "y": 12345},  # Hallucinated!
                )
            ],
            timestamp=time.time(),
        ),
        Message(
            role="tool",
            content="112344",
            tool_call_id="call_1",
            timestamp=time.time(),
        ),
    ]
    
    result = Result(value="112344", status=ResultStatus.SUCCESS)
    results = validate_trace(messages, filesystem, result)
    
    # Should fail the valid values check
    used_valid_check = next(c for c in results["checks"] if c["name"] == "used_only_valid_values")
    assert not used_valid_check["passed"]
    assert "invalid" in used_valid_check["message"].lower()


def test_validator_catches_wrong_answer():
    """Test that validator catches incorrect final answer."""
    filesystem = load_filesystem("basic")
    
    messages = [
        Message(
            role="user",
            content="Find test files...",
            timestamp=time.time(),
        ),
        Message(
            role="assistant",
            content='{"reasoning": "Done", "is_finished": true, "result": "12345"}',
            timestamp=time.time(),
            metadata={"is_finished": True, "result": "12345"},
        ),
    ]
    
    result = Result(value="12345", status=ResultStatus.SUCCESS)
    results = validate_trace(messages, filesystem, result)
    
    # Should fail the correct answer check
    correct_answer_check = next(c for c in results["checks"] if c["name"] == "correct_answer")
    assert not correct_answer_check["passed"]
    assert "wrong" in correct_answer_check["message"].lower()


def run_all_tests():
    """Run all validator tests (for standalone execution)."""
    print("Running validator tests...")
    
    print("\n1. Testing all validators pass on successful trace...")
    test_all_validators_pass_on_successful_trace()
    print("   ✅ PASS")
    
    print("\n2. Testing validator catches hallucinated values...")
    test_validator_catches_hallucinated_value()
    print("   ✅ PASS")
    
    print("\n3. Testing validator catches wrong answer...")
    test_validator_catches_wrong_answer()
    print("   ✅ PASS")
    
    print("\n" + "=" * 70)
    print("✅ All validator tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

