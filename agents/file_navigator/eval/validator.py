"""Tool Call Validator - State-based trace validation.

This validator processes agent traces chronologically, maintaining state
about what tool calls are required and what values are valid. It validates
both correctness (no hallucinations) and completeness (all required calls made).
"""

import re
from typing import Any

from agentic.framework.messages import Message, Result, ToolCall


class ToolCallValidator:
    """Validates tool calls against required/allowed lists, tracking task completion.
    
    Core concept:
    - Each tool call is validated against current state
    - Each tool result updates state (adds new valid values, new required calls)
    - At end, verify all required calls were made and final answer is valid
    
    This is a state machine validator that processes the trace chronologically,
    rather than doing forensic analysis after the fact.
    """
    
    def __init__(self, filesystem: dict, initial_path: str = "framework"):
        """Initialize validator with filesystem and expected starting point.
        
        Args:
            filesystem: The mock filesystem structure
            initial_path: Path that should be listed first (from prompt)
        """
        self.filesystem = filesystem
        
        # Required tool calls (must be completed)
        # Format: (tool_name, key_arg) - e.g., ("mocklistdirectory", "framework")
        self.required_calls: list[tuple[str, str]] = [
            ("mocklistdirectory", initial_path)
        ]
        
        # Valid values for calculator arguments (file sizes + calculator results)
        self.valid_values: set[int] = set()
        
        # Track all test file sizes we've seen
        self.test_file_sizes_seen: set[int] = set()
        
        # Track violations and warnings
        self.violations: list[dict] = []
        self.warnings: list[dict] = []
        
        # Metrics
        self.tool_call_count = 0
        self.calculator_call_count = 0
        self.listdir_call_count = 0
    
    def validate_tool_call(self, tool_call: ToolCall, turn: int) -> bool:
        """Validate a tool call against current state.
        
        Args:
            tool_call: The tool call to validate
            turn: Current turn number (for error reporting)
        
        Returns:
            True if valid, False if violation detected
        """
        self.tool_call_count += 1
        
        if tool_call.tool == "mocklistdirectory":
            return self._validate_listdir_call(tool_call, turn)
        elif tool_call.tool == "calculator":
            return self._validate_calculator_call(tool_call, turn)
        else:
            self.violations.append({
                "type": "unknown_tool",
                "tool": tool_call.tool,
                "turn": turn,
            })
            return False
    
    def _validate_listdir_call(self, tool_call: ToolCall, turn: int) -> bool:
        """Validate a listdirectory call."""
        self.listdir_call_count += 1
        path = tool_call.args.get("path", "")
        call_signature = ("mocklistdirectory", path)
        
        # Check if this is a required call
        if call_signature in self.required_calls:
            # Good! Remove from required list (checked off)
            self.required_calls.remove(call_signature)
            return True
        
        # Not required, but might be valid exploration
        # (e.g., agent exploring __pycache__ - not required but not wrong)
        self.warnings.append({
            "type": "extra_exploration",
            "path": path,
            "turn": turn,
            "message": f"Listed '{path}' but not required for task"
        })
        return True  # Allow it, just warn
    
    def _validate_calculator_call(self, tool_call: ToolCall, turn: int) -> bool:
        """Validate a calculator call - all arguments must be from valid values."""
        self.calculator_call_count += 1
        args = tool_call.args
        
        # Check each numeric argument
        for arg_name in ["x", "y"]:
            if arg_name in args:
                value = args[arg_name]
                if isinstance(value, int | float):
                    int_value = int(value)
                    if int_value not in self.valid_values:
                        # This is a hallucination or mental math
                        self.violations.append({
                            "type": "invalid_calculator_arg",
                            "arg": arg_name,
                            "value": int_value,
                            "turn": turn,
                            "valid_at_time": sorted(list(self.valid_values))[:20],
                            "message": f"Calculator arg '{arg_name}={int_value}' not from known values"
                        })
                        return False
        
        return True
    
    def process_tool_result(self, tool_call: ToolCall, result_msg: Message):
        """Update state after tool execution.
        
        Args:
            tool_call: The tool call that was executed
            result_msg: The result message from the tool
        """
        if tool_call.tool == "mocklistdirectory":
            self._process_listdir_result(tool_call, result_msg.content)
        elif tool_call.tool == "calculator":
            self._process_calculator_result(result_msg.content)
    
    def _process_listdir_result(self, tool_call: ToolCall, result: str):
        """Process listdirectory result - extract subdirs and file sizes."""
        base_path = tool_call.args.get("path", "")
        
        # Extract subdirectories - add them to required calls
        # Format: "subdir/ (directory, N items)"
        subdir_pattern = r"^\s*([^/\s]+)/\s+\(directory"
        for line in result.split("\n"):
            match = re.match(subdir_pattern, line)
            if match:
                subdir_name = match.group(1)
                # Skip __pycache__
                if subdir_name == "__pycache__":
                    continue
                
                # Add to required calls
                new_path = f"{base_path}/{subdir_name}" if base_path else subdir_name
                call_sig = ("mocklistdirectory", new_path)
                if call_sig not in self.required_calls:
                    self.required_calls.append(call_sig)
        
        # Extract file sizes - add to valid values
        # Format: "filename.py (file, 1,234 bytes)"
        file_pattern = r"^\s*(\S+)\s+\(file,\s+([\d,]+)\s+bytes\)"
        for line in result.split("\n"):
            match = re.match(file_pattern, line)
            if match:
                filename = match.group(1)
                size_str = match.group(2).replace(",", "")
                size = int(size_str)
                
                # Add to valid values
                self.valid_values.add(size)
                
                # Track test file sizes
                if filename.startswith("test_") and filename.endswith(".py"):
                    self.test_file_sizes_seen.add(size)
    
    def _process_calculator_result(self, result: str):
        """Process calculator result - add to valid values."""
        try:
            result_value = int(result.strip())
            self.valid_values.add(result_value)
        except ValueError:
            pass  # Ignore non-numeric results
    
    def check_final_answer(self, final_result: Result | None, expected_answer: int) -> dict:
        """Verify final answer is correct and from a tool result.
        
        Args:
            final_result: The Result object from agent.run()
            expected_answer: The expected correct answer
        
        Returns:
            Dict with validation results
        """
        issues = []
        
        # Extract final answer
        final_answer = None
        if final_result and final_result.value:
            # Try to extract number from result
            numbers = re.findall(r"\d[\d,]*", str(final_result.value))
            if numbers:
                final_answer = int(numbers[0].replace(",", ""))
        
        # Check answer is correct
        if final_answer != expected_answer:
            issues.append({
                "type": "wrong_answer",
                "expected": expected_answer,
                "actual": final_answer,
            })
        
        # Check answer came from a tool result (not mental math)
        if final_answer is not None and final_answer not in self.valid_values:
            issues.append({
                "type": "answer_not_from_tool",
                "answer": final_answer,
                "message": "Final answer not in tool results (mental calculation?)"
            })
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "final_answer": final_answer,
        }
    
    def check_completeness(self, expected_test_file_sizes: set[int]) -> dict:
        """Verify all required exploration was done.
        
        Args:
            expected_test_file_sizes: Set of test file sizes that should have been found
        
        Returns:
            Dict with completeness validation results
        """
        issues = []
        
        # Check all required calls were made
        if self.required_calls:
            issues.append({
                "type": "incomplete_exploration",
                "missing_calls": sorted(list(self.required_calls)),
                "message": f"Failed to explore {len(self.required_calls)} required paths"
            })
        
        # Check all test files were found
        missing_files = expected_test_file_sizes - self.test_file_sizes_seen
        if missing_files:
            issues.append({
                "type": "missing_test_files",
                "missing_sizes": sorted(list(missing_files)),
                "message": f"Failed to find {len(missing_files)} test files"
            })
        
        # Check for extra files (incorrect filtering)
        extra_files = self.test_file_sizes_seen - expected_test_file_sizes
        if extra_files:
            issues.append({
                "type": "extra_files_included",
                "extra_sizes": sorted(list(extra_files)),
                "message": f"Included {len(extra_files)} non-test files"
            })
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
        }
    
    def get_summary(self) -> dict:
        """Get validation summary with all results.
        
        Returns:
            Complete validation summary
        """
        total_issues = len(self.violations)
        total_warnings = len(self.warnings)
        
        return {
            "passed": total_issues == 0,
            "violations": self.violations,
            "warnings": self.warnings,
            "metrics": {
                "total_tool_calls": self.tool_call_count,
                "calculator_calls": self.calculator_call_count,
                "listdir_calls": self.listdir_call_count,
            },
            "summary": {
                "violations": total_issues,
                "warnings": total_warnings,
            }
        }


def validate_trace(
    messages: list[Message],
    filesystem: dict,
    expected_answer: int,
    expected_test_file_sizes: set[int],
    final_result: Result | None = None,
) -> dict:
    """Validate a complete agent trace.
    
    This is the main entry point for validation. It processes the trace
    chronologically and returns a comprehensive validation report.
    
    Args:
        messages: Complete message history from agent execution
        filesystem: The mock filesystem used
        expected_answer: The correct final answer
        expected_test_file_sizes: Set of file sizes that should be summed
        final_result: Optional Result object from agent.run()
    
    Returns:
        Comprehensive validation report with all checks
    """
    # Initialize validator
    validator = ToolCallValidator(filesystem, initial_path="framework")
    
    # Process trace chronologically
    turn = 0
    for i, msg in enumerate(messages):
        if msg.role == "assistant":
            turn += 1
            
            # Validate tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    validator.validate_tool_call(tc, turn)
        
        elif msg.role == "tool":
            # Process tool results to update state
            # Find the corresponding tool call
            if msg.tool_call_id:
                # Search backwards for the tool call
                for prev_msg in reversed(messages[:i]):
                    if prev_msg.tool_calls:
                        for tc in prev_msg.tool_calls:
                            if tc.id == msg.tool_call_id:
                                validator.process_tool_result(tc, msg)
                                break
    
    # Check final answer
    answer_check = validator.check_final_answer(final_result, expected_answer)
    
    # Check completeness
    completeness_check = validator.check_completeness(expected_test_file_sizes)
    
    # Get overall summary
    summary = validator.get_summary()
    
    # Combine all results
    all_passed = (
        summary["passed"]
        and answer_check["passed"]
        and completeness_check["passed"]
    )
    
    return {
        "passed": all_passed,
        "answer": answer_check,
        "completeness": completeness_check,
        "trace_validation": summary,
        "metrics": summary["metrics"],
    }

