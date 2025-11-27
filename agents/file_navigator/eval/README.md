# File Navigator Agent - Evaluation Infrastructure

This directory contains the complete evaluation infrastructure for the file navigator agent.

## Directory Structure

```
eval/
├── __init__.py              # Package exports
├── validator.py             # State-based trace validator ⭐
├── orchestrator.py          # Agent runner with mock filesystem
├── run_eval.py              # Main evaluation script
├── compare_models.py        # Multi-model comparison tool
├── ground_truth.py          # Extract expected values from scenarios
├── mock_tools.py            # Mock filesystem tools
├── test_validators.py       # Tests for validator logic
├── scenarios/               # Filesystem definitions (JSON)
│   ├── basic.json
│   └── README.md
└── prompts/                 # Versioned prompts
    ├── v1_find_test_files.txt
    └── README.md
```

## Core Concept: State-Based Validation

The validator uses a **state machine** approach rather than forensic analysis:

### Traditional Approach (Forensic)
```python
# After-the-fact analysis
tool_calls = extract_all_tool_calls(messages)
for tc in tool_calls:
    if tc.arg not in valid_values:
        FAIL("hallucination")
```

### Our Approach (State Machine)
```python
# Process chronologically, maintaining state
validator = ToolCallValidator(filesystem)

for tool_call in trace:
    # Validate against current state
    validator.validate_tool_call(tool_call)
    
    # Update state
    validator.process_tool_result(tool_call, result)

# Check completeness
validator.check_final_answer()
```

## Key Features

###  1. **Required Tool Calls Tracking**

The validator maintains a list of required tool calls:

```python
required_calls = [
    ("mocklistdirectory", "framework")  # From prompt
]
```

As the agent explores:
- **Validate**: Is this call in `required_calls`? If yes, remove it (✓ checked off)
- **Discover**: Did this result reveal new subdirectories? Add them to `required_calls`
- **Complete**: At end, `required_calls` must be empty (all paths explored)

### 2. **Valid Values Tracking**

The validator tracks what values are valid for calculator arguments:

```python
valid_values = set()  # Starts empty
```

As tools execute:
- **listdirectory** → extract file sizes → add to `valid_values`
- **calculator** → extract result → add to `valid_values`

When validating calculator calls:
- ✅ `calculator(x=2753, y=2269)` - both values from listdirectory results
- ❌ `calculator(x=16097, y=22874)` - mental math, not in `valid_values`
- ❌ `calculator(x=99999, y=12345)` - hallucinated values

### 3. **Three-Tier Validation**

**Tier 1: Final Answer**
- Is the answer correct?
- Did it come from a tool result? (not mental math)

**Tier 2: Task Completeness**
- Were all required paths explored?
- Were all test files found?
- Were any wrong files included?

**Tier 3: Trace Validation**
- No hallucinated tool call arguments
- No invalid tool calls
- Warnings for inefficiencies

## Usage

### Run Single Evaluation

```bash
# Default model, no JSON mode
python eval/run_eval.py

# Specific model with JSON mode
python eval/run_eval.py "anthropic/claude-haiku-4.5" --json-mode

# Save checkpoint
python eval/run_eval.py "deepseek/deepseek-chat" checkpoint.json
```

### Compare Models

```bash
# Compare multiple models (with and without JSON mode)
python eval/compare_models.py "anthropic/claude-haiku-4.5" "deepseek/deepseek-chat"
```

### Test Validators

```bash
# Run validator tests
pytest eval/test_validators.py -v
```

## Validation Output

The validator produces a comprehensive report:

```python
{
    "passed": bool,  # Overall pass/fail
    
    "answer": {
        "passed": bool,
        "issues": [...],
        "final_answer": int
    },
    
    "completeness": {
        "passed": bool,
        "issues": [...]  # missing_calls, missing_files, extra_files
    },
    
    "trace_validation": {
        "passed": bool,
        "violations": [...],  # Hallucinations, invalid calls
        "warnings": [...]     # Extra exploration, inefficiencies
    },
    
    "metrics": {
        "total_tool_calls": int,
        "calculator_calls": int,
        "listdir_calls": int
    }
}
```

## Example Output

```
Running Validation
======================================================================

Ground Truth:
  Expected answer: 71831
  Number of test files: 14
  Required paths: 4

Efficiency Metrics:
  Total tool calls: 18
  - listdirectory: 4
  - calculator: 14

======================================================================
Validation Results
======================================================================

✓ PASS - Final Answer
     Correct answer: 71831

✓ PASS - Task Completeness
     All required paths explored
     All test files found

✗ FAIL - Trace Validation
     Found 1 violation(s)
     - Turn 5: Calculator arg 'y=0' not from known values

======================================================================
Result: ❌ FAILED (2/3 checks passed)
======================================================================
```

## Design Principles

1. **State-based**: Validate chronologically, not retroactively
2. **Completeness**: Track what MUST be done, not just what was done
3. **Precision**: Distinguish hallucinations from inefficiencies
4. **Clarity**: Clear violation messages with context

## Adding New Scenarios

1. Create JSON file in `scenarios/`
2. Define filesystem structure
3. Run evaluation: `python eval/run_eval.py --filesystem your_scenario`

## Adding New Prompts

1. Create `.txt` file in `prompts/`
2. Use semantic versioning (e.g., `v2_find_test_files.txt`)
3. Run evaluation: `python eval/run_eval.py --prompt v2_find_test_files`

## Testing Philosophy

**"What you can't test doesn't matter"**

This evaluation infrastructure embodies the principle that:
- Robust testing is core to agent development
- Validators must be tested as rigorously as the agent itself
- Clear, actionable feedback drives improvement

---

**Status:** ✅ Production-ready  
**Tests:** Comprehensive unit + integration coverage  
**Design:** State machine validation pattern

