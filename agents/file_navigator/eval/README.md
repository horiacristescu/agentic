# File Navigator - Evaluation

State-based validation for the file navigator agent.

## Structure

```
file_navigator/
├── agent.py              # Real agent (explores actual filesystems)
├── tools.py              # Real filesystem tools
├── test_tools.py         # Tests for real tools
├── README.md
│
├── prompts/              # Task prompts
│   ├── explore_codebase.txt
│   └── find_test_files.txt
│
├── scenarios/            # Mock filesystems for testing
│   ├── basic.json
│   └── README.md
│
└── eval/                 # ⭐ Evaluation infrastructure
    ├── validator.py      # State-based trace validator
    ├── runner.py         # Run agent + validate (all-in-one)
    └── test_validator.py # Tests for validator
```

## Quick Start

### Run Evaluation

```bash
# Default model, no JSON mode
python eval/runner.py

# Specific model with JSON mode
python eval/runner.py "anthropic/claude-haiku-4.5" --json-mode

# Save checkpoint
python eval/runner.py "deepseek/deepseek-chat" checkpoint.json
```

### Run Real Agent

```bash
# Use default prompt
python agent.py

# Use specific prompt
python agent.py explore_codebase
```

## Core Concept: State-Based Validation

The validator maintains state and validates chronologically:

```python
validator = ToolCallValidator(filesystem)

# As agent runs:
for tool_call in trace:
    # 1. Validate against current state
    validator.validate_tool_call(tool_call)  # ✓ or ✗
    
    # 2. Update state
    validator.process_tool_result(result)    # Add new valid values

# At end:
validator.check_completeness()  # All required calls made?
```

### What It Tracks

**Required Calls** (list of tuples):
- Starts with `[("mocklistdirectory", "framework")]`
- As subdirectories discovered → added to list
- As paths explored → removed from list  
- At end → must be empty (all paths explored)

**Valid Values** (set of integers):
- File sizes from `listdirectory` results
- Calculator results from previous calculations
- Used to validate calculator arguments (no hallucinations)

## Validation Tiers

1. **Final Answer**: Correct + from a tool result (not mental math)
2. **Completeness**: All paths explored + all test files found
3. **Trace**: No hallucinated values in tool call arguments

## Example Output

```
File Navigator Agent Evaluation
  Model: deepseek/deepseek-chat
  JSON Mode: enabled
======================================================================

🤖 Model: deepseek/deepseek-chat | Temperature: 0.0 | JSON mode: enabled
...

======================================================================
Running Validation
======================================================================

Ground Truth:
  Expected answer: 71831
  Number of test files: 14

Metrics:
  Total tool calls: 18
  - listdirectory: 4
  - calculator: 14

======================================================================
Validation Results
======================================================================

✓ PASS - Final Answer: 71831
✓ PASS - Task Completeness
✗ FAIL - Trace Validation
     1 violation(s)
     - Turn 5: Calculator arg 'y=0' not from known values

======================================================================
Result: ❌ FAILED (2/3 checks passed)
======================================================================
```

## Files

### `validator.py` (Core)
- `ToolCallValidator` class - state-based validation
- `get_ground_truth()` - extract expected values from scenario JSON
- Validates completeness, correctness, and trace integrity

### `runner.py` (All-in-one)
- Mock filesystem tools (load JSON scenarios)
- Agent orchestration (run with mock tools)
- Validation orchestration (validate + print results)
- Command-line interface

### `test_validator.py`
- Tests for validator logic
- Synthetic traces (perfect execution, hallucinations, wrong answers)

## Design Philosophy

**Minimal**: 3 files in eval/, everything else is consolidated
**Focused**: Validator is the core innovation
**Practical**: One command to run everything

---

**Status**: ✅ Production ready  
**Tests**: Unit tests for validator  
**Pattern**: State machine validation
