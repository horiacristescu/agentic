# File Navigator Agent - Evaluation Infrastructure

This directory contains a complete evaluation infrastructure for the file navigator agent.

## Overview

The file navigator agent explores a directory structure, finds test files (files starting with `test_` and ending with `.py`), and calculates the total size of all test files using a calculator tool.

## Directory Structure

```
agents/file_navigator/
├── agent.py                  # Agent definition (currently a copy of calculator agent)
├── tools.py                  # Real file navigation tools
├── mock_tools.py             # Mock tools that use JSON filesystem data
├── scenarios/                # Test filesystem definitions (JSON)
│   ├── basic.json           # Default scenario with 14 test files
│   └── README.md            # Filesystem JSON format guidelines
├── prompts/                  # Versioned prompt templates
│   ├── v1_find_test_files.txt  # Current task prompt
│   └── README.md            # Prompt versioning guidelines
├── expected_outputs.py       # Ground truth extraction from filesystem JSON
├── validators.py             # Deterministic validation functions
├── eval_agent.py             # Agent execution harness
├── run_eval.py               # Main evaluation orchestration script
├── test_validators.py        # Pytest tests for validator logic
└── README.md                # This file
```

## Usage

### Run Evaluation

```bash
# Run with default model (gemini-2.0-flash-exp)
python agents/file_navigator/run_eval.py

# Run with specific model
python agents/file_navigator/run_eval.py "qwen/qwen-2.5-72b-instruct"

# Run and save checkpoint
python agents/file_navigator/run_eval.py "anthropic/claude-haiku-4.5" "checkpoints/claude_trace.json"
```

### Run Validator Tests

```bash
# Run standalone
python agents/file_navigator/test_validators.py

# Run with pytest
pytest agents/file_navigator/test_validators.py -v
```

## Evaluation Components

### 1. Scenarios (`scenarios/`)

JSON files defining mock filesystems for testing. Each scenario specifies:
- Directory structure
- File names and sizes
- Nested subdirectories

Example:
```json
{
  "framework": {
    "type": "directory",
    "contents": {
      "tests": {
        "type": "directory",
        "contents": {
          "test_example.py": {"type": "file", "size": 1234}
        }
      }
    }
  }
}
```

### 2. Prompts (`prompts/`)

Versioned prompt templates for the task. Currently using `v1_find_test_files.txt` which instructs the agent to:
- Phase 1: Explore the directory tree and find all test files
- Phase 2: Calculate the total size using parallel addition (log n steps)

### 3. Validators (`validators.py`)

Five deterministic validators check agent behavior:

#### `check_used_calculator`
Verifies the agent used the calculator tool (vs. doing arithmetic in its head).

#### `check_correct_answer`
Compares the agent's final answer against the ground truth sum.
Handles multiple answer formats (JSON result field, metadata, plain text).

#### `check_used_only_valid_values`
**Catches hallucinations!** Ensures every number used in calculator calls either:
- Is an actual file size from a listdirectory result, OR
- Is a prior calculator result

**Key feature:** Groups tool calls by turn to handle parallel calls correctly.
When multiple calculator calls happen in one turn, ALL results are added to the
valid set before checking the next turn.

#### `check_used_correct_test_files`
Verifies the agent only summed actual test files (start with `test_` and end with `.py`).
Detects:
- Missing test files (agent didn't find all of them)
- Extra files (agent included non-test files)

#### `check_explored_required_paths`
Ensures the agent explored all necessary directories to find the test files.

### 4. Efficiency Metrics

Tracks:
- Total tool calls
- Tool call breakdown (listdirectory vs calculator)
- Number of turns taken

### 5. Test Suite (`test_validators.py`)

Three pytest tests validate the validators themselves:
- `test_all_validators_pass_on_successful_trace`: Synthetic perfect trace passes all checks
- `test_validator_catches_hallucinated_value`: Detects when calculator uses made-up numbers
- `test_validator_catches_wrong_answer`: Detects incorrect final answer

Run with `pytest agents/file_navigator/test_validators.py -v`

## Ground Truth Extraction

The `expected_outputs.py` module derives all ground truth from the filesystem JSON:
- Recursively extracts all files and their sizes
- Filters for test files (filename starts with `test_` and ends with `.py`)
- Computes required directory paths that must be explored
- Calculates the expected sum

**No hardcoded expected values!** Everything is computed from the scenario JSON.

## Checkpointing

The evaluation harness supports saving agent state:

```python
result, agent = run_eval(
    model_name="anthropic/claude-haiku-4.5",
    save_checkpoint="checkpoints/trace.json"
)
```

Checkpoints contain:
- All messages (user, assistant, tool results)
- Token usage
- Turn count
- Timestamp

Useful for:
- Post-mortem analysis
- Replaying agent behavior
- Building training datasets

## Design Philosophy

### "What you can't test doesn't matter"

This evaluation infrastructure embodies the principle that robust testing is core to agent development, not an afterthought.

### Deterministic > LLM-as-judge

We prefer deterministic validators because they:
- Run instantly (no API calls)
- Give consistent results
- Are easier to debug when they fail
- Enable rapid iteration

### Test the tests

The validator test suite (`test_validators.py`) ensures our evaluation logic is correct. Without it, we can't trust the evaluation results.

## Debugging Workflow

When an agent fails validation:

1. **Check the console trace** - The `ConsoleTracer` shows:
   - Model name
   - Agent reasoning on each turn
   - Tool calls paired with their results
   - Raw responses on errors

2. **Review validator details** - Each failed validator includes:
   - Specific values that caused the failure
   - Context (e.g., which turn, what was valid at that time)

3. **Inspect checkpoints** - Saved agent state for post-mortem analysis

4. **Run validators in isolation** - `test_validators.py` can test specific scenarios

## Known Issues

### Agent Failure Modes

1. **File filtering errors**: Including `__init__.py`, `manual_test_*.py`, etc.
2. **Parse errors**: Returning tool call args instead of agent response JSON
3. **Empty responses**: Getting stuck after errors, returning nothing
4. **Context poisoning**: Echoing back error messages, token usage explosion

### Model-Specific Quirks

- **Gemini 2.0**: Sometimes returns raw tool call syntax instead of JSON
- **Qwen**: May include non-test files in calculation
- **Some models**: Get stuck in empty response loops

## Future Enhancements

- [ ] More scenarios (edge cases, larger filesystems)
- [ ] LLM-as-judge for trajectory analysis
- [ ] State-space validators for semantic correctness
- [ ] Multi-model comparison reports
- [ ] Success rate tracking over time
- [ ] Prompt optimization based on failure patterns

## Philosophy

This infrastructure represents a complete test-first approach to agent development:

1. **Define the task** (scenario + prompt)
2. **Define success criteria** (validators)
3. **Test the validators** (test_validators.py)
4. **Run the agent** (eval_agent.py)
5. **Analyze results** (run_eval.py)
6. **Iterate** on prompts, tools, or agent logic

The goal: Make agent optimization systematic, measurable, and reproducible.

