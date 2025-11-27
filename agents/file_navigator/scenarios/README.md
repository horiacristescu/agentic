# Evaluation Scenarios

This directory contains scenario definitions for deterministic agent testing.
Each scenario includes input data (like mock filesystems) for evaluation.

## Format

Each JSON file represents a filesystem tree where:
- **Directories** are objects (dict)
- **Files** are integers (file size in bytes)

Example:
```json
{
  "src": {
    "main.py": 1234,
    "utils": {
      "helper.py": 567,
      "config.json": 89
    }
  },
  "README.md": 432
}
```

This represents:
```
src/
  main.py (file, 1,234 bytes)
  utils/
    helper.py (file, 567 bytes)
    config.json (file, 89 bytes)
README.md (file, 432 bytes)
```

## Creating New Test Cases

1. **Create a new JSON file** with descriptive name (e.g., `nested_dirs.json`, `large_codebase.json`)
2. **Define the structure** following the format above
3. **Use in evals** by passing the name to `load_filesystem("your_name")`

## Existing Filesystems

- **basic.json** - Simple framework directory with test files in subdirectories
  - Tests basic navigation and file finding
  - Expected total test file size: 68,043 bytes

## Usage

```python
from agentic.agents.file_navigator.mock_tools import load_filesystem

# Load by name
fs = load_filesystem("basic")

# Or by path
fs = load_filesystem("/path/to/custom.json")
```

## Design Principles

- Keep filesystems **focused** - test one scenario per file
- Use **realistic sizes** - helps test context/token management
- Include **edge cases** - empty dirs, deeply nested, files vs dirs with similar names
- **Document expected outcomes** in comments or separate file

