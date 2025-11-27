# Agent Prompts

This directory contains versioned prompts for the file navigator agent.

## Naming Convention

Format: `v{N}_{short_description}.txt`

Examples:
- `v1_find_test_files.txt` - Original task prompt
- `v2_find_test_files.txt` - Refined with better instructions
- `v1_explore_codebase.txt` - Different task entirely

## Versioning Strategy

**Never delete old versions.** Each version is a snapshot of prompt engineering work:

1. **v1** - Initial baseline prompt
2. **v2** - After discovering failure modes, refined instructions
3. **v3** - Further iteration based on multi-model testing

## Why Version Prompts?

- **Reproducibility** - Re-run exact evals from past experiments
- **A/B Testing** - Compare model performance across prompt versions
- **Learning** - Track what changes improved/degraded performance
- **Documentation** - Capture prompt engineering insights over time

## Usage

```python
from agents.file_navigator.eval_agent import run_eval

# Use specific prompt version
run_eval(
    model_name="gpt-4o-mini",
    filesystem_name="basic",
    prompt_name="v1_find_test_files"
)

# Or use a custom prompt file
run_eval(prompt_name="/path/to/custom_prompt.txt")
```

## When to Create a New Version

- Fixing ambiguity or errors in instructions
- Adding constraints discovered through testing
- Changing task structure or phases
- Simplifying language after successful runs

Document changes in git commit messages.

