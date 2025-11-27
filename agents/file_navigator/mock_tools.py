"""Mock filesystem tools for deterministic testing."""

import json
from pathlib import Path

from pydantic import BaseModel, Field


def load_filesystem(name: str) -> dict:
    """Load a filesystem definition from JSON.

    Args:
        name: Filesystem name (e.g., 'basic') or path to JSON file

    Returns:
        Dictionary representing the filesystem structure
    """
    # If name is a path, use it directly
    if "/" in name or name.endswith(".json"):
        fs_path = Path(name)
    else:
        # Otherwise look in scenarios/
        fs_dir = Path(__file__).parent / "scenarios"
        fs_path = fs_dir / f"{name}.json"

    if not fs_path.exists():
        raise FileNotFoundError(f"Filesystem definition not found: {fs_path}")

    with open(fs_path) as f:
        return json.load(f)


class MockListDirectoryTool(BaseModel):
    """Mock directory listing tool that returns deterministic results from a JSON filesystem."""

    path: str = Field(description="Path to list (relative to root)")
    show_hidden: bool | None = Field(default=None, description="Show hidden files")

    def execute(self, filesystem: dict) -> str:
        """List directory contents from mock filesystem.

        Args:
            filesystem: The mock filesystem structure (injected as dependency)
        """

        # Navigate through the mock filesystem
        parts = [p for p in self.path.split("/") if p]
        current = filesystem

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return f"Error: Path '{self.path}' not found"

        if not isinstance(current, dict):
            return f"Error: '{self.path}' is not a directory"

        # Format output like the real ListDirectoryTool
        lines = [f"Contents of '{self.path}':"]
        for name, value in sorted(current.items()):
            if isinstance(value, dict):
                num_items = len(value)
                lines.append(f" {name}/ (directory, {num_items} items)")
            else:
                lines.append(f" {name} (file, {value:,} bytes)")

        return "\n".join(lines)
