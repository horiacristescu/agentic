"""File navigation tools with sandbox support."""

import os
from pathlib import Path

from pydantic import BaseModel, Field


class ListDirectoryTool(BaseModel):
    """List contents of a directory with metadata."""

    path: str = Field(description="Directory path to list (relative or absolute)")
    show_hidden: bool = Field(
        default=False, description="Whether to show hidden files (starting with .)"
    )

    def execute(self, root_directory: str | None = None) -> str:
        """List directory contents with file/folder metadata."""
        # Resolve path relative to sandbox root if provided
        if root_directory:
            root = Path(root_directory).expanduser().resolve()
            target = (root / self.path).resolve()
            try:
                rel_path = target.relative_to(root)
                display_path = str(rel_path) if str(rel_path) != "." else "/"
            except ValueError:
                return "Error: Access denied - path outside allowed directory"
        else:
            target = Path(self.path).expanduser().resolve()
            display_path = self.path

        if not target.exists():
            return f"Error: Path '{display_path}' does not exist"

        if not target.is_dir():
            return f"Error: Path '{display_path}' is not a directory"

        try:
            items = []
            for item in sorted(target.iterdir()):
                # Skip hidden files unless requested
                if not self.show_hidden and item.name.startswith("."):
                    continue

                if item.is_file():
                    size = item.stat().st_size
                    items.append(f"  {item.name} (file, {size:,} bytes)")
                elif item.is_dir():
                    # Count items in directory
                    try:
                        count = len(list(item.iterdir()))
                        items.append(f"  {item.name}/ (directory, {count} items)")
                    except PermissionError:
                        items.append(f"  {item.name}/ (directory, permission denied)")

            if not items:
                return f"Directory '{display_path}' is empty"

            return f"Contents of '{display_path}':\n" + "\n".join(items)

        except PermissionError:
            return f"Error: Permission denied accessing '{display_path}'"
        except Exception as e:
            return f"Error listing directory: {str(e)}"


class ReadFileTool(BaseModel):
    """Read file contents, optionally with line range for large files."""

    path: str = Field(description="File path to read (relative or absolute)")
    start_line: int | None = Field(
        default=None, description="Starting line number (1-indexed). If None, reads from beginning"
    )
    end_line: int | None = Field(
        default=None, description="Ending line number (inclusive). If None, reads to end"
    )

    def execute(self, root_directory: str | None = None) -> str:
        """Read file content, optionally within a line range."""
        # Resolve path relative to sandbox root if provided
        if root_directory:
            root = Path(root_directory).expanduser().resolve()
            target = (root / self.path).resolve()
            try:
                rel_path = target.relative_to(root)
                display_path = str(rel_path)
            except ValueError:
                return "Error: Access denied - path outside allowed directory"
        else:
            target = Path(self.path).expanduser().resolve()
            display_path = self.path

        if not target.exists():
            return f"Error: File '{display_path}' does not exist"

        if not target.is_file():
            return f"Error: Path '{display_path}' is not a file"

        try:
            # Check if binary
            with open(target, "rb") as f:
                chunk = f.read(1024)
                if b"\0" in chunk:
                    return f"Error: File '{display_path}' appears to be binary"

            # Read file
            with open(target, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)

            # Apply line range if specified
            start = (self.start_line - 1) if self.start_line else 0
            end = self.end_line if self.end_line else total_lines

            # Validate range
            if start < 0 or start >= total_lines:
                return f"Error: start_line {self.start_line} out of range (file has {total_lines} lines)"
            if end > total_lines:
                end = total_lines

            selected_lines = lines[start:end]

            # Format with line numbers
            result_lines = []
            for i, line in enumerate(selected_lines, start=start + 1):
                result_lines.append(f"{i:4d} | {line.rstrip()}")

            range_info = (
                f"[Lines {start + 1}-{end} of {total_lines}]"
                if self.start_line or self.end_line
                else f"[{total_lines} lines total]"
            )

            return f"File: {display_path} {range_info}\n" + "\n".join(result_lines)

        except PermissionError:
            return f"Error: Permission denied reading '{display_path}'"
        except UnicodeDecodeError:
            return f"Error: File '{display_path}' encoding not supported (not UTF-8)"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class GetFileInfoTool(BaseModel):
    """Get metadata about a file or directory."""

    path: str = Field(description="Path to file or directory")

    def execute(self, root_directory: str | None = None) -> str:
        """Get file/directory metadata."""
        # Resolve path relative to sandbox root if provided
        if root_directory:
            root = Path(root_directory).expanduser().resolve()
            target = (root / self.path).resolve()
            try:
                rel_path = target.relative_to(root)
                display_path = str(rel_path)
            except ValueError:
                return "Error: Access denied - path outside allowed directory"
        else:
            target = Path(self.path).expanduser().resolve()
            display_path = self.path

        if not target.exists():
            return f"Error: Path '{display_path}' does not exist"

        try:
            stat = target.stat()
            info_lines = [
                f"Path: {display_path}",
                f"Absolute: {target}",
                f"Type: {'directory' if target.is_dir() else 'file'}",
            ]

            if target.is_file():
                info_lines.append(f"Size: {stat.st_size:,} bytes")

                # Count lines if text file
                try:
                    with open(target, encoding="utf-8", errors="replace") as f:
                        lines = sum(1 for _ in f)
                    info_lines.append(f"Lines: {lines:,}")
                except (PermissionError, UnicodeDecodeError, OSError):
                    pass

                # Extension
                if target.suffix:
                    info_lines.append(f"Extension: {target.suffix}")

            elif target.is_dir():
                try:
                    count = len(list(target.iterdir()))
                    info_lines.append(f"Items: {count}")
                except PermissionError:
                    info_lines.append("Items: (permission denied)")

            # Modified time
            from datetime import datetime

            modified = datetime.fromtimestamp(stat.st_mtime)
            info_lines.append(f"Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")

            return "\n".join(info_lines)

        except PermissionError:
            return f"Error: Permission denied accessing '{display_path}'"
        except Exception as e:
            return f"Error getting file info: {str(e)}"


class SearchInDirectoryTool(BaseModel):
    """Search for text pattern in files within a directory (recursive grep)."""

    pattern: str = Field(description="Text pattern to search for (case-sensitive)")
    path: str = Field(description="Directory path to search in")
    file_pattern: str = Field(
        default="*.py", description="File name pattern to match (e.g., '*.py', '*.md', '*')"
    )
    max_results: int = Field(default=50, description="Maximum number of matches to return")

    def execute(self, root_directory: str | None = None) -> str:
        """Search for pattern in files within directory."""
        import fnmatch

        # Resolve path relative to sandbox root if provided
        if root_directory:
            root = Path(root_directory).expanduser().resolve()
            target = (root / self.path).resolve()
            try:
                target.relative_to(root)
                display_path = str(target.relative_to(root))
            except ValueError:
                return "Error: Access denied - path outside allowed directory"
        else:
            target = Path(self.path).expanduser().resolve()
            display_path = self.path

        if not target.exists():
            return f"Error: Path '{display_path}' does not exist"

        if not target.is_dir():
            return f"Error: Path '{display_path}' is not a directory"

        matches = []
        files_searched = 0

        try:
            # Walk directory
            for root_dir, dirs, files in os.walk(target):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith(".")]

                for filename in files:
                    # Skip hidden files
                    if filename.startswith("."):
                        continue

                    # Check file pattern
                    if not fnmatch.fnmatch(filename, self.file_pattern):
                        continue

                    filepath = Path(root_dir) / filename
                    files_searched += 1

                    try:
                        # Skip binary files
                        with open(filepath, "rb") as f:
                            chunk = f.read(1024)
                            if b"\0" in chunk:
                                continue

                        # Search in file
                        with open(filepath, encoding="utf-8", errors="replace") as f:
                            for line_num, line in enumerate(f, 1):
                                if self.pattern in line:
                                    rel_path = filepath.relative_to(target)
                                    matches.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")
                                    if len(matches) >= self.max_results:
                                        break

                        if len(matches) >= self.max_results:
                            break

                    except (PermissionError, UnicodeDecodeError):
                        continue

                if len(matches) >= self.max_results:
                    break

            if not matches:
                return f"No matches found for '{self.pattern}' in {files_searched} files"

            header = f"Found {len(matches)} matches for '{self.pattern}' (searched {files_searched} files):\n"
            return header + "\n".join(matches)

        except PermissionError:
            return f"Error: Permission denied accessing '{display_path}'"
        except Exception as e:
            return f"Error searching directory: {str(e)}"
