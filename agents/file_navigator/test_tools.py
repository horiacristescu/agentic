"""Unit tests for file navigator tools."""

import tempfile
from pathlib import Path

import pytest

from .tools import GetFileInfoTool, ListDirectoryTool, ReadFileTool, SearchInDirectoryTool


class TestListDirectoryTool:
    """Tests for ListDirectoryTool."""

    def test_list_existing_directory(self):
        """Should list contents of existing directory."""
        tool = ListDirectoryTool(path="framework/")
        result = tool.execute()

        assert not result.startswith("Error")
        assert "agents.py" in result
        assert "llm.py" in result
        assert "tests/" in result

    def test_list_nonexistent_directory(self):
        """Should return error for nonexistent directory."""
        tool = ListDirectoryTool(path="nonexistent_folder/")
        result = tool.execute()

        assert result.startswith("Error")
        assert "does not exist" in result

    def test_list_file_instead_of_directory(self):
        """Should return error when path is a file, not directory."""
        tool = ListDirectoryTool(path="framework/agents.py")
        result = tool.execute()

        assert result.startswith("Error")
        assert "not a directory" in result

    def test_show_hidden_files(self):
        """Should show hidden files when requested."""
        # Create temp directory with hidden file
        with tempfile.TemporaryDirectory() as tmpdir:
            hidden_file = Path(tmpdir) / ".hidden"
            hidden_file.write_text("secret")
            normal_file = Path(tmpdir) / "normal.txt"
            normal_file.write_text("public")

            # Without show_hidden
            tool = ListDirectoryTool(path=tmpdir, show_hidden=False)
            result = tool.execute()
            assert ".hidden" not in result
            assert "normal.txt" in result

            # With show_hidden
            tool = ListDirectoryTool(path=tmpdir, show_hidden=True)
            result = tool.execute()
            assert ".hidden" in result
            assert "normal.txt" in result

    def test_empty_directory(self):
        """Should handle empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ListDirectoryTool(path=tmpdir)
            result = tool.execute()

            assert "empty" in result.lower()


class TestReadFileTool:
    """Tests for ReadFileTool."""

    def test_read_entire_file(self):
        """Should read entire file without line range."""
        tool = ReadFileTool(path="README.md")
        result = tool.execute()

        assert not result.startswith("Error")
        assert "lines total" in result
        # Line numbers should be present
        assert "1 |" in result or "   1 |" in result

    def test_read_file_with_line_range(self):
        """Should read file within specified line range."""
        tool = ReadFileTool(path="framework/agents.py", start_line=1, end_line=10)
        result = tool.execute()

        assert not result.startswith("Error")
        assert "[Lines 1-10 of" in result
        assert "1 |" in result or "   1 |" in result
        assert "10 |" in result or "  10 |" in result

    def test_read_nonexistent_file(self):
        """Should return error for nonexistent file."""
        tool = ReadFileTool(path="nonexistent.txt")
        result = tool.execute()

        assert result.startswith("Error")
        assert "does not exist" in result

    def test_read_directory_instead_of_file(self):
        """Should return error when path is directory."""
        tool = ReadFileTool(path="framework/")
        result = tool.execute()

        assert result.startswith("Error")
        assert "not a file" in result

    def test_read_with_invalid_line_range(self):
        """Should handle invalid line ranges."""
        # Create temp file with known line count
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_path = f.name

        try:
            # Start line beyond file length
            tool = ReadFileTool(path=temp_path, start_line=100)
            result = tool.execute()
            assert result.startswith("Error")
            assert "out of range" in result

        finally:
            Path(temp_path).unlink()

    def test_read_binary_file(self):
        """Should detect and reject binary files."""
        # Create temp binary file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03")
            temp_path = f.name

        try:
            tool = ReadFileTool(path=temp_path)
            result = tool.execute()
            assert result.startswith("Error")
            assert "binary" in result.lower()

        finally:
            Path(temp_path).unlink()


class TestGetFileInfoTool:
    """Tests for GetFileInfoTool."""

    def test_get_file_info(self):
        """Should return metadata for existing file."""
        tool = GetFileInfoTool(path="framework/agents.py")
        result = tool.execute()

        assert not result.startswith("Error")
        assert "Path:" in result
        assert "Type: file" in result
        assert "Size:" in result
        assert "Lines:" in result
        assert "Extension: .py" in result
        assert "Modified:" in result

    def test_get_directory_info(self):
        """Should return metadata for directory."""
        tool = GetFileInfoTool(path="framework/")
        result = tool.execute()

        assert not result.startswith("Error")
        assert "Type: directory" in result
        assert "Items:" in result

    def test_get_info_nonexistent_path(self):
        """Should return error for nonexistent path."""
        tool = GetFileInfoTool(path="nonexistent_file.txt")
        result = tool.execute()

        assert result.startswith("Error")
        assert "does not exist" in result


class TestSearchInDirectoryTool:
    """Tests for SearchInDirectoryTool."""

    def test_search_finds_matches(self):
        """Should find pattern in files."""
        tool = SearchInDirectoryTool(pattern="class Agent", path="framework/", file_pattern="*.py")
        result = tool.execute()

        assert not result.startswith("Error")
        assert "Found" in result
        assert "agents.py" in result
        assert "class Agent" in result

    def test_search_no_matches(self):
        """Should report when no matches found."""
        tool = SearchInDirectoryTool(
            pattern="this_pattern_will_never_exist_xyz123", path="framework/", file_pattern="*.py"
        )
        result = tool.execute()

        assert "No matches found" in result

    def test_search_with_file_pattern(self):
        """Should respect file pattern filter."""
        # Search for something that exists in .py but we'll filter to .md
        tool = SearchInDirectoryTool(pattern="class Agent", path="framework/", file_pattern="*.md")
        result = tool.execute()

        # Should not find it in .md files
        assert "No matches found" in result or "agents.py" not in result

    def test_search_nonexistent_directory(self):
        """Should return error for nonexistent directory."""
        tool = SearchInDirectoryTool(pattern="test", path="nonexistent_dir/", file_pattern="*.py")
        result = tool.execute()

        assert result.startswith("Error")
        assert "does not exist" in result

    def test_search_in_file_not_directory(self):
        """Should return error when path is a file."""
        tool = SearchInDirectoryTool(
            pattern="test", path="framework/agents.py", file_pattern="*.py"
        )
        result = tool.execute()

        assert result.startswith("Error")
        assert "not a directory" in result

    def test_search_respects_max_results(self):
        """Should limit results to max_results."""
        tool = SearchInDirectoryTool(
            pattern="def",  # Common pattern, will have many matches
            path="framework/",
            file_pattern="*.py",
            max_results=5,
        )
        result = tool.execute()

        # Should find exactly max_results matches
        assert "Found 5 matches" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
