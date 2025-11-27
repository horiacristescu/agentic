"""Extract ground truth from filesystem JSON for validation."""

from pathlib import Path


def extract_all_file_sizes(filesystem: dict, path: str = "") -> dict[str, int]:
    """Recursively extract all files and their sizes from filesystem.
    
    Args:
        filesystem: Dict representation of filesystem
        path: Current path prefix
    
    Returns:
        Dict mapping file paths to sizes: {"framework/agents.py": 18104, ...}
    """
    files = {}
    
    for name, value in filesystem.items():
        current_path = f"{path}/{name}" if path else name
        
        if isinstance(value, dict):
            # Directory - recurse
            files.update(extract_all_file_sizes(value, current_path))
        else:
            # File - record size
            files[current_path] = value
    
    return files


def extract_test_files(filesystem: dict) -> dict[str, int]:
    """Extract only test_*.py files from filesystem.
    
    Returns:
        Dict mapping test file paths to sizes
    """
    all_files = extract_all_file_sizes(filesystem)
    
    test_files = {
        path: size
        for path, size in all_files.items()
        if Path(path).name.startswith("test_") and path.endswith(".py")
    }
    
    return test_files


def compute_required_paths(test_files: dict[str, int]) -> list[str]:
    """Compute all directory paths that must be explored to find test files.
    
    Args:
        test_files: Dict of test file paths to sizes
    
    Returns:
        List of directory paths that contain test files
    """
    directories = set()
    
    for file_path in test_files.keys():
        # Add all parent directories
        path_parts = file_path.split("/")
        for i in range(len(path_parts)):
            dir_path = "/".join(path_parts[:i+1])
            if dir_path != file_path:  # Don't include the file itself
                directories.add(dir_path)
    
    return sorted(directories)


def compute_expected_answer(test_files: dict[str, int]) -> int:
    """Compute expected total of test file sizes.
    
    Args:
        test_files: Dict of test file paths to sizes
    
    Returns:
        Sum of all test file sizes
    """
    return sum(test_files.values())


def get_ground_truth(filesystem: dict) -> dict:
    """Extract all ground truth data from filesystem.
    
    Args:
        filesystem: Dict representation of filesystem
    
    Returns:
        Dict with:
            - all_files: All files and sizes
            - test_files: Only test_*.py files and sizes
            - test_file_sizes: Set of valid test file sizes
            - required_paths: Directories that must be explored
            - expected_answer: Correct sum
    """
    all_files = extract_all_file_sizes(filesystem)
    test_files = extract_test_files(filesystem)
    
    return {
        "all_files": all_files,
        "test_files": test_files,
        "test_file_sizes": set(test_files.values()),
        "required_paths": compute_required_paths(test_files),
        "expected_answer": compute_expected_answer(test_files),
        "num_test_files": len(test_files),
    }

