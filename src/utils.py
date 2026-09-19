"""
Utility functions for file path resolution and common operations in ResumeIQ.
Provides cross-platform path handling for Windows, macOS, and Linux.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Returns the absolute Path to the root directory of the ResumeIQ project.
    """
    return Path(__file__).resolve().parent.parent


def get_data_path(filename: str = "jobs.csv") -> Path:
    """
    Resolves the absolute path to a data file, prioritizing data/ directory
    and falling back to the project root for maximum compatibility.

    Args:
        filename: Name of the data file (default: 'jobs.csv').

    Returns:
        Path: Path object pointing to the existing file or the standard data/ path.
    """
    root = get_project_root()
    data_file = root / "data" / filename
    if data_file.exists():
        return data_file

    root_file = root / filename
    if root_file.exists():
        return root_file

    return data_file
