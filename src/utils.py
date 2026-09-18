"""
Utility functions for file path resolution and common operations in ResumeIQ.
"""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """
    Returns the root directory of the ResumeIQ project.
    """
    return Path(__file__).resolve().parent.parent


def get_data_path(filename: str = "jobs.csv") -> Path:
    """
    Resolves the absolute path to a data file, checking both data/ directory
    and the project root for maximum compatibility.

    Args:        filename: Name of the data file (default: 'jobs.csv').

    Returns:
        Path object pointing to the existing file or the expected data/ path.
    """
    root = get_project_root()
    data_file = root / "data" / filename
    if data_file.exists():
        return data_file

    root_file = root / filename
    if root_file.exists():
        return root_file

    return data_file
