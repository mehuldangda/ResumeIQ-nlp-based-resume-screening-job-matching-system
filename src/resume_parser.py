"""
Resume and document parsing utilities for ResumeIQ.
Supports extracting plain text from PDF documents and text files.
"""

from pathlib import Path
from typing import BinaryIO, Union
import pdfplumber


def extract_pdf_data(file: Union[str, Path, BinaryIO]) -> str:
    """
    Extract plain text from an uploaded PDF file or a PDF file path.

    Iterates across all pages in the PDF document and concatenates extracted text
    with newline separators. Handles corrupted or empty files gracefully.

    Args:
        file: A file path string, Path object, or file-like object (e.g., Streamlit UploadedFile).

    Returns:
        str: Extracted text content from all pages, or an error message string if extraction fails.
    """
    data = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    data += text + "\n"
    except Exception as e:
        return f"Error reading PDF: {e}"

    return data.strip()


def extract_text_data(file_path: Union[str, Path]) -> str:
    """
    Read plain text from a text file on the local filesystem.

    Args:
        file_path: Path to the text file.

    Returns:
        str: String content of the file.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found at {path}"

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read().strip()
