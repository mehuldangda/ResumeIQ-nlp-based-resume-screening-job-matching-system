"""
Legacy entrypoint for backward compatibility.
Redirects to src.resume_scanner and src.skill_extractor.
"""

from src.resume_scanner import compare
from src.skill_extractor import TECH_SKILLS, calculate_skill_match, extract_skills

__all__ = [
    "TECH_SKILLS",
    "extract_skills",
    "calculate_skill_match",
    "compare",
]
