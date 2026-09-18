"""
Technical skill extraction and skill match computation for ResumeIQ.
"""

import re
from typing import List

# Standard tech skills dictionary
TECH_SKILLS = [
    "python", "java", "c++", "c", "javascript", "typescript",
    "html", "css", "react", "angular", "node.js", "express",
    "mongodb", "mysql", "sql", "postgresql",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "machine learning", "deep learning", "nlp",
    "streamlit", "flask", "django",
    "git", "github", "docker",
    "power bi", "tableau", "excel",
    "aws", "azure", "gcp",
    "statistics", "data analysis", "data visualization",
    "rest api", "rest apis", "api",
    "feature engineering", "model training", "model evaluation",
    "data preprocessing", "computer vision", "keras",
    "scipy", "matplotlib", "seaborn"
]


def extract_skills(text: str) -> List[str]:
    """
    Extract technical skills from text using word boundary regex matching.

    Args:
        text: Input text string (resume or job description).

    Returns:
        Sorted list of unique extracted technical skills.
    """
    if not text:
        return []

    text_lower = text.lower()
    found_skills = []

    for skill in TECH_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return sorted(set(found_skills))


def calculate_skill_match(resume_skills: List[str], jd_skills: List[str]) -> float:
    """
    Calculate the percentage of required JD skills present in the resume.

    Args:
        resume_skills: List of skills detected in resume.
        jd_skills: List of skills detected in job description.

    Returns:
        Skill match percentage (0.0 to 100.0).
    """
    if not jd_skills:
        return 0.0

    matched_skills = set(resume_skills).intersection(set(jd_skills))
    return (len(matched_skills) / len(jd_skills)) * 100.0
