"""
Technical skill extraction and skill match computation for ResumeIQ.
Uses word-boundary regular expressions to detect technical competencies
from unstructured text.
"""

import re
from typing import List, Optional

# Standard catalog of technical skills spanning languages, frameworks, databases, and DevOps
TECH_SKILLS = [
    # Programming Languages
    "python", "java", "c++", "c", "javascript", "typescript",
    "html", "css", "sql", "r", "go", "rust",
    # Frontend & Backend Frameworks
    "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi", "streamlit",
    # Databases & Storage
    "mongodb", "mysql", "postgresql", "sqlite", "redis", "cassandra",
    # Data Science, Machine Learning & Deep Learning
    "pandas", "numpy", "scipy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "machine learning", "deep learning", "nlp", "computer vision",
    "statistics", "data analysis", "data visualization", "matplotlib", "seaborn",
    # DevOps, Cloud & Tools
    "git", "github", "docker", "kubernetes", "aws", "azure", "gcp", "linux",
    # Analytics & Business Intelligence
    "power bi", "tableau", "excel",
    # APIs & Engineering Practices
    "rest api", "rest apis", "api", "graphql", "microservices",
    "feature engineering", "model training", "model evaluation",
    "data preprocessing", "data structures", "oop"
]


def extract_skills(text: Optional[str]) -> List[str]:
    """
    Extract technical skills from text using word-boundary regular expressions.

    The word boundary pattern `\\b<skill>\\b` ensures that exact terms are matched
    without false positives from sub-strings (e.g., prevents matching "c" inside "react").

    Args:
        text: Input text string (candidate resume or job description).

    Returns:
        List[str]: Alphabetically sorted list of unique extracted technical skills.
    """
    if not text or not isinstance(text, str):
        return []

    text_lower = text.lower()
    found_skills = []

    for skill in TECH_SKILLS:
        # Match complete words/phrases with boundary checks
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return sorted(set(found_skills))


def calculate_skill_match(resume_skills: List[str], jd_skills: List[str]) -> float:
    """
    Calculate the percentage of required job description skills present in the resume.

    Formula:
        Skill Match % = (|Resume Skills ∩ JD Skills| / |JD Skills|) * 100

    Args:
        resume_skills: List of skills detected in the candidate resume.
        jd_skills: List of skills detected in the target job description.

    Returns:
        float: Skill match percentage between 0.0 and 100.0.
    """
    if not jd_skills:
        return 0.0

    matched_skills = set(resume_skills).intersection(set(jd_skills))
    score = (len(matched_skills) / len(jd_skills)) * 100.0
    return round(score, 2)
