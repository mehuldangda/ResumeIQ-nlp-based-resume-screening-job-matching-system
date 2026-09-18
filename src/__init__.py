"""
ResumeIQ - AI Resume Screening and Job Recommendation System.
"""

from src.models import cosine, get_HF_embeddings, get_doc2vec_embeddings, mean_pooling
from src.resume_parser import extract_pdf_data, extract_text_data
from src.skill_extractor import TECH_SKILLS, calculate_skill_match, extract_skills
from src.resume_scanner import compare
from src.job_recommender import recommend_jobs
from src.utils import get_data_path, get_project_root

__version__ = "1.0.0"
__all__ = [
    "get_HF_embeddings",
    "get_doc2vec_embeddings",
    "cosine",
    "mean_pooling",
    "extract_pdf_data",
    "extract_text_data",
    "TECH_SKILLS",
    "extract_skills",
    "calculate_skill_match",
    "compare",
    "recommend_jobs",
    "get_project_root",
    "get_data_path",
]
