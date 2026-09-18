"""
Job recommendation engine for ResumeIQ.
Matches candidate resumes against a database of available career opportunities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.resume_scanner import compare
from src.utils import get_data_path


def recommend_jobs(
    resume_text: str,
    jobs_file: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Recommend jobs from the jobs dataset ranked by match score.

    Args:
        resume_text: Resume plain text.
        jobs_file: Optional path to jobs dataset CSV. If None, resolves data/jobs.csv.

    Returns:
        List of recommended jobs sorted in descending order of match score.
    """
    if jobs_file is None:
        target_path = get_data_path("jobs.csv")
    else:
        target_path = Path(jobs_file)

    if not target_path.exists():
        raise FileNotFoundError(f"Jobs dataset not found at: {target_path}")

    jobs = pd.read_csv(target_path)
    recommendations = []

    for _, job in jobs.iterrows():
        job_desc = str(job.get("job_description", ""))
        job_title = str(job.get("job_title", "Unknown Role"))
        company = str(job.get("company", "Unknown Company"))

        result = compare([resume_text], job_desc, flag="HuggingFace-BERT")[0]

        recommendations.append({
            "job_title": job_title,
            "company": company,
            "match_score": result["final_score"],
            "semantic_score": result["semantic_score"],
            "skill_score": result["skill_score"],
            "matched_skills": result["matched_skills"],
            "missing_skills": result["missing_skills"]
        })

    # Sort descending by final match score
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)

    return recommendations
