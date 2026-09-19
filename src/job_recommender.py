"""
Job recommendation engine for ResumeIQ.
Matches candidate resumes against a database of available career opportunities
and returns ranked results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.resume_scanner import compare
from src.utils import get_data_path


def recommend_jobs(
    resume_text: str,
    jobs_file: Optional[Union[str, Path]] = None,
    flag: str = "HuggingFace-BERT"
) -> List[Dict[str, Any]]:
    """
    Recommend jobs from the jobs catalog ranked by hybrid match score in descending order.

    Args:
        resume_text: Candidate resume plain text.
        jobs_file: Optional path to jobs dataset CSV. If None, resolves data/jobs.csv.
        flag: Embedding model flag ("HuggingFace-BERT" or "Doc2Vec").

    Returns:
        List[Dict[str, Any]]: List of recommended jobs sorted descending by match score.
    """
    if not resume_text or not resume_text.strip():
        return []

    if jobs_file is None:
        target_path = get_data_path("jobs.csv")
    else:
        target_path = Path(jobs_file)

    if not target_path.exists():
        raise FileNotFoundError(f"Jobs dataset not found at: {target_path}")

    jobs_df = pd.read_csv(target_path)
    recommendations = []

    for _, job in jobs_df.iterrows():
        job_desc = str(job.get("job_description", "")).strip()
        job_title = str(job.get("job_title", "Unknown Role")).strip()
        company = str(job.get("company", "Unknown Company")).strip()

        if not job_desc:
            continue

        comparison_results = compare([resume_text], job_desc, flag=flag)
        if not comparison_results:
            continue

        result = comparison_results[0]
        recommendations.append({
            "job_title": job_title,
            "company": company,
            "match_score": result["final_score"],
            "semantic_score": result["semantic_score"],
            "skill_score": result["skill_score"],
            "matched_skills": result["matched_skills"],
            "missing_skills": result["missing_skills"]
        })

    # Sort in descending order of final match score
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)

    return recommendations
