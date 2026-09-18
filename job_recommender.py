"""
Legacy entrypoint for backward compatibility.
Redirects to src.job_recommender.
"""

from src.job_recommender import recommend_jobs

__all__ = [
    "recommend_jobs",
]
