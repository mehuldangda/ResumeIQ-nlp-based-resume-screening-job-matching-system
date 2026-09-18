"""
Resume comparison and hybrid matching engine for ResumeIQ.
Combines BERT semantic embeddings with explicit skill overlap analysis.
"""

from typing import Any, Dict, List
from src.models import cosine, get_HF_embeddings
from src.skill_extractor import calculate_skill_match, extract_skills


def compare(
    resume_texts: List[str],
    JD_text: str,
    flag: str = "HuggingFace-BERT"
) -> List[Dict[str, Any]]:
    """
    Compare multiple resumes against a job description.

    Args:
        resume_texts: List of resume text contents.
        JD_text: Target job description text content.
        flag: Embedding model flag ("HuggingFace-BERT" or "Doc2Vec").

    Returns:
        List of dictionaries with match scores, skills, and gaps for each resume.
    """
    results = []

    # Extract required skills from Job Description
    jd_skills = extract_skills(JD_text)

    if flag != "HuggingFace-BERT":
        return results

    # Generate JD embedding
    JD_embedding = get_HF_embeddings(JD_text)

    # Process every resume
    for resume_text in resume_texts:
        # Generate resume embedding
        resume_embedding = get_HF_embeddings(resume_text)

        # BERT semantic similarity
        semantic_score = cosine([resume_embedding], JD_embedding)[0]
        semantic_score = float(semantic_score)

        # Extract resume skills
        resume_skills = extract_skills(resume_text)

        # Calculate skill match
        skill_score = calculate_skill_match(resume_skills, jd_skills)

        # Skills present in both
        matched_skills = sorted(set(resume_skills).intersection(set(jd_skills)))

        # Skills required by JD but missing from resume
        missing_skills = sorted(set(jd_skills) - set(resume_skills))

        # Final hybrid score: 60% semantic similarity + 40% technical skill overlap
        final_score = (semantic_score * 0.60) + (skill_score * 0.40)

        results.append({
            "final_score": round(final_score, 2),
            "semantic_score": round(semantic_score, 2),
            "skill_score": round(skill_score, 2),
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })

    return results
