"""
Resume comparison and hybrid matching engine for ResumeIQ.
Combines semantic embeddings (HuggingFace BERT or Doc2Vec) with explicit skill overlap analysis.
"""

from typing import Any, Dict, List
from src.models import cosine, get_HF_embeddings, get_doc2vec_embeddings
from src.skill_extractor import calculate_skill_match, extract_skills


def compare(
    resume_texts: List[str],
    JD_text: str,
    flag: str = "HuggingFace-BERT"
) -> List[Dict[str, Any]]:
    """
    Compare multiple resumes against a target job description using a hybrid scoring model.

    The hybrid score consists of:
    - 60% Semantic Similarity: Measures conceptual and contextual alignment.
    - 40% Technical Skill Match: Quantifies hard skill prerequisites.

    Args:
        resume_texts: List of candidate resume text contents.
        JD_text: Target job description text content.
        flag: Embedding model identifier ("HuggingFace-BERT" or "Doc2Vec").

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing match scores,
                              skill lists, and gap diagnostics for each resume.
    """
    results = []

    if not resume_texts or not JD_text.strip():
        return results

    # Extract required skills from Job Description
    jd_skills = extract_skills(JD_text)

    # Branch 1: Doc2Vec Vectorization
    if flag == "Doc2Vec":
        jd_embedding, resume_embeddings = get_doc2vec_embeddings(JD_text, resume_texts)
        raw_semantic_scores = cosine(resume_embeddings, jd_embedding)

        for i, resume_text in enumerate(resume_texts):
            semantic_score = float(raw_semantic_scores[i])

            # Extract resume technical skills
            resume_skills = extract_skills(resume_text)

            # Calculate skill overlap percentage
            skill_score = calculate_skill_match(resume_skills, jd_skills)

            # Categorize matched and missing competencies
            matched_skills = sorted(set(resume_skills).intersection(set(jd_skills)))
            missing_skills = sorted(set(jd_skills) - set(resume_skills))

            # Compute 60/40 hybrid score
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

    # Branch 2: Hugging Face BERT Embeddings (Default)
    # Generate JD embedding once for efficiency
    JD_embedding = get_HF_embeddings(JD_text)

    for resume_text in resume_texts:
        # Generate resume embedding
        resume_embedding = get_HF_embeddings(resume_text)

        # Compute semantic similarity via cosine metric
        semantic_score = float(cosine([resume_embedding], JD_embedding)[0])

        # Extract resume technical skills
        resume_skills = extract_skills(resume_text)

        # Calculate skill overlap percentage
        skill_score = calculate_skill_match(resume_skills, jd_skills)

        # Categorize matched and missing competencies
        matched_skills = sorted(set(resume_skills).intersection(set(jd_skills)))
        missing_skills = sorted(set(jd_skills) - set(resume_skills))

        # Compute 60/40 hybrid score
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
