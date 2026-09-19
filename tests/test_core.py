"""
Comprehensive unit and integration test suite for ResumeIQ core functionalities.
Compatible with both unittest and pytest runners.
"""

from pathlib import Path
import tempfile
import unittest
import torch
import numpy as np

from src.models import (
    mean_pooling,
    get_HF_embeddings,
    get_doc2vec_embeddings,
    cosine,
    ensure_nltk_resources
)
from src.resume_parser import extract_text_data, extract_pdf_data
from src.skill_extractor import TECH_SKILLS, extract_skills, calculate_skill_match
from src.resume_scanner import compare
from src.job_recommender import recommend_jobs
from src.utils import get_data_path, get_project_root


class TestSkillExtractor(unittest.TestCase):
    """Tests for regex-based skill extraction and overlap calculations."""

    def test_extract_skills_empty(self):
        self.assertEqual(extract_skills(""), [])
        self.assertEqual(extract_skills(None), [])

    def test_extract_skills_single_and_multiword(self):
        text = "Experienced software engineer proficient in Python, SQL, Docker, and Machine Learning."
        skills = extract_skills(text)
        self.assertIn("python", skills)
        self.assertIn("sql", skills)
        self.assertIn("docker", skills)
        self.assertIn("machine learning", skills)

    def test_extract_skills_case_insensitivity(self):
        text = "Deep Learning with PYTORCH and TENSORFLOW"
        skills = extract_skills(text)
        self.assertIn("deep learning", skills)
        self.assertIn("pytorch", skills)
        self.assertIn("tensorflow", skills)

    def test_calculate_skill_match(self):
        # 100% match
        resume_skills = ["python", "sql", "git"]
        jd_skills = ["python", "sql"]
        self.assertEqual(calculate_skill_match(resume_skills, jd_skills), 100.0)

        # 50% match
        jd_skills = ["python", "sql", "docker", "aws"]
        self.assertEqual(calculate_skill_match(resume_skills, jd_skills), 50.0)

        # 0% match
        jd_skills = ["react", "node.js"]
        self.assertEqual(calculate_skill_match(resume_skills, jd_skills), 0.0)

        # Empty JD
        self.assertEqual(calculate_skill_match(resume_skills, []), 0.0)


class TestResumeParser(unittest.TestCase):
    """Tests for document text extraction from files."""

    def test_extract_text_data(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("Senior Data Scientist with NLP background.")
            temp_name = f.name

        try:
            content = extract_text_data(temp_name)
            self.assertIn("Senior Data Scientist with NLP background.", content)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def test_extract_pdf_data_nonexistent(self):
        result = extract_pdf_data("non_existent_file.pdf")
        self.assertIn("Error reading PDF", result)


class TestModels(unittest.TestCase):
    """Tests for attention mean pooling and cosine similarity."""

    def test_mean_pooling(self):
        # Mock token embeddings (batch_size=1, seq_len=3, hidden_size=4)
        token_embeddings = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0],
                                          [0.0, 0.0, 0.0, 0.0]]])
        attention_mask = torch.tensor([[1, 1, 0]])
        model_output = (token_embeddings,)

        pooled = mean_pooling(model_output, attention_mask)
        expected = torch.tensor([[3.0, 4.0, 5.0, 6.0]])
        self.assertTrue(torch.allclose(pooled, expected, atol=1e-5))

    def test_cosine_similarity(self):
        vec1 = np.array([[1.0, 0.0, 0.0]])
        vec2 = np.array([[1.0, 0.0, 0.0]])
        scores = cosine([vec1], vec2)
        self.assertEqual(float(scores[0]), 100.0)

    def test_doc2vec_embeddings(self):
        jd = "Python and Machine Learning engineer"
        resumes = ["Python developer with ML", "React frontend developer"]
        jd_emb, res_embs = get_doc2vec_embeddings(jd, resumes)

        self.assertEqual(jd_emb.shape[0], 1)
        self.assertEqual(jd_emb.shape[1], 512)
        self.assertEqual(len(res_embs), 2)
        self.assertEqual(res_embs[0].shape[1], 512)


class TestResumeScanner(unittest.TestCase):
    """Tests for hybrid comparison using BERT and Doc2Vec."""

    def test_compare_huggingface_bert(self):
        resume_text = "Proficient in Python, machine learning, and SQL."
        jd_text = "Looking for a Python and machine learning developer with SQL and Git skills."

        results = compare([resume_text], jd_text, flag="HuggingFace-BERT")
        self.assertEqual(len(results), 1)

        res = results[0]
        self.assertIn("final_score", res)
        self.assertIn("semantic_score", res)
        self.assertIn("skill_score", res)
        self.assertIn("matched_skills", res)
        self.assertIn("missing_skills", res)

        self.assertIn("python", res["matched_skills"])
        self.assertIn("machine learning", res["matched_skills"])
        self.assertIn("sql", res["matched_skills"])
        self.assertIn("git", res["missing_skills"])

        # Check hybrid score formula: 60% semantic + 40% skill
        expected_score = round((res["semantic_score"] * 0.60) + (res["skill_score"] * 0.40), 2)
        self.assertEqual(res["final_score"], expected_score)

    def test_compare_doc2vec(self):
        resume_text = "Proficient in Python, machine learning, and SQL."
        jd_text = "Looking for a Python and machine learning developer with SQL and Git skills."

        results = compare([resume_text], jd_text, flag="Doc2Vec")
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertIn("final_score", res)
        self.assertGreater(res["final_score"], 0)

    def test_compare_empty_inputs(self):
        self.assertEqual(compare([], "some jd"), [])
        self.assertEqual(compare(["some resume"], ""), [])


class TestJobRecommender(unittest.TestCase):
    """Tests for dataset-driven job recommendations."""

    def test_recommend_jobs(self):
        resume_text = "Experienced Machine Learning Engineer with Python, SQL, Pandas, Scikit-learn, and TensorFlow."
        recommendations = recommend_jobs(resume_text)

        self.assertGreater(len(recommendations), 0)
        first_job = recommendations[0]

        # Verify sorting by match score in descending order
        scores = [job["match_score"] for job in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))

        self.assertIn("job_title", first_job)
        self.assertIn("company", first_job)
        self.assertIn("match_score", first_job)
        self.assertIn("semantic_score", first_job)
        self.assertIn("skill_score", first_job)
        self.assertIn("matched_skills", first_job)
        self.assertIn("missing_skills", first_job)


class TestUtils(unittest.TestCase):
    """Tests for path resolution utilities."""

    def test_get_project_root(self):
        root = get_project_root()
        self.assertTrue((root / "src").exists())

    def test_get_data_path(self):
        path = get_data_path("jobs.csv")
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
