"""
Unit and integration test suite for ResumeIQ core functionalities.
Works with both unittest and pytest.
"""

from pathlib import Path
import tempfile
import unittest
import torch

from src.models import mean_pooling, get_HF_embeddings, cosine
from src.resume_parser import extract_text_data, extract_pdf_data
from src.skill_extractor import TECH_SKILLS, extract_skills, calculate_skill_match
from src.resume_scanner import compare
from src.job_recommender import recommend_jobs
from src.utils import get_data_path, get_project_root


class TestSkillExtractor(unittest.TestCase):
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


class TestResumeScanner(unittest.TestCase):
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


class TestJobRecommender(unittest.TestCase):
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


class TestLegacyCompatibility(unittest.TestCase):
    def test_legacy_models_import(self):
        import Models
        self.assertTrue(hasattr(Models, "get_HF_embeddings"))
        self.assertTrue(hasattr(Models, "cosine"))

    def test_legacy_scanner_import(self):
        import Resume_scanner
        self.assertTrue(hasattr(Resume_scanner, "compare"))
        self.assertTrue(hasattr(Resume_scanner, "extract_skills"))

    def test_legacy_recommender_import(self):
        import job_recommender
        self.assertTrue(hasattr(job_recommender, "recommend_jobs"))


if __name__ == "__main__":
    unittest.main()
