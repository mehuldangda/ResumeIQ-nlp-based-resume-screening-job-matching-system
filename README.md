# ResumeIQ – AI-Powered Resume Screening & Job Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent NLP and Machine Learning career intelligence platform that parses resumes, evaluates candidate compatibility against job descriptions using transformer embeddings, extracts technical skills, detects skill gaps, and recommends matching career opportunities.

---

## 1. Project Overview

**ResumeIQ** is an end-to-end Machine Learning and Natural Language Processing (NLP) system designed to bridge the gap between job seekers and recruiters. By evaluating candidate resumes against target job descriptions using both deep semantic understanding (BERT sentence embeddings) and granular technical skill matching, ResumeIQ delivers accurate compatibility scores, actionable skill gap insights, and intelligent job recommendations.

The system features an interactive, dark-themed **Streamlit** dashboard as well as a scriptable **CLI mode** for automated screening.

---

## 2. Problem Statement

Traditional keyword-based Applicant Tracking Systems (ATS) frequently fail because:

1. **Keyword Rigidity**: Candidates with equivalent experience might use synonyms (e.g., "Deep Learning" vs. "Neural Networks") that simple string matching misses.
2. **Lack of Context**: Simple keyword counts ignore semantic context, project experience depth, and relevance.
3. **No Actionable Feedback**: Job seekers rarely receive concrete feedback explaining why their resume did not match or which specific technical skills they need to acquire.
4. **Disjointed Job Discovery**: Candidates must manually search through hundreds of job listings to identify roles matching their specific skill set.

**ResumeIQ** solves this by combining transformer-based contextual semantic similarity with explicit technical skill extraction, providing transparent hybrid scoring and automated job matching.

---

## 3. Key Features

- 📄 **Multi-Resume PDF Ingestion**: Extracts text seamlessly from single or multiple candidate resume PDFs using `pdfplumber`.
- 🧠 **Contextual BERT Embeddings**: Computes 768-dimensional document embeddings using `sentence-transformers/bert-base-nli-mean-tokens` with attention-weighted mean pooling.
- 🛠️ **Automated Technical Skill Extraction**: Scans resume and job description text for 45+ technical competencies spanning programming languages, ML/DL frameworks, databases, and cloud platforms.
- 🎯 **Hybrid Compatibility Scoring**: Combines **60% Semantic Similarity** (Cosine Similarity) with **40% Technical Skill Overlap** to ensure holistic matching.
- 📊 **Granular Skill Gap Analysis**: Categorizes skills into **Matched Skills** (strengths) and **Skills to Improve** (missing prerequisites).
- 💼 **Intelligent Job Recommendation**: Evaluates candidate profiles against a database of open roles and ranks matching opportunities.
- 🖥️ **Modern Interactive UI**: High-performance Streamlit interface with responsive dark mode cards, real-time feedback, and tabbed navigation.
- ⚡ **CLI Automation**: Supports direct command-line execution for rapid batch resume evaluation.
- 📚 **Interactive Learning Notebooks**: 4 well-documented Jupyter notebooks covering data exploration, preprocessing, model embeddings, and matching algorithms.

---

## 4. Application Workflow

```text
       Candidate Resume (PDF)                 Target Job Description (Text)
                 │                                           │
                 ▼                                           ▼
      PDF Text Extraction                         Text Sanitization & Ingestion
                 │                                           │
                 ├─────────────────────┬─────────────────────┤
                 │                     │                     │
                 ▼                     ▼                     ▼
     HuggingFace BERT Model    Skill Extractor       Doc2Vec Engine (Optional)
     (Mean-Pooled Embeddings)  (Regex Boundary)      (Dense Document Vectors)
                 │                     │                     │
                 ▼                     ▼                     │
         Cosine Similarity        Skill Overlap              │
          (Semantic Score)        (Skill Score)              │
                 │                     │                     │
                 └──────────┬──────────┘                     │
                            │                                │
                            ▼                                │
                 Hybrid Compatibility Score ◄────────────────┘
                (60% Semantic + 40% Skill)
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
   Candidate Matching Report        Job Recommendation Engine
  - Final Match Percentage         - Matches Against Database (jobs.csv)
  - Matched vs Missing Skills      - Ranked Fit Scores & Recommendations
  - Skill Gap Improvement Plan     - Candidate Readiness Assessment
```

---

## 5. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Core Language** | Python 3.10 / 3.11 |
| **NLP & Deep Learning** | Hugging Face Transformers, PyTorch, NLTK, Gensim (Doc2Vec) |
| **Machine Learning** | Scikit-Learn (`cosine_similarity`), NumPy, Pandas |
| **Document Processing** | `pdfplumber` |
| **Web Framework & UI** | Streamlit |
| **Testing & Quality** | `unittest`, `pytest` |

---

## 6. Mathematical Formulation

### 1. BERT-Based Sentence Embeddings & Attention Mean Pooling
Token embeddings from BERT's final hidden state are aggregated into a single document-level vector using attention-weighted mean pooling:

$$\mathbf{u} = \frac{\sum_{i=1}^{N} \mathbf{h}_i \cdot m_i}{\sum_{i=1}^{N} m_i}$$

Where $\mathbf{h}_i$ is the $i$-th token representation and $m_i \in \{0, 1\}$ is the attention mask indicating non-padding tokens.

### 2. Cosine Semantic Similarity
Semantic similarity between the resume vector $\mathbf{u}$ and the job description vector $\mathbf{v}$ is calculated as:

$$\text{Semantic Score} = \left( \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} \right) \times 100$$

### 3. Technical Skill Overlap
The skill match percentage is calculated as:

$$\text{Skill Score} = \left( \frac{|\text{Resume Skills} \cap \text{JD Skills}|}{|\text{JD Skills}|} \right) \times 100$$

### 4. Hybrid Compatibility Formula

$$\text{Final Match Score} = (0.60 \times \text{Semantic Score}) + (0.40 \times \text{Skill Score})$$

---

## 7. Folder Structure

```text
ResumeIQ/
│
├── app.py                      # Main Streamlit web application & CLI entrypoint
├── requirements.txt            # Pinned project dependencies
├── README.md                   # Comprehensive project documentation
├── DOCUMENTATION.md            # In-depth technical architecture manual
├── LICENSE                     # MIT Open-Source License
├── run.bat                     # Windows one-click launcher
├── .gitignore                  # Git ignore rules for Python & OS files
│
├── .devcontainer/
│   └── devcontainer.json       # Visual Studio Code Dev Container configuration
│
├── .streamlit/
│   └── config.toml             # Streamlit server & dark theme configuration
│
├── src/                        # Modular source package
│   ├── __init__.py             # Package exports & version metadata
│   ├── models.py               # BERT model caching, mean pooling, cosine similarity
│   ├── resume_parser.py        # PDF & text file extraction functions
│   ├── skill_extractor.py      # Technical skills catalog & regex extraction
│   ├── resume_scanner.py       # Hybrid resume-JD comparison engine
│   ├── job_recommender.py      # Dataset-driven job recommendation logic
│   └── utils.py                # Cross-platform path resolution helpers
│
├── data/
│   └── jobs.csv                # Job listings dataset (Role, Company, Description)
│
├── notebooks/                  # Educational Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_model_exploration.ipynb
│   └── 04_resume_job_matching.ipynb
│
├── assets/
│   └── screenshots/            # UI screenshots for documentation
│       ├── home_page.png
│       ├── analysis_results.png
│       └── workflow.png
│
└── tests/
    ├── __init__.py
    └── test_core.py            # Unit & integration test suite (unittest / pytest)
```

---

## 8. Installation and Local Setup

### Prerequisites
- Python 3.10 or 3.11 installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/mehuldangda/ResumeIQ-nlp-based-resume-screening-job-matching-system.git
cd ResumeIQ-nlp-based-resume-screening-job-matching-system
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
py -3.11 -m venv .venv
.\.venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. How to Run the Application

### Option A: Launch Interactive Web Application
```bash
streamlit run app.py
```
*The web interface will open automatically in your browser at `http://localhost:8501`.*

### Option B: Run via Command-Line Interface (CLI)
Compare a resume PDF against a job description text file directly:
```bash
python app.py path/to/resume.pdf path/to/job_description.txt
```

### Option C: Run Automated Tests
```bash
python -m unittest discover -s tests
```

---

## 10. Streamlit Community Cloud Deployment Guide

To deploy ResumeIQ seamlessly on **Streamlit Community Cloud**:

1. **Push your code to GitHub**:
   Ensure all changes and the `data/jobs.csv` file are pushed to your GitHub repository.
2. **Open Streamlit Community Cloud**:
   Navigate to [share.streamlit.io](https://share.streamlit.io) and log in.
3. **Create a New App**:
   - **Repository**: `mehuldangda/ResumeIQ-nlp-based-resume-screening-job-matching-system`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. **Select Python Version (Crucial)**:
   - Click **Advanced settings...**
   - Under **Python version**, select **3.11** (or 3.10).
5. **Click Deploy!**:
   Streamlit will install the requirements from `requirements.txt` using prebuilt binary wheels and start the application.

---

## 11. Learning Notebooks

The `notebooks/` directory contains 4 educational Jupyter notebooks:

| Notebook | Topic | Description |
| :--- | :--- | :--- |
| `01_data_exploration.ipynb` | Data Analysis | Explores the job catalog, required skills, and distributions. |
| `02_text_preprocessing.ipynb` | NLP Preprocessing | Demonstrates PDF parsing with `pdfplumber` and regex skill extraction. |
| `03_model_exploration.ipynb` | Deep Learning | Explores BERT tokenization, attention masks, mean pooling, and Doc2Vec. |
| `04_resume_job_matching.ipynb` | Matching Engine | Demonstrates cosine similarity, skill matching, hybrid score, and recommendations. |

---

## 12. Screenshots

### Home Screen & Resume Upload
![Home Screen](assets/screenshots/home_page.png)

### Resume Matching & Skill Gap Results
![Analysis Results](assets/screenshots/analysis_results.png)

### System Architecture & Workflow
![System Workflow](assets/screenshots/workflow.png)

---

## 13. License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 14. Author & Acknowledgements

- **Developer**: Mehul Dangda
- **Repository**: [ResumeIQ on GitHub](https://github.com/mehuldangda/ResumeIQ-nlp-based-resume-screening-job-matching-system)
- **Pretrained Models**: [Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens)
- **Frameworks**: [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/), [Scikit-learn](https://scikit-learn.org/)
