# ResumeIQ — Comprehensive Technical Documentation

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Architecture & Design](#2-system-architecture--design)
3. [NLP & Machine Learning Pipeline](#3-nlp--machine-learning-pipeline)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Component & Module Breakdown](#5-component--module-breakdown)
6. [Data Specifications & File Hierarchy](#6-data-specifications--file-hierarchy)
7. [Streamlit Community Cloud Deployment Guide](#7-streamlit-community-cloud-deployment-guide)
8. [Learning Notebooks Guide](#8-learning-notebooks-guide)
9. [Troubleshooting & Common Pitfalls](#9-troubleshooting--common-pitfalls)
10. [Future Roadmap & Scalability](#10-future-roadmap--scalability)

---

## 1. Executive Summary

**ResumeIQ** is an industry-standard Natural Language Processing (NLP) and Machine Learning system designed to evaluate candidate resumes against job descriptions, perform granular skill gap diagnostics, and rank candidate profiles against available career opportunities.

Unlike keyword-only Applicant Tracking Systems (ATS) that suffer from rigid vocabulary mismatch (e.g., missing candidates who write "Neural Networks" instead of "Deep Learning"), ResumeIQ employs a **dual-engine hybrid approach**:
1. **Contextual Semantic Matching**: Leverages a transformer model (`sentence-transformers/bert-base-nli-mean-tokens`) with attention-weighted mean pooling and cosine similarity to capture the conceptual essence of candidate experience.
2. **Explicit Competency Extraction**: Employs compiled word-boundary regular expressions across a curated 45+ technical skills taxonomy to quantify hard technical prerequisite fulfillment and identify explicit skill gaps.

The result is an objective, weighted **Compatibility Score (60% Semantic + 40% Skill Match)** paired with actionable diagnostics and career path recommendations.

---

## 2. System Architecture & Design

ResumeIQ follows a clean, modular, layered architecture separating user interface, business logic, machine learning inference, and data persistence.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          User Interface Layer                          │
│        Streamlit Web Application (app.py)  /  CLI Batch Runner         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                         Core Processing Layer                          │
│                                                                        │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │
│   │  src/resume_parser  │  │ src/skill_extractor │  │ src/utils.py │  │
│   │  - PDF extraction   │  │ - 45+ Tech Catalog  │  │ - Safe paths │  │
│   │  - Text extraction  │  │ - Regex Boundaries  │  │ - Root lookup│  │
│   └──────────┬──────────┘  └──────────┬──────────┘  └──────────────┘  │
│              │                        │                                │
│              └───────────┬────────────┘                                │
│                          ▼                                             │
│   ┌──────────────────────────────────────────────┐                     │
│   │              src/resume_scanner              │                     │
│   │  - Hybrid scoring (0.60 Semantic + 0.40 Skill)                     │
│   │  - Matched vs. Missing skill categorization  │                     │
│   └──────────────────────┬───────────────────────┘                     │
│                          │                                             │
│   ┌──────────────────────▼───────────────────────┐                     │
│   │              src/job_recommender             │                     │
│   │  - Catalog ranking against data/jobs.csv     │                     │
│   │  - Descending score sorting & assessment     │                     │
│   └──────────────────────┬───────────────────────┘                     │
└──────────────────────────┼─────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────────┐
│                          ML & Modeling Layer                           │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                          src/models                            │   │
│   │  - Hugging Face BERT (bert-base-nli-mean-tokens)               │   │
│   │  - Attention-Weighted Mean Pooling                             │   │
│   │  - Cosine Similarity & Normalization                           │   │
│   │  - Gensim Doc2Vec fallback engine                              │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. NLP & Machine Learning Pipeline

### Phase 1: Ingestion & Document Parsing (`src/resume_parser.py`)
- Ingests PDF format files (or raw text files).
- Utilizes `pdfplumber` for structured layout-preserving character extraction across multi-page documents.
- Includes exception boundaries to intercept corrupted PDF byte streams gracefully.

### Phase 2: Technical Entity Recognition (`src/skill_extractor.py`)
- Maintains a curated taxonomy (`TECH_SKILLS`) spanning programming languages, data engineering frameworks, ML/DL toolkits, databases, cloud providers, and visualization platforms.
- Employs word-boundary regular expressions (`\b<skill>\b`) with case-normalization to prevent sub-token false positives (e.g., matching "c" inside "react" or "sql" inside "nosql" inappropriately).

### Phase 3: Contextual Transformer Vectorization (`src/models.py`)
- Loads `sentence-transformers/bert-base-nli-mean-tokens` via Hugging Face Transformers and PyTorch.
- Tokenizes inputs with padding, sequence truncation at 512 tokens, and attention mask generation.
- Generates 768-dimensional token representations across all non-padding tokens.

### Phase 4: Attention-Weighted Mean Pooling (`src/models.py`)
- Aggregates token vectors into a single sentence/document embedding vector by taking the attention mask into account, avoiding distortion from zero-padded tokens.

### Phase 5: Pairwise Cosine Similarity Calculation
- Measures the angular distance between the 768-dimensional resume vector and job description vector.

### Phase 6: Hybrid Synthesis & Ranking (`src/resume_scanner.py`, `src/job_recommender.py`)
- Computes final hybrid match score ($0.60 \times \text{Semantic} + 0.40 \times \text{Skill}$).
- Determines matched competencies and missing prerequisites.
- Iterates across the job catalog (`data/jobs.csv`) and generates ranked recommendations.

---

## 4. Mathematical Formulation

### 1. Contextual BERT Token Extraction
Given an input token sequence $X = (x_1, x_2, \dots, x_N)$, BERT outputs contextual hidden state representations:

$$\mathbf{H} = \text{BERT}(X) = (\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_N), \quad \mathbf{h}_i \in \mathbb{R}^{768}$$

### 2. Attention-Weighted Mean Pooling
Padding tokens are masked out using attention mask $m_i \in \{0, 1\}$:

$$\mathbf{u} = \frac{\sum_{i=1}^{N} \mathbf{h}_i \cdot m_i}{\sum_{i=1}^{N} m_i}$$

### 3. Cosine Semantic Similarity
The cosine similarity between document embedding $\mathbf{u}$ and job description embedding $\mathbf{v}$ is:

$$\text{Sim}_{\cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{k=1}^{768} u_k v_k}{\sqrt{\sum_{k=1}^{768} u_k^2} \sqrt{\sum_{k=1}^{768} v_k^2}}$$

$$\text{Semantic Score} = \text{Sim}_{\cos}(\mathbf{u}, \mathbf{v}) \times 100$$

### 4. Technical Skill Overlap
Let $S_{\text{resume}}$ and $S_{\text{JD}}$ denote the detected technical skill sets:

$$\text{Skill Score} = \begin{cases} \left( \frac{|S_{\text{resume}} \cap S_{\text{JD}}|}{|S_{\text{JD}}|} \right) \times 100, & \text{if } |S_{\text{JD}}| > 0 \\ 0.0, & \text{if } |S_{\text{JD}}| = 0 \end{cases}$$

### 5. Hybrid Compatibility Score

$$\text{Final Score} = (0.60 \times \text{Semantic Score}) + (0.40 \times \text{Skill Score})$$

---

## 5. Component & Module Breakdown

| Module | Primary Responsibility | Key Functions / Classes |
| :--- | :--- | :--- |
| `app.py` | Streamlit web UI & CLI batch executor | Main UI layout, file upload, reactive tabs, CLI parser |
| `src/models.py` | Pretrained model loading & vector mathematics | `load_bert_model_and_tokenizer`, `mean_pooling`, `get_HF_embeddings`, `get_doc2vec_embeddings`, `cosine` |
| `src/resume_parser.py` | PDF and text file parsing | `extract_pdf_data`, `extract_text_data` |
| `src/skill_extractor.py` | Regex-based entity recognition | `extract_skills`, `calculate_skill_match`, `TECH_SKILLS` |
| `src/resume_scanner.py` | Multi-resume screening & hybrid scoring | `compare` (BERT & Doc2Vec pipelines) |
| `src/job_recommender.py` | Catalog-wide career recommendation | `recommend_jobs` |
| `src/utils.py` | Cross-platform file path resolution | `get_project_root`, `get_data_path` |
| `tests/test_core.py` | Comprehensive unit and integration test suite | `TestSkillExtractor`, `TestResumeParser`, `TestModels`, `TestResumeScanner`, `TestJobRecommender`, `TestUtils` |

---

## 6. Data Specifications & File Hierarchy

```text
ResumeIQ/
│
├── app.py                      # Primary Streamlit web application & CLI runner
├── requirements.txt            # Production dependencies with compatible versions
├── README.md                   # User-facing guide & documentation
├── DOCUMENTATION.md            # In-depth technical architecture manual
├── LICENSE                     # MIT Open Source License
├── run.bat                     # Windows one-click launcher
├── .gitignore                  # Git ignore rules for Python, cache, and OS artifacts
│
├── .streamlit/
│   └── config.toml             # Streamlit server and dark-mode configuration
│
├── src/                        # Modular package source
│   ├── __init__.py             # Public API exports
│   ├── models.py               # BERT model caching, mean pooling, cosine similarity
│   ├── resume_parser.py        # PDF and text document parsers
│   ├── skill_extractor.py      # Technical competency catalog & regex extractors
│   ├── resume_scanner.py       # Dual-engine comparison engine
│   ├── job_recommender.py      # Job catalog ranking logic
│   └── utils.py                # Safe cross-platform path resolution
│
├── data/
│   └── jobs.csv                # Canonical job listings dataset
│
├── notebooks/                  # Educational Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_model_exploration.ipynb
│   └── 04_resume_job_matching.ipynb
│
├── assets/
│   └── screenshots/            # Documentation screenshots
│
└── tests/
    ├── __init__.py
    └── test_core.py            # Automated test suite
```

---

## 7. Streamlit Community Cloud Deployment Guide

To deploy ResumeIQ on **Streamlit Community Cloud** without build errors:

### Step 1: Push Repository to GitHub
Ensure all code and the `data/` directory are committed to GitHub on the `main` branch.

### Step 2: Configure Streamlit Community Cloud
1. Log in to [share.streamlit.io](https://share.streamlit.io).
2. Click **"New app"**.
3. Select your repository: `mehuldangda/ResumeIQ-nlp-based-resume-screening-job-matching-system`.
4. Set **Main file path**: `app.py`.
5. Open **Advanced settings...** and select **Python 3.11** (or 3.10) as the Python version.
6. Click **Deploy!**.

> [!IMPORTANT]
> Do NOT use Python 3.14 on Streamlit Cloud. Python 3.11 ensures that prebuilt binary wheels for PyTorch, Transformers, Gensim, and Scikit-learn install instantly without compilation errors.

---

## 8. Learning Notebooks Guide

The `notebooks/` directory provides interactive Jupyter notebooks designed for students and developers:

1. **`01_data_exploration.ipynb`**: Learn how job datasets are structured, inspect required skills across job titles, and analyze demand.
2. **`02_text_preprocessing.ipynb`**: Understand PDF extraction with `pdfplumber`, text sanitization, and word-boundary regular expression extraction.
3. **`03_model_exploration.ipynb`**: Explore BERT tokenization, attention masks, mathematical mean pooling, and Doc2Vec vector generation.
4. **`04_resume_job_matching.ipynb`**: Step through cosine similarity calculation, skill gap diagnostics, hybrid score formulation, and catalog ranking.

---

## 9. Troubleshooting & Common Pitfalls

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `Failed building wheel for gensim / PyLongObject has no member named ob_digit` | Running on Python 3.14 preview where CPython C-API changed | Use **Python 3.11** or **Python 3.10**, which have prebuilt wheels |
| `Resource punkt_tab not found` | Recent NLTK versions require `punkt_tab` for tokenization | `src/models.py` automatically downloads `punkt` and `punkt_tab` via `ensure_nltk_resources()` |
| `FileNotFoundError: jobs.csv` | Relative working directory issues when running from subdirectories | `src/utils.py` dynamically anchors paths to project root |
| Slow first model run | Hugging Face model weights (~420MB) downloading on initial run | Uses Streamlit `@st.cache_resource` to cache model in memory permanently after initial download |

---

## 10. Future Roadmap & Scalability

- [ ] **Dynamic Named Entity Recognition (NER)**: Fine-tune a domain-specific transformer model for open-vocabulary technical skill and soft skill extraction.
- [ ] **Live Job API Integration**: Integrate real-time job board APIs (Adzuna, LinkedIn, Indeed) for live matching.
- [ ] **Automated Bullet Point Optimization**: Utilize LLMs to offer contextual phrasing recommendations for candidate resumes.
- [ ] **Exportable Candidate Diagnostic Reports**: Generate downloadable PDF scorecards.
