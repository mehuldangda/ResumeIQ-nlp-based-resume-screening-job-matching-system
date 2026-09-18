# ResumeIQ — Comprehensive Technical Documentation

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Architecture & Design](#2-system-architecture--design)
3. [NLP & Machine Learning Pipeline](#3-nlp--machine-learning-pipeline)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Component & Module Breakdown](#5-component--module-breakdown)
6. [Data Specifications](#6-data-specifications)
7. [Installation, Configuration & Execution](#7-installation-configuration--execution)
8. [Performance Optimization & Caching](#8-performance-optimization--caching)
9. [Edge Cases & Error Handling](#9-edge-cases--error-handling)
10. [Future Roadmap & Scalability](#10-future-roadmap--scalability)

---

## 1. Executive Summary

**ResumeIQ** is an enterprise-grade Natural Language Processing (NLP) and Machine Learning system engineered to evaluate candidate resumes against target job descriptions and rank candidates against available job catalogs.

Unlike traditional keyword-counting Applicant Tracking Systems (ATS) that suffer from high false-negative rates due to vocabulary mismatch, ResumeIQ employs a **dual-engine hybrid approach**:
1. **Contextual Semantic Matching**: Leverages a 110M-parameter transformer (`sentence-transformers/bert-base-nli-mean-tokens`) with attention-weighted mean pooling and cosine similarity to capture the conceptual essence of candidate experience.
2. **Explicit Competency Extraction**: Employs compiled word-boundary regular expressions across 45+ categorized technical domains to quantify hard technical prerequisite fulfillment and identify explicit skill gaps.

The result is an objective, weighted **Compatibility Score (60% Semantic + 40% Skill Match)** paired with actionable diagnostics and career path recommendations.

---

## 2. System Architecture & Design

ResumeIQ follows a modular, layered architecture ensuring separation of concerns between presentation, processing, modeling, and persistence.

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
- Accepts PDF format files (or raw text files).
- Utilizes `pdfplumber` for structured layout-preserving character extraction across multi-page documents.
- Includes exception boundaries to intercept corrupted PDF byte streams gracefully.

### Phase 2: Technical Entity Recognition (`src/skill_extractor.py`)
- Maintains a curated repository (`TECH_SKILLS`) spanning programming languages, data engineering frameworks, ML/DL toolkits, databases, cloud providers, and visualization platforms.
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
- Combines semantic similarity and skill overlap percentages into a composite score.
- Queries `data/jobs.csv`, performs batch comparison, and generates sorted job recommendations with skill gap diagnostics.

---

## 4. Mathematical Formulation

### 1. Attention-Weighted Mean Pooling
Given BERT output token vectors $\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_N \in \mathbb{R}^d$ and binary attention mask values $m_1, m_2, \dots, m_N \in \{0, 1\}$ (where $m_i = 1$ for genuine tokens and $0$ for padding tokens):

$$\mathbf{e} = \frac{\sum_{i=1}^{N} (\mathbf{h}_i \cdot m_i)}{\max\left(\sum_{i=1}^{N} m_i, \epsilon\right)}$$

where $\epsilon = 10^{-9}$ prevents division-by-zero.

### 2. Semantic Similarity Score
Given resume embedding $\mathbf{u} \in \mathbb{R}^d$ and job description embedding $\mathbf{v} \in \mathbb{R}^d$:

$$\text{Sim}_{\text{semantic}}(\mathbf{u}, \mathbf{v}) = \left( \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \, \|\mathbf{v}\|_2} \right) \times 100$$

### 3. Skill Overlap Score
Let $S_{\text{resume}} \subset \mathcal{S}$ be the set of skills detected in the resume, and $S_{\text{JD}} \subset \mathcal{S}$ be the set of skills detected in the job description:

$$\text{Score}_{\text{skill}} = \begin{cases} \left( \dfrac{|S_{\text{resume}} \cap S_{\text{JD}}|}{|S_{\text{JD}}|} \right) \times 100, & \text{if } |S_{\text{JD}}| > 0 \\[8pt] 0.0, & \text{if } |S_{\text{JD}}| = 0 \end{cases}$$

### 4. Hybrid Compatibility Formula
$$\text{Score}_{\text{final}} = \left( 0.60 \times \text{Sim}_{\text{semantic}} \right) + \left( 0.40 \times \text{Score}_{\text{skill}} \right)$$

---

## 5. Component & Module Breakdown

### `src/models.py`
- `load_bert_model_and_tokenizer(model_name)`: Cached loader returning Hugging Face AutoTokenizer and AutoModel.
- `mean_pooling(model_output, attention_mask)`: Tensor pooling algorithm.
- `get_HF_embeddings(sentences)`: Main inference pipeline generating pooled BERT tensors.
- `get_doc2vec_embeddings(JD, text_resume)`: Unsupervised Gensim Doc2Vec vectorizer for alternative dense representation.
- `cosine(embeddings1, embeddings2)`: Pairwise cosine similarity converting tensors/ndarrays safely to percentage metrics.

### `src/skill_extractor.py`
- `TECH_SKILLS`: Curated list of 45+ industry technical competencies.
- `extract_skills(text)`: Word-boundary regex extractor returning sorted unique skill lists.
- `calculate_skill_match(resume_skills, jd_skills)`: Mathematical set intersection calculation.

### `src/resume_parser.py`
- `extract_pdf_data(file)`: Multi-page PDF plain-text extractor with exception safety.
- `extract_text_data(file_path)`: UTF-8 plain-text file reader.

### `src/resume_scanner.py`
- `compare(resume_texts, JD_text, flag)`: Core comparator producing structured scoring dictionaries.

### `src/job_recommender.py`
- `recommend_jobs(resume_text, jobs_file)`: Iterates dataset, scores resume compatibility against each role, and returns sorted candidate matches.

### `src/utils.py`
- `get_project_root()`: Returns absolute Path to project base directory.
- `get_data_path(filename)`: Path resolver checking `data/` and project root for dataset compatibility.

---

## 6. Data Specifications

The job catalog is stored in `data/jobs.csv` with standard schema:
| Column | Type | Description | Example |
|---|---|---|---|
| `job_title` | `string` | Official title of the role | `Machine Learning Engineer` |
| `company` | `string` | Hiring organization | `ABC Technologies` |
| `job_description` | `string` | Textual requirements and key skills | `Python, Machine Learning, NLP, SQL, Git` |

---

## 7. Installation, Configuration & Execution

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

### Step-by-Step Setup
```powershell
# 1. Clone repository
git clone https://github.com/your-username/ResumeIQ.git
cd ResumeIQ

# 2. Setup Virtual Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install Requirements
pip install -r requirements.txt

# 4. Launch Application
.\run.bat
# or
streamlit run app.py
```

---

## 8. Performance Optimization & Caching

1. **Streamlit Resource Caching (`@st.cache_resource`)**:
   - Model weights (approx. 440 MB) and tokenizer are loaded **once** into memory upon initial startup. Subsequent runs reuse the in-memory PyTorch computation graph, eliminating cold-start latency.
2. **Batch Embedding Computation**:
   - Multiple sentences/resumes are tokenized simultaneously in single forward passes.
3. **No-Grad Inference Context (`torch.no_grad()`)**:
   - Disables autograd gradient tracking during inference, reducing memory allocation by ~50% and boosting throughput.

---

## 9. Edge Cases & Error Handling

- **Corrupted / Empty PDFs**: `extract_pdf_data` wraps parsing in try-catch blocks and returns clear diagnostics rather than raising unhandled exceptions.
- **Empty Job Description / Empty Resume**: Form validation warns users before triggering heavy transformer forward passes.
- **Missing Technical Skills**: Gracefully falls back to 100% semantic score weighing if no skills are detected in the job description.
- **NumPy 2.0 & PyTorch Tensor Compatibility**: Tensor conversion uses `.detach().cpu().numpy()` with dimension checking (`ndim == 1 -> reshape(1, -1)`) to avoid deprecation warnings and dimension mismatches.

---

## 10. Future Roadmap & Scalability

1. **Vector Database Migration**: Transition from in-memory CSV iteration to **FAISS**, **ChromaDB**, or **Qdrant** for sub-millisecond similarity search across millions of job listings.
2. **Dynamic Domain NER**: Implement fine-tuned transformer NER (e.g., RoBERTa/DeBERTa) for open-vocabulary entity extraction (certifications, degrees, patents).
3. **LLM Explanations**: Integrate local LLMs (e.g., Llama 3 / Gemma) to auto-generate personalized resume improvement suggestions based on identified skill gaps.
