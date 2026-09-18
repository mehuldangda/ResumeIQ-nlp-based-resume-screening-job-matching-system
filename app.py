"""
ResumeIQ - AI Resume Screening and Job Recommendation System
Main Streamlit web application and command-line execution entrypoint.
"""

import sys
from pathlib import Path
import streamlit as st

from src.job_recommender import recommend_jobs
from src.resume_parser import extract_pdf_data, extract_text_data
from src.resume_scanner import compare
from src.skill_extractor import extract_skills

# ============================================================
# COMMAND LINE EXECUTION SUPPORT
# ============================================================
if len(sys.argv) > 1:
    if len(sys.argv) == 3:
        resume_path = sys.argv[1]
        jd_path = sys.argv[2]

        resume_data = extract_pdf_data(resume_path)
        jd_data = extract_text_data(jd_path)

        result = compare(
            [resume_data],
            jd_data,
            flag="HuggingFace-BERT"
        )
        print(result)
    sys.exit(0)


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="ResumeIQ - Resume Analysis & Job Recommendation",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown(
    """
    <style>
    /* Global Styles */
    .stApp {
        background: #0B1020;
        color: #F8FAFC;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5 {
        color: #F8FAFC !important;
    }

    p {
        color: #CBD5E1;
    }

    label {
        color: #CBD5E1 !important;
    }

    /* Sidebar Styles */
    section[data-testid="stSidebar"] {
        background: #080D1A;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] * {
        color: #CBD5E1;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
    }

    /* Brand Header */
    .brand {
        font-size: 1.65rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.8px;
    }

    .brand span {
        color: #818CF8;
    }

    .brand-subtitle {
        color: #64748B;
        font-size: 0.8rem;
    }

    /* Hero Component */
    .hero-box {
        background:
            radial-gradient(
                circle at 85% 20%,
                rgba(99,102,241,0.25),
                transparent 32%
            ),
            radial-gradient(
                circle at 10% 90%,
                rgba(139,92,246,0.12),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #11182D 0%,
                #151B35 50%,
                #10172A 100%
            );
        border: 1px solid #293653;
        border-radius: 26px;
        padding: 55px;
        margin-bottom: 40px;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 30px;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(129,140,248,0.3);
        color: #A5B4FC;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 3.2rem;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -2px;
        color: #F8FAFC;
        margin-bottom: 18px;
    }

    .hero-title span {
        color: #818CF8;
    }

    .hero-text {
        font-size: 1.08rem;
        line-height: 1.7;
        color: #94A3B8;
        max-width: 730px;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 35px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #94A3B8;
        font-size: 0.98rem;
        margin-bottom: 22px;
    }

    /* Cards */
    .feature-card,
    .step-card,
    .result-card,
    .job-card {
        background: #111827;
        border: 1px solid #263247;
        border-radius: 18px;
        padding: 25px;
        box-sizing: border-box;
    }

    .feature-card {
        min-height: 180px;
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #F8FAFC;
        font-weight: 750;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #94A3B8;
        line-height: 1.6;
        font-size: 0.92rem;
    }

    .step-card {
        min-height: 180px;
    }

    .step-number {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366F1, #4F46E5);
        color: white;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .step-title {
        color: #F8FAFC;
        font-weight: 750;
        font-size: 1.05rem;
        margin-bottom: 7px;
    }

    .step-text {
        color: #94A3B8;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    /* Inputs */
    div[data-testid="stFileUploader"] {
        background: #111827;
        border: 1px dashed #475569;
        border-radius: 16px;
        padding: 12px;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent;
    }

    textarea {
        background: #111827 !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1, #4F46E5);
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 750;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #818CF8, #6366F1);
        color: white;
    }

    /* Score Cards */
    .score-card {
        background: linear-gradient(135deg, #161D36, #111827);
        border: 1px solid #303B5B;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        box-sizing: border-box;
    }

    .score-label {
        color: #94A3B8;
        font-size: 0.76rem;
        font-weight: 650;
        margin-bottom: 7px;
    }

    .score-value {
        color: #818CF8;
        font-size: 2.1rem;
        font-weight: 850;
    }

    /* Job Card */
    .job-card {
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .job-title {
        color: #F8FAFC;
        font-size: 1.35rem;
        font-weight: 800;
    }

    .company-name {
        color: #818CF8;
        font-weight: 650;
        margin-top: 5px;
    }

    /* Skill Badges */
    .skill-badge {
        display: inline-block;
        padding: 7px 12px;
        margin: 4px;
        border-radius: 20px;
        background: #1E293B;
        border: 1px solid #334155;
        color: #CBD5E1;
        font-size: 0.82rem;
    }

    .skill-good {
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.25);
        color: #86EFAC;
    }

    .skill-bad {
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.25);
        color: #FCA5A5;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748B;
        padding-top: 45px;
        padding-bottom: 20px;
        font-size: 0.84rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="brand">Resume<span>IQ</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Resume Analysis & Job Matching'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ Analysis Settings")

    flag = st.selectbox(
        "Embedding Model",
        [
            "HuggingFace-BERT",
            "Doc2Vec"
        ],
        key="embedding_model"
    )

    st.divider()

    st.markdown("### ✨ ResumeIQ")

    st.caption(
        "Analyze resumes, compare job requirements, "
        "identify skill gaps and explore "
        "relevant career opportunities."
    )


# ============================================================
# NAVIGATION TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(
    [
        "🏠 Home",
        "📊 Results",
        "💼 Job Recommendations"
    ]
)


# ============================================================
# TAB 1: HOME
# ============================================================
with tab1:
    # Hero Box
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-badge">
                ✨ Career Intelligence
            </div>
            <div class="hero-title">
                Understand your resume and
                <span>discover your opportunities.</span>
            </div>
            <div class="hero-text">
                Analyze your resume against job descriptions,
                identify skill gaps and explore
                opportunities that match your profile.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Features Section
    st.markdown(
        """
        <div class="section-title">
            Understand Your Career Fit
        </div>
        <div class="section-subtitle">
            One platform for resume analysis, job matching
            and career recommendations.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Resume Analysis</div>
                <div class="feature-text">
                    Extract important information and technical
                    skills from your resume.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Job Matching</div>
                <div class="feature-text">
                    Compare your resume with a job description
                    using semantic and skill-based analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💼</div>
                <div class="feature-title">Job Recommendations</div>
                <div class="feature-text">
                    Explore relevant job roles and understand
                    the skills you can improve.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # How It Works
    st.markdown(
        """
        <div class="section-title">
            How it works
        </div>
        <div class="section-subtitle">
            Review useful insights in four simple steps.
        </div>
        """,
        unsafe_allow_html=True
    )

    steps = [
        ("01", "Upload Resume", "Upload your PDF resume to begin the analysis."),
        ("02", "Add Job Description", "Paste the job description you want to target."),
        ("03", "Analyze", "The system evaluates semantic similarity and skills."),
        ("04", "Get Insights", "View your score, skill gaps, and job recommendations.")
    ]

    columns = st.columns(4)
    for column, step in zip(columns, steps):
        with column:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">
                        {step[0]}
                    </div>
                    <div class="step-title">
                        {step[1]}
                    </div>
                    <div class="step-text">
                        {step[2]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Resume Upload & Job Description Form
    st.markdown(
        """
        <div class="section-title">
            Analyze Your Resume
        </div>
        <div class="section-subtitle">
            Upload your resume and enter the target job description.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose your resume PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_uploader"
    )

    JD = st.text_area(
        "Job Description",
        height=220,
        placeholder=(
            "Paste the job description here...\n\n"
            "Example:\n"
            "We are looking for a Python developer "
            "with experience in machine learning, "
            "SQL and REST APIs."
        ),
        key="job_description"
    )

    comp_pressed = st.button(
        "🚀 Analyze Resume",
        use_container_width=True,
        key="analyze_resume"
    )

    # Processing Analysis
    if comp_pressed:
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
        elif not JD.strip():
            st.warning("Please enter a job description.")
        else:
            with st.spinner("Analyzing your resume..."):
                resume_texts = []
                for uploaded_file in uploaded_files:
                    text = extract_pdf_data(uploaded_file)
                    resume_texts.append(text)

                results = compare(
                    resume_texts,
                    JD,
                    flag=flag
                )

            # Store in session state
            st.session_state["results"] = results
            st.session_state["resume_names"] = [f.name for f in uploaded_files]
            st.session_state["resume_texts"] = resume_texts

            st.success(
                "Resume analysis completed successfully! "
                "Open the Results tab to review your results."
            )

            # Detected Resume Skills Summary
            st.markdown(
                """
                <div class="section-title">
                    Detected Resume Skills
                </div>
                """,
                unsafe_allow_html=True
            )

            for i, resume_text in enumerate(resume_texts):
                st.markdown(f"### 📄 {uploaded_files[i].name}")
                resume_skills = extract_skills(resume_text)

                if resume_skills:
                    skill_html = "".join(
                        f'<span class="skill-badge">{skill}</span>'
                        for skill in resume_skills
                    )
                    st.markdown(skill_html, unsafe_allow_html=True)
                else:
                    st.warning("No technical skills detected.")


# ============================================================
# TAB 2: RESULTS
# ============================================================
with tab2:
    st.markdown(
        """
        <div class="section-title">
            📊 Resume Matching Results
        </div>
        <div class="section-subtitle">
            Review how closely your resume matches
            the target job description.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "results" not in st.session_state:
        st.info(
            "Upload a resume and run an analysis from "
            "the Home tab to view your results here."
        )
    else:
        results = st.session_state["results"]
        resume_names = st.session_state.get("resume_names", [])

        if not results:
            st.warning("No analysis results were generated.")
        else:
            for i, result in enumerate(results):
                resume_name = resume_names[i] if i < len(resume_names) else f"Resume {i + 1}"

                st.markdown(f"## 📄 {resume_name}")

                # Score Metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        f"""
                        <div class="score-card">
                            <div class="score-label">
                                🎯 FINAL MATCH SCORE
                            </div>
                            <div class="score-value">
                                {result.get("final_score", 0):.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div class="score-card">
                            <div class="score-label">
                                🧠 SEMANTIC SIMILARITY
                            </div>
                            <div class="score-value">
                                {result.get("semantic_score", 0):.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col3:
                    st.markdown(
                        f"""
                        <div class="score-card">
                            <div class="score-label">
                                🛠️ SKILL MATCH
                            </div>
                            <div class="score-value">
                                {result.get("skill_score", 0):.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.write("")

                # Skills Comparison Breakdown
                matched = result.get("matched_skills", [])
                missing = result.get("missing_skills", [])
                jd_skills = result.get("jd_skills", [])

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### ✅ Matched Skills")
                    if matched:
                        skill_html = "".join(
                            f'<span class="skill-badge skill-good">{skill}</span>'
                            for skill in matched
                        )
                        st.markdown(skill_html, unsafe_allow_html=True)
                    else:
                        st.info("No matching skills detected.")

                with col2:
                    st.markdown("### ❌ Skills to Improve")
                    if missing:
                        skill_html = "".join(
                            f'<span class="skill-badge skill-bad">{skill}</span>'
                            for skill in missing
                        )
                        st.markdown(skill_html, unsafe_allow_html=True)
                    elif jd_skills:
                        st.success("Excellent! No required skills are missing.")
                    else:
                        st.info("No technical skills detected.")

                # Job Description Required Skills
                st.markdown("### 📋 Skills Required in the Job Description")
                if jd_skills:
                    skill_html = "".join(
                        f'<span class="skill-badge">{skill}</span>'
                        for skill in jd_skills
                    )
                    st.markdown(skill_html, unsafe_allow_html=True)
                else:
                    st.info("No technical skills were detected in the Job Description.")

                st.divider()


# ============================================================
# TAB 3: JOB RECOMMENDATIONS
# ============================================================
with tab3:
    st.markdown(
        """
        <div class="section-title">
            💼 Job Recommendations
        </div>
        <div class="section-subtitle">
            Explore roles that match your resume
            and identify skills you can improve.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "resume_texts" not in st.session_state:
        st.info("Upload and analyze a resume from the Home tab first.")
    else:
        resume_texts = st.session_state["resume_texts"]
        resume_names = st.session_state.get("resume_names", [])

        if not resume_texts:
            st.info("Please upload a resume first.")
        else:
            selected_resume = st.selectbox(
                "Select resume",
                resume_names,
                key="selected_resume"
            )

            selected_index = resume_names.index(selected_resume)
            resume_text = resume_texts[selected_index]

            st.info(f"Recommendations based on: **{selected_resume}**")

            find_jobs = st.button(
                "🔎 Find Matching Jobs",
                use_container_width=True,
                key="find_jobs"
            )

            if find_jobs:
                with st.spinner("Finding relevant job matches..."):
                    recommendations = recommend_jobs(resume_text)

                if not recommendations:
                    st.warning("No suitable jobs were found.")
                else:
                    st.success(f"Found {len(recommendations)} matching job opportunities.")

                    for i, job in enumerate(recommendations):
                        job_title = job.get("job_title", "Unknown Job")
                        company = job.get("company", "Unknown Company")

                        st.markdown(
                            f"""
                            <div class="job-card">
                                <div class="job-title">
                                    {i + 1}. {job_title}
                                </div>
                                <div class="company-name">
                                    {company}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(
                                f"""
                                <div class="score-card">
                                    <div class="score-label">
                                        🎯 MATCH SCORE
                                    </div>
                                    <div class="score-value">
                                        {job.get("match_score", 0):.2f}%
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with col2:
                            st.markdown(
                                f"""
                                <div class="score-card">
                                    <div class="score-label">
                                        🧠 SEMANTIC MATCH
                                    </div>
                                    <div class="score-value">
                                        {job.get("semantic_score", 0):.2f}%
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with col3:
                            st.markdown(
                                f"""
                                <div class="score-card">
                                    <div class="score-label">
                                        🛠️ SKILL MATCH
                                    </div>
                                    <div class="score-value">
                                        {job.get("skill_score", 0):.2f}%
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        # Match Assessment Level
                        match_score = job.get("match_score", 0)
                        if match_score >= 80:
                            potential = "🟢 Strong"
                            recommendation = "Strong Match"
                        elif match_score >= 65:
                            potential = "🟡 Good"
                            recommendation = "Good Match"
                        elif match_score >= 50:
                            potential = "🟠 Moderate"
                            recommendation = "Potential Match"
                        else:
                            potential = "🔴 Low"
                            recommendation = "Lower Match"

                        st.markdown("### 📈 Match Assessment")
                        if match_score >= 65:
                            st.success(f"{potential} — {recommendation}")
                        elif match_score >= 50:
                            st.warning(f"{potential} — {recommendation}")
                        else:
                            st.error(f"{potential} — {recommendation}")

                        # Skills Breakdown for Job
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### ✅ Skills You Have")
                            matched = job.get("matched_skills", [])
                            if matched:
                                skill_html = "".join(
                                    f'<span class="skill-badge skill-good">{skill}</span>'
                                    for skill in matched
                                )
                                st.markdown(skill_html, unsafe_allow_html=True)
                            else:
                                st.info("No matching technical skills detected.")

                        with col2:
                            st.markdown("### ❌ Skills to Improve")
                            missing = job.get("missing_skills", [])
                            if missing:
                                skill_html = "".join(
                                    f'<span class="skill-badge skill-bad">{skill}</span>'
                                    for skill in missing
                                )
                                st.markdown(skill_html, unsafe_allow_html=True)
                            else:
                                st.success("No detected skill gaps.")

                        st.divider()


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        ResumeIQ • Resume Analysis & Job Matching
        <br><br>
        Built with Python • Streamlit • NLP • Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
