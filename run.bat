@echo off
title ResumeIQ - Resume Analysis and Job Recommendation System
echo ============================================================
echo Starting ResumeIQ Web Application...
echo ============================================================

REM Activate virtual environment if available
if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

streamlit run app.py
pause
