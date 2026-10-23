@echo off
setlocal enabledelayedexpansion

echo =======================================================================
echo   Campus Knowledge Assistant (CampusRAG) - Local Demo Launcher
echo   Designed for German Tech Recruiter AI/ML Portfolio Evaluation
echo =======================================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating Python 3.12 virtual environment using uv / python...
    uv venv .venv --python 3.12 2>nul || python -m venv .venv
)

echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [*] Installing dependencies...
pip install -r requirements.txt

echo.
echo =======================================================================
echo [*] Starting FastAPI Backend on http://localhost:8000 ...
start "CampusRAG Backend" cmd /k "call .venv\Scripts\activate.bat && set PYTHONPATH=backend && uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 >nul

echo [*] Starting Streamlit Frontend on http://localhost:8501 ...
start "CampusRAG Frontend" cmd /k "call .venv\Scripts\activate.bat && set BACKEND_API_URL=http://localhost:8000/api/v1 && streamlit run frontend/app.py"

echo.
echo [OK] Both services launched successfully!
echo   - Backend API Docs: http://localhost:8000/docs
echo   - Interactive UI:   http://localhost:8501
echo =======================================================================
pause
