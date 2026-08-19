@echo off
title Local GPU LLM Hub & Server
cd /d "%~dp0"

echo ============================================================
echo   Starting Local GPU LLM Hub & Web Dashboard...
echo ============================================================
echo.

:: Check for venv or Conda
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    call conda activate llm_server >nul 2>&1
)

echo Opening Web Dashboard in your browser...
start http://localhost:11434/
echo.
python server.py --host 0.0.0.0 --port 11434
pause
