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
:server_loop
echo [%date% %time%] Launching LLM Server Watchdog...
python server.py --host 0.0.0.0 --port 11434
echo.
echo [WATCHDOG] Server stopped. Automatically restarting in 3 seconds...
echo (Press Ctrl+C to permanently close)
timeout /t 3 /nobreak >nul
goto server_loop
