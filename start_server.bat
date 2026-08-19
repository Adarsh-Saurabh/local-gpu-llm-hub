@echo off
title RTX 5000 GPU LLM Hub & Server
cd /d "%~dp0"
echo ============================================================
echo   Starting RTX 5000 GPU LLM Hub & Web Dashboard...
echo ============================================================
echo.
echo Activating isolated conda environment (llm_server)...
call conda activate llm_server
echo.
echo Server is launching on port 11434.
echo Web Dashboard URL: http://localhost:11434/
echo Remote Tailscale URL: http://100.78.12.27:11434/
echo.
echo Opening Web Dashboard in your browser...
start http://localhost:11434/
echo.
python server.py --host 0.0.0.0 --port 11434
pause
