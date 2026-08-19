@echo off
title Model Downloader
cd /d "%~dp0"
call conda activate llm_server
python download_model.py
pause
