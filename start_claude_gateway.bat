@echo off
title Claude Code Anthropic Gateway (Port 4000)
cd /d "%~dp0"
call conda activate llm_server
litellm --config litellm_config.yaml --host 0.0.0.0 --port 4000
pause
