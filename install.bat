@echo off
title Local GPU LLM Hub - Windows Installer
cd /d "%~dp0"
echo ============================================================
echo       Local GPU LLM Hub - Automated Windows Setup
echo ============================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10, 3.11, or 3.12 from python.org
    pause
    exit /b 1
)

:: Create Virtual Environment if it doesn't exist
if not exist "venv" (
    echo [1/3] Creating Python Virtual Environment (venv)...
    python -m venv venv
)

echo [2/3] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo [3/3] Installing Base Requirements...
pip install --upgrade pip
pip install -r requirements.txt

:: Auto-detect NVIDIA CUDA
echo.
echo Probing GPU Hardware...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU Detected! Installing CUDA-accelerated llama-cpp-python...
    pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
) else (
    echo [INFO] No NVIDIA GPU found. Installing standard CPU acceleration...
    pip install llama-cpp-python
)

echo.
echo ============================================================
echo   Installation Complete! Run start.bat to launch the Hub.
echo ============================================================
pause
