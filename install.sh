#!/usr/bin/env bash
set -e

echo "============================================================"
echo "      Local GPU LLM Hub - Linux / macOS Setup Script"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "[1/3] Creating Python Virtual Environment (venv)..."
    python3 -m venv venv
fi

echo "[2/3] Activating Virtual Environment..."
source venv/bin/activate

echo "[3/3] Installing Base Requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Detect OS & Hardware
OS="$(uname -s)"
ARCH="$(uname -m)"

echo ""
echo "Probing Hardware: $OS ($ARCH)..."

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "[OK] Apple Silicon detected! Compiling llama-cpp-python with Metal acceleration..."
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir
elif command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected! Installing CUDA-accelerated llama-cpp-python..."
    pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
else
    echo "[INFO] No discrete GPU detected. Installing standard CPU acceleration..."
    pip install llama-cpp-python
fi

echo ""
echo "============================================================"
echo "   Installation Complete! Run ./start.sh to launch the Hub."
echo "============================================================"
