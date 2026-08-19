#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "============================================================"
echo "   Starting Local GPU LLM Hub & Web Dashboard..."
echo "============================================================"
echo ""

if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Attempt to open browser
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:11434/" >/dev/null 2>&1 &
elif command -v open &> /dev/null; then
    open "http://localhost:11434/" >/dev/null 2>&1 &
fi

python3 server.py --host 0.0.0.0 --port 11434
