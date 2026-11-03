#!/usr/bin/env bash
# Chạy ứng dụng web trên Linux/macOS
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -c "import torch, flask, cv2, scipy" 2>/dev/null || pip install -r requirements.txt
python app.py --open "$@"
