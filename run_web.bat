@echo off
chcp 65001 >nul
cd /d %~dp0
echo === Ung dung phan doan khoi u sieu am vu (U-Net + Attention) ===
if not exist .venv (
  echo Dang tao moi truong ao Python...
  python -m venv .venv || (echo Can cai Python 3.10+ va them vao PATH. & pause & exit /b 1)
)
call .venv\Scripts\activate
python -c "import torch, flask, cv2, scipy" 2>nul || (
  echo Dang cai thu vien lan dau, vui long cho...
  python -m pip install --upgrade pip
  pip install -r requirements.txt || (echo Cai dat that bai. & pause & exit /b 1)
)
python app.py --open
pause
