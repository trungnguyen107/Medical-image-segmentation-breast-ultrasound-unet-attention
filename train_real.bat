@echo off
REM Tải BUS-BRA (cần Internet), nhập, huấn luyện U-Net và U-Net+Attention trên ảnh thật, ghi SOURCES.md
cd /d "%~dp0"
python -m pip install -r requirements.txt
python fetch_data.py
python manage.py progress
python manage.py train-real --models M1,M4 --epochs 30 --size 128 --base 16
pause
