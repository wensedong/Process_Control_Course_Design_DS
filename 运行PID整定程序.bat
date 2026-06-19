@echo off
chcp 65001 >nul
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 PID 参数整定程序...
python app.py
pause
