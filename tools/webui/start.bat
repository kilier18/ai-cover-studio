@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\"
set "PY=%~dp0..\..\py311\pythonw.exe"
if not exist "%PY%" set "PY=pythonw.exe"
start "" "%PY%" "%~dp0app.py" --port 8765 --open
exit
