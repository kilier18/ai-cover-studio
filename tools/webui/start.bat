@echo off
chcp 65001 >nul
cd /d "%~dp0."
start "" "%~dp0py311\pythonw.exe" "%~dp0tools\webui\app.py" --port 8765 --open
exit
