@echo off
chcp 65001 >nul
set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "127.0.0.1:8765" ^| findstr LISTENING') do (
  taskkill /F /T /PID %%p >nul 2>&1
  set FOUND=1
)
if "%FOUND%"=="1" (echo [OK] studio stopped) else (echo [--] nothing running on 8765)
