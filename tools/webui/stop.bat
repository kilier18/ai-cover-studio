@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set found=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8765" ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1
  set found=1
)
if "!found!"=="1" (echo [ok] service stopped.) else (echo [i] service not running.)
timeout /t 2 >nul
exit
