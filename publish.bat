@echo off
chcp 65001 >nul
cd /d "%~dp0."
echo [1/2] remote ...
git remote get-url origin >nul 2>&1 && git remote set-url origin https://github.com/Pinn/ai-cover-studio.git || git remote add origin https://github.com/Pinn/ai-cover-studio.git
echo [2/2] push (browser login may pop up once) ...
git push -u origin main
echo.
echo Done. Check https://github.com/Pinn/ai-cover-studio
pause
