@echo off
chcp 65001 >nul 2>&1
title twstock-predictor (DEV)

cd /d "%~dp0"

echo ============================================
echo   twstock-predictor [DEV MODE]
echo ============================================
echo.

:: Kill leftover processes on port 8000 and 3000
echo   Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo   Frontend : http://localhost:3000
echo   API      : http://127.0.0.1:8000
echo.
echo   Press Ctrl+C to stop.
echo ============================================
echo.

:: Strip Claude Code env vars so claude -p subprocess won't be blocked
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=

:: Start API in background window
start "twstock-API" cmd /c "set CLAUDECODE= & set CLAUDE_CODE_ENTRYPOINT= & .venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir src --reload-dir api"

:: Wait a moment for API to initialize
timeout /t 2 /nobreak >nul

:: Start frontend (stays in foreground)
cd web && npm run dev

pause
