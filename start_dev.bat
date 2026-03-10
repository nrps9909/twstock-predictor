@echo off
chcp 65001 >nul 2>&1
title twstock-predictor (DEV)

cd /d "%~dp0"

echo ============================================
echo   twstock-predictor [DEV MODE]
echo ============================================
echo.

:: Kill leftover processes on port 8000 and 3000 (/T = kill entire process tree)
echo   Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    echo   Killing PID %%a on port 8000
    taskkill /T /F /PID %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000 " ^| findstr "LISTENING"') do (
    echo   Killing PID %%a on port 3000
    taskkill /T /F /PID %%a 2>nul
)
:: Kill any orphaned claude CLI (node) processes spawned by the API
taskkill /F /IM node.exe /FI "WINDOWTITLE eq claude*" 2>nul
timeout /t 1 /nobreak >nul

echo.
echo   Frontend : http://localhost:3000
echo   API      : http://127.0.0.1:8000
echo.
echo   Press Ctrl+C to stop both.
echo ============================================
echo.

:: Strip Claude Code env vars so claude -p subprocess won't be blocked
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=

:: Start API in background (same window, no pipe)
start /b .venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir src --reload-dir api

:: Wait for API to initialize (uvicorn --reload needs extra time)
timeout /t 5 /nobreak >nul

:: Start frontend in foreground — Ctrl+C kills npm, then we clean up API
cd web && npm run dev

:: After frontend exits (Ctrl+C or close), kill leftover API + subprocess tree
echo.
echo   Shutting down API...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /T /F /PID %%a >nul 2>&1
)
echo   Done.
