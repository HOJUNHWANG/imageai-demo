@echo off
setlocal enabledelayedexpansion
title ImageAI Studio

echo ========================================
echo   ImageAI Studio - Setup ^& Start
echo ========================================
echo.

:: ── Step 0: Check Python ─────────────────────────────────────────────────────
echo [0/5] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11 from https://python.org
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo        Found Python %PY_VER%

:: ── Step 1: Check Node.js ────────────────────────────────────────────────────
echo [1/5] Checking Node.js...
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)
for /f "tokens=1" %%v in ('node --version 2^>^&1') do set NODE_VER=%%v
echo        Found Node.js %NODE_VER%

:: ── Step 2: Create virtualenv if missing ─────────────────────────────────────
echo [2/5] Checking virtual environment...
if not exist "%~dp0venv311\Scripts\python.exe" (
    echo        Creating venv311...
    python -m venv "%~dp0venv311"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo        venv311 created.
) else (
    echo        venv311 already exists.
)

set VENV_PY=%~dp0venv311\Scripts\python.exe
set VENV_PIP=%~dp0venv311\Scripts\pip.exe

:: ── Step 3: Install Python dependencies ──────────────────────────────────────
echo [3/5] Checking Python dependencies...

:: Use a sentinel file to skip reinstall if requirements.txt hasn't changed
set SENTINEL=%~dp0venv311\.installed_hash
set REQ_FILE=%~dp0requirements.txt

:: Compute a simple hash (file size + last-modified) as a cheap change check
for %%F in ("%REQ_FILE%") do set REQ_SIG=%%~zF-%%~tF

set NEED_INSTALL=1
if exist "%SENTINEL%" (
    set /p CACHED_SIG=<"%SENTINEL%"
    if "!CACHED_SIG!"=="%REQ_SIG%" set NEED_INSTALL=0
)

if "!NEED_INSTALL!"=="1" (
    echo        Installing / updating packages from requirements.txt...
    "%VENV_PIP%" install --upgrade pip -q
    "%VENV_PIP%" install -r "%REQ_FILE%" -q
    if errorlevel 1 (
        echo [ERROR] pip install failed. Check your internet connection.
        pause
        exit /b 1
    )

    :: Check for NVIDIA GPU and install xformers if available
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        echo        NVIDIA GPU detected — installing GPU extras ^(xformers^)...
        "%VENV_PIP%" install -r "%~dp0requirements-gpu.txt" -q
    )

    echo %REQ_SIG%> "%SENTINEL%"
    echo        Dependencies installed.
) else (
    echo        Dependencies up to date.
)

:: ── Step 4: Install frontend dependencies ────────────────────────────────────
echo [4/5] Checking frontend dependencies...
if not exist "%~dp0frontend\node_modules\.package-lock.json" (
    echo        Running npm install in frontend\...
    pushd "%~dp0frontend"
    call npm install --prefer-offline --loglevel error
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        popd
        pause
        exit /b 1
    )
    popd
    echo        Frontend dependencies installed.
) else (
    echo        node_modules already present.
)

:: ── Step 5: Clean up old processes and start ─────────────────────────────────
echo [5/5] Starting servers...

for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000 " ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
del /f "%~dp0frontend\.next\dev\lock" >nul 2>&1
timeout /t 1 /nobreak >nul

start "ImageAI Backend" cmd /k "cd /d %~dp0 && venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul
start "ImageAI Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo   Backend:  http://127.0.0.1:8000
echo   Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to open the app in your browser...
pause >nul
start http://localhost:3000
