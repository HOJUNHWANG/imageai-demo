@echo off
setlocal enabledelayedexpansion
title Morrow Local Image Studio

echo ========================================
echo   Morrow - Setup ^& Start
echo ========================================
echo.

:: ── Step 0: Check Python ─────────────────────────────────────────────────────
echo [0/5] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11 from https://python.org
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo        Found Python %PY_VER%

:: ── Step 1: Check Node.js ────────────────────────────────────────────────────
echo [1/5] Checking Node.js...
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    pause & exit /b 1
)
for /f "tokens=1" %%v in ('node --version 2^>^&1') do set NODE_VER=%%v
echo        Found Node.js %NODE_VER%

:: ── Step 2: Create virtualenv if missing ─────────────────────────────────────
echo [2/5] Checking virtual environment...
if not exist "%~dp0venv311\Scripts\python.exe" (
    echo        Creating venv311...
    python -m venv "%~dp0venv311"
    if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
    echo        venv311 created.
) else (
    echo        venv311 already exists.
)

set VENV_PY=%~dp0venv311\Scripts\python.exe

:: ── Step 3: Install Python dependencies ──────────────────────────────────────
echo [3/5] Checking Python dependencies...

:: Clean up corrupted partial installs (e.g. ~orch, ~ympy left by interrupted pip)
for /d %%D in ("%~dp0venv311\Lib\site-packages\~*") do (
    echo        Removing corrupted package: %%~nxD
    rmdir /s /q "%%D" >nul 2>&1
)

:: Determine GPU presence once
set HAS_GPU=0
nvidia-smi >nul 2>&1
if not errorlevel 1 set HAS_GPU=1

:: ── 3a: Install / verify torch (CUDA or CPU) ─────────────────────────────────
set TORCH_OK=0
"%VENV_PY%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 set TORCH_OK=1

if "%HAS_GPU%"=="1" (
    if "%TORCH_OK%"=="0" (
        echo        Installing CUDA-enabled PyTorch ^(cu124^)...
        "%VENV_PY%" -m pip install torch torchvision ^
            --index-url https://download.pytorch.org/whl/cu124 -q
        if errorlevel 1 ( echo [ERROR] Failed to install CUDA PyTorch. & pause & exit /b 1 )
        echo        CUDA PyTorch installed.
    ) else (
        echo        PyTorch with CUDA already installed.
    )
) else (
    :: No GPU — install CPU torch if not present at all
    "%VENV_PY%" -c "import torch" >nul 2>&1
    if errorlevel 1 (
        echo        No GPU detected — installing CPU PyTorch...
        "%VENV_PY%" -m pip install torch torchvision -q
    ) else (
        echo        PyTorch ^(CPU mode^) already installed.
    )
)

:: Install only the xFormers wheel whose ABI matches torch 2.6 + CUDA 12.4.
:: Other torch/CUDA combinations safely keep native SDPA.
if "%HAS_GPU%"=="1" (
    "%VENV_PY%" -c "import torch; exit(0 if torch.__version__.startswith('2.6.') and torch.version.cuda == '12.4' else 1)" >nul 2>&1
    if not errorlevel 1 (
        "%VENV_PY%" -c "import xformers; exit(0 if xformers.__version__ == '0.0.29.post2' else 1)" >nul 2>&1
        if errorlevel 1 (
            echo        Installing compatible xFormers 0.0.29.post2...
            "%VENV_PY%" -m pip install --no-deps xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu124 -q
            if errorlevel 1 echo        xFormers unavailable - native SDPA will be used.
        ) else (
            echo        Compatible xFormers already installed.
        )
    ) else (
        echo        Torch/CUDA ABI differs - keeping native SDPA instead of forcing xFormers.
    )
)

:: ── 3b: Install remaining requirements ───────────────────────────────────────
set SENTINEL=%~dp0venv311\.installed_hash
set REQ_FILE=%~dp0requirements.txt
for %%F in ("%REQ_FILE%") do set REQ_SIG=%%~zF-%%~tF

set NEED_INSTALL=1
if exist "%SENTINEL%" (
    set /p CACHED_SIG=<"%SENTINEL%"
    if "!CACHED_SIG!"=="%REQ_SIG%" set NEED_INSTALL=0
)

if "!NEED_INSTALL!"=="1" (
    echo        Installing packages from requirements.txt...
    "%VENV_PY%" -m pip install --upgrade pip -q
    "%VENV_PY%" -m pip install -r "%REQ_FILE%" -q
    if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )
    echo %REQ_SIG%> "%SENTINEL%"
    echo        Packages installed.
) else (
    echo        Packages up to date.
)

:: ── Step 4: Install frontend dependencies ────────────────────────────────────
echo [4/6] Checking frontend dependencies...
if not exist "%~dp0frontend\node_modules\next\package.json" (
    echo        Running npm install...
    pushd "%~dp0frontend"
    call npm install --prefer-offline --no-audit --no-fund --loglevel error
    if errorlevel 1 ( echo [ERROR] npm install failed. & popd & pause & exit /b 1 )
    popd
    echo        Frontend dependencies installed.
) else (
    echo        node_modules already present.
)

:: ── Step 5: Load .env.local (model configuration, never committed) ──────────
if exist "%~dp0.env.local" (
    echo [ENV]  Loading .env.local...
    for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env.local") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" if not "%%A"=="" (
            set "%%A=%%B"
        )
    )
)

:: ── Step 6: Clean up old processes and start ─────────────────────────────────
echo [6/6] Starting servers...

for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000 " ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
del /f "%~dp0frontend\.next\dev\lock" >nul 2>&1
timeout /t 1 /nobreak >nul

start "Morrow Backend" cmd /k "cd /d %~dp0 && venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul
start "Morrow Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo   Backend:  http://127.0.0.1:8000
echo   Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to open the app in your browser...
pause >nul
start http://localhost:3000
