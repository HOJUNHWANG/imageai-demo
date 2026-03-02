@echo off
echo ========================================
echo   ImageAI Studio - Starting...
echo ========================================
echo.

:: Kill only processes using our specific ports (8000 and 3000)
echo [0/2] Cleaning up old processes on ports 8000 and 3000...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000 " ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
timeout /t 1 /nobreak > nul

:: Remove Next.js lock file if it exists
del /f "%~dp0frontend\.next\dev\lock" >nul 2>&1

:: Start FastAPI Backend
echo [1/2] Starting FastAPI Backend (port 8000)...
start "ImageAI Backend" cmd /k "cd /d %~dp0 && venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

:: Wait for backend to start
timeout /t 3 /nobreak > nul

:: Start Next.js Frontend
echo [2/2] Starting Next.js Frontend (port 3000)...
start "ImageAI Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo   Backend:  http://127.0.0.1:8000
echo   Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to open the app in your browser...
pause > nul
start http://localhost:3000
