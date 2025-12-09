@echo off
REM Local development startup script for Windows

echo ===================================
echo People Counter - Local Development
echo ===================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    exit /b 1
)

echo Python version:
python --version

REM Create virtual environment if not exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install backend dependencies
echo Installing backend dependencies...
pip install -r requirements.txt

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed
    exit /b 1
)

echo Node version:
node --version

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
if not exist "node_modules\" (
    call npm install
)
cd ..

REM Create necessary directories
echo Creating directories...
if not exist "cameras\" mkdir cameras
if not exist "models\" mkdir models
if not exist "config\" mkdir config
if not exist "temp\" mkdir temp

REM Start backend
echo Starting backend server...
start "Backend Server" cmd /k "venv\Scripts\activate.bat && python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for backend
timeout /t 5 /nobreak

REM Start frontend
echo Starting frontend server...
start "Frontend Server" cmd /k "cd frontend && npm start"

echo.
echo ===================================
echo Application started successfully!
echo ===================================
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Frontend: http://localhost:3000
echo.
echo Close the terminal windows to stop services
echo ===================================

pause
