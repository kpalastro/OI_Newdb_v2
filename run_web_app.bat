@echo off
cd /d "%~dp0"
echo ==========================================
echo     OI Gemini - Trading System
echo ==========================================
echo.
echo Working Directory: %CD%
echo.
echo Checking virtual environment...
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found!
    echo Please create it with: python -m venv .venv
    echo.
)
echo.
echo Starting server...
echo.
python oi_tracker_kimi_new.py
echo.
echo.
echo Server stopped.
pause

