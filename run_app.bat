@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting Lip Reading System (MCA Final Year Project)...
echo ========================================================

if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv\
    echo Please run: py -3.10 -m venv venv
    pause
    exit /b
)

echo Using project virtual environment: %~dp0venv
call .\venv\Scripts\activate.bat

echo Starting Streamlit app...
".\venv\Scripts\python.exe" -m streamlit run app.py

pause
