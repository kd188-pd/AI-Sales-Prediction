@echo off
setlocal
cd /d "%~dp0"
echo =====================================================
echo  AI-Based Sales Prediction and Business Analytics
echo =====================================================
if not exist ".venv\Scripts\python.exe" (
  echo [1/5] Creating Python virtual environment...
  py -m venv .venv
  if errorlevel 1 (
    echo Could not create environment using py. Trying python...
    python -m venv .venv
    if errorlevel 1 goto failed
  )
)
echo [2/5] Installing required packages...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo [3/5] Generating sample data...
call ".venv\Scripts\python.exe" generate_data.py
if errorlevel 1 goto failed
echo [4/5] Training model...
call ".venv\Scripts\python.exe" train_model.py
if errorlevel 1 goto failed
echo [5/5] Starting Flask app...
echo Open http://127.0.0.1:5000 in your browser.
call ".venv\Scripts\python.exe" app.py
goto end
:failed
echo.
echo Setup failed. Please read the error above and check that Python is installed.
pause
:end
endlocal
