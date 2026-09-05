@echo off
REM CC1 Vision - one-click Windows launcher.
REM Creates the virtual environment on first run, then starts the dashboard.
REM Training the CNN is NOT done here: it takes about half an hour, so it is a
REM deliberate step you run once with train_classifier.py.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [CC1 Vision] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :nopython
    echo [CC1 Vision] Installing dependencies (this includes TensorFlow)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements-ml.txt
    if errorlevel 1 (
        echo.
        echo [CC1 Vision] TensorFlow could not be installed - it needs Python 3.10-3.12.
        echo Falling back to the base requirements. The dashboard will run on the
        echo heuristic classifier and will say so in the sidebar.
        ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    )
)

if not exist "samples\sample_floor.jpg" (
    echo [CC1 Vision] Generating the sample floor image...
    ".venv\Scripts\python.exe" sample_generator.py
)

if not exist "models\stain_classifier.keras" (
    echo.
    echo [CC1 Vision] No trained model found in models\.
    echo The dashboard will start on the heuristic fallback.
    echo To train the CNN ^(about 30 minutes, one time^):
    echo     .venv\Scripts\python.exe dataset_generator.py
    echo     .venv\Scripts\python.exe train_classifier.py
    echo.
)

echo [CC1 Vision] Starting the dashboard at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py
goto :eof

:nopython
echo.
echo [CC1 Vision] Python was not found on your PATH.
echo Install Python 3.10, 3.11 or 3.12 from python.org and tick
echo "Add Python to PATH" during installation, then run this file again.
echo TensorFlow has no wheels for Python 3.13 or 3.14 yet.
pause
