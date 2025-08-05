@echo off
REM setup_environment.bat - Windows environment setup for GNN Fraud Detection

echo GNN Fraud Detection - Environment Setup (Windows)
echo =============================================

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python %PYTHON_VERSION% detected

REM Create virtual environment
echo Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch (CPU version)
echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install PyTorch Geometric
echo Installing PyTorch Geometric...
pip install torch-geometric

REM Install requirements
echo Installing project requirements...
pip install -r requirements.txt

REM Create directories
echo Setting up data directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\graphs mkdir data\graphs
if not exist results\models\checkpoints mkdir results\models\checkpoints
if not exist results\models\trained mkdir results\models\trained
if not exist logs\training mkdir logs\training
if not exist logs\inference mkdir logs\inference

REM Create .gitkeep files
echo. > data\raw\.gitkeep
echo. > data\processed\.gitkeep
echo. > data\graphs\.gitkeep
echo. > results\models\.gitkeep
echo. > results\models\checkpoints\.gitkeep
echo. > results\models\trained\.gitkeep

REM Run diagnostic
echo Running system diagnostic...
python diagnostic.py

echo.
echo =============================================
echo Environment setup complete!
echo.
echo Next steps:
echo 1. Download IEEE-CIS Fraud Detection dataset from Kaggle
echo 2. Place CSV files in data\raw\ directory
echo 3. Run 'python diagnostic.py' to verify complete setup
echo 4. Run 'python main.py' to start the system
echo.
echo To activate environment in future sessions:
echo venv\Scripts\activate.bat
echo.
pause
