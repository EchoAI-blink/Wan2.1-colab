@echo off
echo Setting up environment for WAN 2.1 on T4 GPU...

:: Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.10+ and try again.
    exit /b 1
)

:: Check for CUDA installation
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo CUDA is not installed or not in PATH. Please install CUDA 11.8+ for optimal performance.
    echo Continuing anyway, but performance may be affected...
)

:: Create a virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install other dependencies
echo Installing other dependencies...
pip install -r requirements.txt

:: Create necessary directories
if not exist output mkdir output
if not exist models mkdir models

echo Environment setup complete!
echo.
echo To use the WAN 2.1 model:
echo 1. For command line: python run_wan_t4.py --prompt "Your prompt here"
echo 2. For web interface: python gradio_interface.py
echo.
echo Press any key to exit...
pause >nul
