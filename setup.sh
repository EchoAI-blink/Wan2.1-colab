#!/bin/bash

# Make script executable with chmod +x setup.sh
echo "Setting up environment for WAN 2.1 on Linux..."

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python_version_major=$(echo $python_version | cut -d. -f1)
python_version_minor=$(echo $python_version | cut -d. -f2)

if [[ $python_version_major -lt 3 ]] || [[ $python_version_major -eq 3 && $python_version_minor -lt 8 ]]; then
    echo "Python 3.8+ is required. Found Python $python_version"
    exit 1
fi

echo "Found Python $python_version"

# Check for CUDA installation
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available. GPU details:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "CUDA is not installed or NVIDIA driver is not properly configured."
    echo "Continuing anyway, but performance may be affected..."
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Check if CUDA is available within the Python environment
cuda_available=$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)

# Install PyTorch with appropriate CUDA version
if [[ "$cuda_available" == "True" ]]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU version)..."
    pip install torch==2.4.0 torchvision torchaudio
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install CUDA-specific optimizations if CUDA is available
if [[ "$cuda_available" == "True" ]]; then
    echo "Installing CUDA optimizations..."
    pip install flash-attn==2.7.2.post1
    pip install sageattention==1.0.6
fi

# Create necessary directories
mkdir -p output models

echo "Environment setup complete!"
echo
echo "To use the WAN 2.1 model:"
echo "1. First activate the environment with: source venv/bin/activate"
echo "2. For command line: python run_wan_t4.py --prompt \"Your prompt here\""
echo "3. For web interface: python gradio_interface.py"
echo "4. For GPU monitoring: python monitor_gpu.py --command \"python run_wan_t4.py --prompt 'Your prompt here'\""
echo
echo "Press any key to exit..."
read -n 1
