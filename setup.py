#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import argparse

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"Python version: {platform.python_version()} ✓")

def check_cuda():
    """Check for CUDA installation."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            cuda_version = torch.version.cuda
            print(f"CUDA is available. Version: {cuda_version} ✓")
            print(f"Found {device_count} CUDA device(s): {', '.join(device_names)}")
            return True
        else:
            print("Warning: CUDA is not available. GPU acceleration will not be used.")
            return False
    except ImportError:
        print("Warning: PyTorch is not installed yet. CUDA availability will be checked after installation.")
        return False

def create_virtual_env(venv_path):
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(venv_path):
        print(f"Creating virtual environment at {venv_path}...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            print("Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            sys.exit(1)
    else:
        print(f"Virtual environment already exists at {venv_path}.")

def run_command(cmd, desc=None):
    """Run a command with a description."""
    if desc:
        print(f"{desc}...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def get_pip_command(venv_path):
    """Get the pip command based on the platform and virtual environment."""
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip")
    else:  # Linux/MacOS
        pip_path = os.path.join(venv_path, "bin", "pip")
    return pip_path

def get_python_command(venv_path):
    """Get the python command based on the platform and virtual environment."""
    if platform.system() == "Windows":
        python_path = os.path.join(venv_path, "Scripts", "python")
    else:  # Linux/MacOS
        python_path = os.path.join(venv_path, "bin", "python")
    return python_path

def setup_environment(use_venv=True, venv_path="venv"):
    """Set up the environment for WAN 2.1."""
    print("Setting up environment for WAN 2.1...")
    
    # Check Python version
    check_python_version()
    
    # Create directories
    for directory in ["output", "models"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Create and activate virtual environment if requested
    if use_venv:
        create_virtual_env(venv_path)
        pip_cmd = get_pip_command(venv_path)
        python_cmd = get_python_command(venv_path)
    else:
        pip_cmd = [sys.executable, "-m", "pip"]
        python_cmd = sys.executable
    
    # Upgrade pip
    run_command([pip_cmd, "install", "--upgrade", "pip"], "Upgrading pip")
    
    # Install PyTorch with appropriate CUDA support
    is_cuda_available = check_cuda()
    
    if platform.system() == "Windows":
        # Windows PyTorch installation
        cuda_suffix = "cu121" if is_cuda_available else "cpu"
        run_command(
            [pip_cmd, "install", "torch==2.4.0", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/{cuda_suffix}"],
            "Installing PyTorch for Windows"
        )
    else:
        # Linux PyTorch installation
        if is_cuda_available:
            run_command(
                [pip_cmd, "install", "torch==2.4.0", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"],
                "Installing PyTorch for Linux with CUDA"
            )
        else:
            run_command(
                [pip_cmd, "install", "torch==2.4.0", "torchvision", "torchaudio"],
                "Installing PyTorch for Linux (CPU only)"
            )
    
    # Install other requirements
    run_command([pip_cmd, "install", "-r", "requirements.txt"], "Installing requirements")
    
    # Optional: Install platform-specific optimizations
    if is_cuda_available:
        if platform.system() == "Linux":
            print("Installing optional CUDA optimizations for Linux...")
            # These can be challenging to install on Windows, so only attempt on Linux
            run_command([pip_cmd, "install", "flash-attn==2.7.2.post1"], "Installing Flash Attention")
            # Only try to install if on Linux
            run_command([pip_cmd, "install", "sageattention==1.0.6"], "Installing Sage Attention")
    
    print("\nEnvironment setup complete!")
    print("\nTo use the WAN 2.1 model:")
    
    if use_venv:
        if platform.system() == "Windows":
            python_cmd_str = f"{venv_path}\\Scripts\\python"
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:
            python_cmd_str = f"{venv_path}/bin/python"
            activate_cmd = f"source {venv_path}/bin/activate"
        
        print(f"First activate the virtual environment:")
        print(f"  {activate_cmd}")
        print(f"Then run one of the following:")
    else:
        python_cmd_str = "python"
        print(f"Run one of the following:")
    
    print(f"1. For command line: {python_cmd_str} run_wan_t4.py --prompt \"Your prompt here\"")
    print(f"2. For web interface: {python_cmd_str} gradio_interface.py")
    print(f"3. For GPU monitoring: {python_cmd_str} monitor_gpu.py --command \"{python_cmd_str} run_wan_t4.py --prompt 'Your prompt here'\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up environment for WAN 2.1")
    parser.add_argument("--no-venv", action="store_true", help="Don't create a virtual environment")
    parser.add_argument("--venv-path", default="venv", help="Path for the virtual environment")
    args = parser.parse_args()
    
    setup_environment(not args.no_venv, args.venv_path)
