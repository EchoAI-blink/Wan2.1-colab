# WAN 2.1 on T4 GPU

This repository contains scripts to run Alibaba Cloud's WAN 2.1 video generation model on a T4 GPU with optimized memory usage and performance. The T4 GPU has 16GB of VRAM, which presents challenges when running large video generation models like WAN 2.1.

## Features

- Memory-optimized implementation for T4 GPUs (16GB VRAM)
- Support for both Text-to-Video and Image-to-Video generation
- Command-line interface for scripting
- Gradio web interface for easy interaction
- Automatic model downloads from Hugging Face
- Cross-platform support for Windows and Linux
- Memory optimization techniques:
  - Model offloading
  - T5 encoder CPU offloading
  - 8-bit quantization options
  - Efficient attention mechanisms
  
## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/wan-t4.git
cd wan-t4
```

2. Install the required dependencies:

### Windows Setup
```bash
# Option 1: Using the batch script
setup_environment.bat

# Option 2: Using the Python setup script
python setup.py
```

### Linux Setup
```bash
# Option 1: Using the shell script (make it executable first)
chmod +x setup.sh
./setup.sh

# Option 2: Using the Python setup script
python setup.py
```

## Running via Command Line

The `run_wan_t4.py` script provides a command-line interface for running WAN 2.1 on a T4 GPU:

```bash
python run_wan_t4.py --prompt "A beautiful sunset over a calm ocean with sailing boats" --task t2v-1.3B --size 832*480
```

### Command Line Options

- `--prompt` (required): Text prompt for video generation
- `--task`: Model to use (`t2v-1.3B`, `t2v-14B`, `i2v-14B-480P`, `i2v-14B-720P`). Default: `t2v-1.3B` (recommended for T4)
- `--size`: Output video resolution. Default: `832*480`
- `--output_dir`: Directory to save generated videos. Default: `./output`
- `--low_vram`: Enable low VRAM optimizations (automatically enabled for T4)
- `--quantize`: Enable 8-bit quantization for further memory reduction
- `--model_dir`: Directory to store/load the model. Default: `./Wan2.1-model`
- `--repo_dir`: Directory for the WAN 2.1 repository. Default: `./Wan2.1`

## Running via Gradio Web Interface

For a more user-friendly experience, you can use the Gradio web interface:

```bash
python gradio_interface.py
```

This will start a local web server with a simple interface where you can:
1. Select the model type
2. Enter your text prompt
3. Choose the video resolution
4. Generate videos with a single click

## Monitoring GPU Usage

You can monitor the GPU, CPU, and RAM usage during video generation:

```bash
# Monitor system resources without running a command
python monitor_gpu.py

# Monitor resources while running WAN 2.1
python monitor_gpu.py --command "python run_wan_t4.py --prompt 'Your prompt here'"
```

## Optimization Techniques

The implementation uses several techniques to optimize for T4 GPUs:

1. **Model Offloading**: Parts of the model are offloaded to CPU when not in use
2. **T5 Text Encoder on CPU**: The T5 text encoder runs on CPU to save GPU memory
3. **Memory-Efficient Attention**: Uses optimized attention mechanisms to reduce memory usage
4. **Gradient Checkpointing**: Trades computation for memory by recomputing intermediate activations
5. **Quantization**: Optional 8-bit quantization for further memory reduction
6. **CUDA Memory Optimization**: Sets appropriate CUDA memory allocation parameters

## Recommended Settings for T4 GPU

- Use the `t2v-1.3B` model (1.3 billion parameters) which requires less memory
- Generate videos at 480p resolution (`832*480`)
- Enable quantization for larger models
- Keep videos short (around 5 seconds)

## Platform-Specific Notes

### Windows
- CUDA optimization libraries like Flash Attention and Sage Attention may be difficult to install
- The setup script will install the basic dependencies without these optimizations
- For best performance, consider using Linux

### Linux
- Allows installation of additional CUDA optimizations (Flash Attention, Sage Attention)
- Generally provides better performance for deep learning tasks
- Recommended for production use

## Troubleshooting

If you encounter "Out of Memory" errors:
1. Use a smaller model (t2v-1.3B)
2. Reduce the video resolution to 480p
3. Enable quantization with the `--quantize` flag
4. Close other GPU-intensive applications
5. Restart the Python kernel to clear GPU memory

## License

This project follows the same license as the original WAN 2.1 model. The code in this repository is provided for research and educational purposes only.
