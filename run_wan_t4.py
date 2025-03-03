import os
import sys
import argparse
import torch
import gc
import logging
from pathlib import Path
import subprocess
import time
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WAN21Runner:
    def __init__(self, args):
        self.args = args
        self.model_dir = Path(args.model_dir)
        self.repo_dir = Path(args.repo_dir)
        self.task = args.task
        self.prompt = args.prompt
        self.size = args.size
        self.output_dir = Path(args.output_dir)
        self.low_vram = args.low_vram
        self.quantize = args.quantize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_windows = platform.system() == "Windows"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check CUDA availability and GPU info
        self.check_gpu()
        
        # Clone the repository if needed
        self.setup_repository()
        
        # Download the model if needed
        self.download_model()

    def check_gpu(self):
        """Check GPU information and memory"""
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Running on CPU will be extremely slow.")
            return
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB VRAM")
        
        # Check if it's a T4 GPU
        if "T4" in gpu_name:
            logger.info("Detected T4 GPU. Applying T4-specific optimizations.")
            if not self.low_vram:
                logger.info("Automatically enabling low VRAM mode for T4 GPU")
                self.low_vram = True
        
        # Log current GPU memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # Convert to GB
        logger.info(f"Current GPU memory usage: Allocated {memory_allocated:.2f} GB, Reserved {memory_reserved:.2f} GB")

    def setup_repository(self):
        """Clone the WAN 2.1 repository if it doesn't exist"""
        if not self.repo_dir.exists():
            logger.info(f"Cloning WAN 2.1 repository to {self.repo_dir}")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/Wan-Video/Wan2.1.git", str(self.repo_dir)],
                    check=True
                )
                logger.info("Repository cloned successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone repository: {e}")
                sys.exit(1)
        else:
            logger.info(f"Repository directory {self.repo_dir} already exists")

    def download_model(self):
        """Download the model if it doesn't exist"""
        if not self.model_dir.exists():
            logger.info(f"Model directory {self.model_dir} doesn't exist. Creating it...")
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {self.task} model to {self.model_dir}")
            
            if self.task == "t2v-1.3B":
                model_id = "Wan-AI/Wan2.1-T2V-1.3B"
            elif self.task == "t2v-14B":
                model_id = "Wan-AI/Wan2.1-T2V-14B"
            elif self.task == "i2v-14B-480P":
                model_id = "Wan-AI/Wan2.1-I2V-14B-480P"
            elif self.task == "i2v-14B-720P":
                model_id = "Wan-AI/Wan2.1-I2V-14B-720P"
            else:
                logger.error(f"Unknown task: {self.task}")
                sys.exit(1)
            
            try:
                subprocess.run(
                    ["pip", "install", "huggingface_hub[cli]"],
                    check=True
                )
                subprocess.run(
                    ["huggingface-cli", "download", model_id, "--local-dir", str(self.model_dir)],
                    check=True
                )
                logger.info(f"Model {model_id} downloaded successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download model: {e}")
                sys.exit(1)
        else:
            logger.info(f"Model directory {self.model_dir} already exists")

    def optimize_memory(self):
        """Perform memory optimization before running the model"""
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory efficient attention
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        if self.low_vram:
            # Enable gradient checkpointing for more memory efficiency (at the cost of speed)
            os.environ["DIFFUSERS_FORCE_GRADIENT_CHECKPOINTING"] = "true"
            
            # Force attention implementation to be memory efficient
            os.environ["FORCE_MEM_EFFICIENT_ATTN"] = "true"

    def get_generation_command(self):
        """Construct the command to run the model"""
        cmd = [
            sys.executable,  # Use the current Python interpreter  
            os.path.join(str(self.repo_dir), "generate.py"),
            f"--task", self.task,
            f"--size", self.size,
            f"--ckpt_dir", str(self.model_dir),
            f"--prompt", self.prompt,
            f"--outdir", str(self.output_dir)
        ]
        
        # Add T4-specific optimizations
        if self.low_vram:
            cmd.extend([
                "--offload_model", "True",
                "--t5_cpu"
            ])
        
        # For T2V-1.3B model, add recommended settings
        if self.task == "t2v-1.3B":
            cmd.extend([
                "--sample_shift", "8",
                "--sample_guide_scale", "6"
            ])
            
        # Apply quantization if requested
        if self.quantize:
            cmd.extend([
                "--dtype", "fp8_e4m3fn",
                "--quant_attn", "True"
            ])
            
        return cmd

    def run(self):
        """Run the WAN 2.1 model with optimizations"""
        logger.info("Starting WAN 2.1 generation with memory optimizations")
        
        # Apply memory optimizations
        self.optimize_memory()
        
        # Get the command to run
        cmd = self.get_generation_command()
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Change working directory to repo dir
        original_dir = os.getcwd()
        os.chdir(str(self.repo_dir))
        
        try:
            # Run the generation
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    logger.info(line)
            
            process.stdout.close()
            return_code = process.wait()
            end_time = time.time()
            
            if return_code != 0:
                stderr = process.stderr.read()
                logger.error(f"Generation failed with return code {return_code}: {stderr}")
                return False
            
            logger.info(f"Generation completed successfully in {end_time - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return False
        finally:
            # Change back to original directory
            os.chdir(original_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Run WAN 2.1 on T4 GPU with optimizations")
    
    parser.add_argument("--repo_dir", type=str, default="./Wan2.1",
                        help="Directory to clone/store the WAN 2.1 repository")
    
    parser.add_argument("--task", type=str, default="t2v-1.3B", 
                        choices=["t2v-1.3B", "t2v-14B", "i2v-14B-480P", "i2v-14B-720P"],
                        help="Task to run. For T4 GPU, t2v-1.3B is recommended")
    
    parser.add_argument("--model_dir", type=str, default="./Wan2.1-model",
                        help="Directory to store/load the model")
    
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for generation")
    
    parser.add_argument("--size", type=str, default="832*480",
                        help="Size of generated video. For T4, use 832*480")
    
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save generated videos")
    
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low VRAM optimizations (automatically enabled for T4)")
    
    parser.add_argument("--quantize", action="store_true",
                        help="Enable 8-bit quantization for further memory reduction")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    runner = WAN21Runner(args)
    success = runner.run()
    
    if success:
        logger.info(f"Video generated successfully. Check {args.output_dir} for results")
    else:
        logger.error("Failed to generate video")
