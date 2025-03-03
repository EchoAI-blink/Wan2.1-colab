import os
import sys
import gradio as gr
import torch
import subprocess
import logging
import time
import platform
from pathlib import Path
import tempfile
import shutil
import threading
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WAN21Interface:
    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.repo_dir = script_dir / "Wan2.1"
        self.model_base_dir = script_dir / "models"
        self.output_dir = script_dir / "output"
        
        # Create directories if they don't exist
        os.makedirs(self.model_base_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check CUDA availability and GPU info
        self.check_gpu()
        
        # Setup the repository if needed
        self.setup_repository()
        
    def check_gpu(self):
        """Check GPU information and memory"""
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Running on CPU will be extremely slow.")
            return
        
        self.gpu_name = torch.cuda.get_device_name(0)
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        logger.info(f"Using GPU: {self.gpu_name} with {self.gpu_memory:.2f} GB VRAM")
        
        # Check if it's a T4 GPU
        self.is_t4 = "T4" in self.gpu_name
        if self.is_t4:
            logger.info("Detected T4 GPU. Will apply T4-specific optimizations.")
        
        # Log current GPU memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # Convert to GB
        logger.info(f"Current GPU memory usage: Allocated {memory_allocated:.2f} GB, Reserved {memory_reserved:.2f} GB")

    def setup_repository(self):
        """Clone the WAN 2.1 repository if it doesn't exist"""
        if not os.path.exists(self.repo_dir):
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

    def download_model(self, task):
        """Download the model if it doesn't exist"""
        model_dir = self.model_base_dir / f"Wan2.1-{task}"
        
        if not os.path.exists(model_dir):
            logger.info(f"Model directory {model_dir} doesn't exist. Creating it...")
            os.makedirs(model_dir, exist_ok=True)
            
            logger.info(f"Downloading {task} model to {model_dir}")
            
            if task == "t2v-1.3B":
                model_id = "Wan-AI/Wan2.1-T2V-1.3B"
            elif task == "t2v-14B":
                model_id = "Wan-AI/Wan2.1-T2V-14B"
            elif task == "i2v-14B-480P":
                model_id = "Wan-AI/Wan2.1-I2V-14B-480P"
            elif task == "i2v-14B-720P":
                model_id = "Wan-AI/Wan2.1-I2V-14B-720P"
            else:
                logger.error(f"Unknown task: {task}")
                return None
            
            try:
                subprocess.run(
                    ["pip", "install", "huggingface_hub[cli]"],
                    check=True
                )
                subprocess.run(
                    ["huggingface-cli", "download", model_id, "--local-dir", str(model_dir)],
                    check=True
                )
                logger.info(f"Model {model_id} downloaded successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download model: {e}")
                return None
        else:
            logger.info(f"Model directory {model_dir} already exists")
        
        return model_dir

    def get_generation_command(self, task, prompt, size, model_dir, output_path, low_vram, quantize):
        """Construct the command to run the model"""
        cmd = [
            sys.executable,  # Use the current Python interpreter 
            os.path.join(str(self.repo_dir), "generate.py"),
            f"--task", task,
            f"--size", size,
            f"--ckpt_dir", str(model_dir),
            f"--prompt", prompt,
            f"--outdir", os.path.dirname(output_path),
            f"--outfile", os.path.basename(output_path)
        ]
        
        # Add T4-specific optimizations
        if low_vram or self.is_t4:
            cmd.extend([
                "--offload_model", "True",
                "--t5_cpu"
            ])
        
        # For T2V-1.3B model, add recommended settings
        if task == "t2v-1.3B":
            cmd.extend([
                "--sample_shift", "8",
                "--sample_guide_scale", "6"
            ])
            
        # Apply quantization if requested
        if quantize:
            cmd.extend([
                "--dtype", "fp8_e4m3fn",
                "--quant_attn", "True"
            ])
            
        return cmd

    def generate_video(self, task, prompt, size, progress=gr.Progress()):
        """Generate video using WAN 2.1"""
        try:
            # Download model if needed
            model_dir = self.download_model(task)
            if not model_dir:
                return None, "Failed to download model"
            
            # Create a unique output file
            timestamp = int(time.time())
            output_filename = f"generated_{timestamp}.mp4"
            output_path = os.path.join(str(self.output_dir), output_filename)
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Set memory efficient attention
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            os.environ["DIFFUSERS_FORCE_GRADIENT_CHECKPOINTING"] = "true"
            os.environ["FORCE_MEM_EFFICIENT_ATTN"] = "true"
            
            # Get the command to run
            low_vram = True  # Always use low_vram optimizations for T4
            quantize = True  # Use quantization for T4
            cmd = self.get_generation_command(task, prompt, size, model_dir, output_path, low_vram, quantize)
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Change working directory to repo dir
            original_dir = os.getcwd()
            os.chdir(str(self.repo_dir))
            
            # Capture output for progress updates
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Process output and update progress
            progress(0, desc="Starting generation...")
            
            for i, line in enumerate(iter(process.stdout.readline, '')):
                line = line.strip()
                if line:
                    logger.info(line)
                    # Try to extract progress information
                    if "step" in line.lower() and "progress" in line.lower():
                        try:
                            progress_str = line.split("progress")[1].strip()
                            progress_val = float(progress_str.strip("%").strip(":").strip()) / 100
                            progress(progress_val, desc=f"Generating: {line}")
                        except:
                            progress((i % 50) / 50, desc=f"Processing: {line}")
                    else:
                        progress((i % 50) / 50, desc=f"Processing: {line}")
            
            process.stdout.close()
            return_code = process.wait()
            
            # Change back to original directory
            os.chdir(original_dir)
            
            if return_code != 0:
                stderr = process.stderr.read()
                logger.error(f"Generation failed with return code {return_code}: {stderr}")
                return None, f"Generation failed: {stderr}"
            
            logger.info(f"Generation completed successfully")
            
            # Check if the file exists
            if os.path.exists(output_path):
                return output_path, "Generation completed successfully!"
            else:
                return None, "Generation completed but output file not found"
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return None, f"Error: {str(e)}"

    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(title="WAN 2.1 Video Generator for T4 GPU") as interface:
            gr.Markdown("# WAN 2.1 Video Generator for T4 GPU")
            gr.Markdown(f"Running on: {self.gpu_name} with {self.gpu_memory:.2f} GB VRAM")
            
            with gr.Row():
                with gr.Column():
                    task = gr.Dropdown(
                        choices=["t2v-1.3B", "t2v-14B", "i2v-14B-480P", "i2v-14B-720P"],
                        value="t2v-1.3B",
                        label="Model Type",
                        info="For T4 GPU, t2v-1.3B is recommended"
                    )
                    
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter a descriptive prompt for your video",
                        lines=3
                    )
                    
                    size = gr.Dropdown(
                        choices=["832*480", "1280*720"],
                        value="832*480",
                        label="Video Size",
                        info="For T4 GPU, 832x480 is recommended"
                    )
                    
                    generate_btn = gr.Button("Generate Video", variant="primary")
                    
                    status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    output_video = gr.Video(label="Generated Video")
            
            generate_btn.click(
                fn=self.generate_video,
                inputs=[task, prompt, size],
                outputs=[output_video, status]
            )
            
            gr.Markdown("""
            ## Tips for Running on T4 GPU
            
            1. Choose the 1.3B model for better performance on T4
            2. Use 480p resolution to reduce memory usage
            3. Keep prompts clear and concise
            4. First generation will be slower as it downloads the model
            5. Generation typically takes several minutes per video
            """)
            
        return interface

def main():
    interface = WAN21Interface()
    app = interface.create_interface()
    app.launch(share=True)

if __name__ == "__main__":
    main()
