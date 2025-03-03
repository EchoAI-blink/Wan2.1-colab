import torch
import time
import argparse
import subprocess
import sys
import os
import threading
import psutil
import platform
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

class GPUMonitor:
    def __init__(self, interval=1.0, plot=False, log_file=None):
        self.interval = interval
        self.plot = plot
        self.log_file = log_file
        self.stop_event = threading.Event()
        self.gpu_data = []
        self.cpu_data = []
        self.ram_data = []
        self.time_points = []
        self.start_time = None
        self.is_windows = platform.system() == "Windows"
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            print("CUDA is not available. Only monitoring CPU and RAM.")
        else:
            self.device_count = torch.cuda.device_count()
            self.device_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
            print(f"Found {self.device_count} CUDA device(s): {', '.join(self.device_names)}")
    
    def _get_gpu_memory(self):
        """Get GPU memory usage for all devices"""
        if not self.cuda_available:
            return []
        
        memory_info = []
        for i in range(self.device_count):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
            memory_info.append({
                "device": i,
                "name": self.device_names[i],
                "allocated_gb": allocated,
                "reserved_gb": reserved
            })
        return memory_info
    
    def _get_system_memory(self):
        """Get system CPU and RAM usage"""
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_percent = psutil.virtual_memory().percent
        return cpu_percent, ram_percent
    
    def _log_data(self, gpu_info, cpu_percent, ram_percent, elapsed):
        """Log data to file if log_file is specified"""
        if self.log_file:
            with open(self.log_file, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} [Elapsed: {elapsed:.2f}s] CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%")
                
                if self.cuda_available:
                    for gpu in gpu_info:
                        f.write(f", GPU {gpu['device']} ({gpu['name']}): Allocated: {gpu['allocated_gb']:.2f} GB, Reserved: {gpu['reserved_gb']:.2f} GB")
                f.write("\n")
    
    def _print_data(self, gpu_info, cpu_percent, ram_percent, elapsed):
        """Print current resource usage"""
        print(f"\r[Elapsed: {elapsed:.2f}s] CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%", end="")
        
        if self.cuda_available:
            for gpu in gpu_info:
                print(f", GPU {gpu['device']}: {gpu['allocated_gb']:.2f}/{gpu['reserved_gb']:.2f} GB", end="")
        sys.stdout.flush()
    
    def monitor_thread(self):
        """Thread function to monitor resources"""
        if self.log_file:
            with open(self.log_file, "w") as f:
                f.write("Timestamp,Elapsed,CPU,RAM,GPU_Allocated,GPU_Reserved\n")
        
        self.start_time = time.time()
        while not self.stop_event.is_set():
            gpu_info = self._get_gpu_memory()
            cpu_percent, ram_percent = self._get_system_memory()
            
            elapsed = time.time() - self.start_time
            
            # Store data for plotting
            self.time_points.append(elapsed)
            self.cpu_data.append(cpu_percent)
            self.ram_data.append(ram_percent)
            
            if self.cuda_available:
                gpu_allocated = gpu_info[0]["allocated_gb"] if gpu_info else 0
                gpu_reserved = gpu_info[0]["reserved_gb"] if gpu_info else 0
                self.gpu_data.append(gpu_allocated)
            
            # Log and print data
            self._log_data(gpu_info, cpu_percent, ram_percent, elapsed)
            self._print_data(gpu_info, cpu_percent, ram_percent, elapsed)
            
            # Sleep for the interval
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring"""
        print("Starting resource monitoring...")
        self.monitor_thread_obj = threading.Thread(target=self.monitor_thread)
        self.monitor_thread_obj.daemon = True
        self.monitor_thread_obj.start()
    
    def stop(self):
        """Stop monitoring and plot if requested"""
        self.stop_event.set()
        if hasattr(self, 'monitor_thread_obj'):
            self.monitor_thread_obj.join(timeout=2.0)
        print("\nMonitoring stopped.")
        
        if self.plot and self.time_points:
            self._generate_plot()
    
    def _generate_plot(self):
        """Generate a plot of resource usage over time"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.time_points, self.cpu_data, label='CPU (%)', color='blue')
        plt.plot(self.time_points, self.ram_data, label='RAM (%)', color='green')
        
        if self.cuda_available and self.gpu_data:
            # Convert to percentage of total VRAM
            # For T4, total is around 16GB
            total_vram = 16.0  # Assuming T4 with 16GB
            gpu_percent = [100 * g / total_vram for g in self.gpu_data]
            plt.plot(self.time_points, gpu_percent, label='GPU (%)', color='red')
        
        plt.title('Resource Usage During WAN 2.1 Execution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig('resource_usage.png')
        print(f"Resource usage plot saved to resource_usage.png")

def run_with_monitoring(command, interval=1.0, plot=True, log_file="resource_monitor.log"):
    """Run a command while monitoring system resources"""
    monitor = GPUMonitor(interval=interval, plot=plot, log_file=log_file)
    
    try:
        monitor.start()
        print(f"Running command: {command}")
        
        # Determine shell parameter based on platform
        use_shell = platform.system() == "Windows"
        
        process = subprocess.Popen(
            command,
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream the output of the process
        for line in iter(process.stdout.readline, ''):
            print("\n" + line.strip())
        
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            print(f"\nCommand failed with return code {return_code}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    finally:
        monitor.stop()

def main():
    parser = argparse.ArgumentParser(description="Monitor GPU/CPU/RAM usage while running a command")
    parser.add_argument("--command", type=str, help="Command to run")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--log-file", type=str, default="resource_monitor.log", help="Log file path")
    
    args = parser.parse_args()
    
    if args.command:
        run_with_monitoring(
            args.command,
            interval=args.interval,
            plot=not args.no_plot,
            log_file=args.log_file
        )
    else:
        monitor = GPUMonitor(interval=args.interval, plot=not args.no_plot, log_file=args.log_file)
        try:
            monitor.start()
            print("Press Ctrl+C to stop monitoring")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            monitor.stop()

if __name__ == "__main__":
    main()
