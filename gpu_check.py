import os
import sys
import subprocess


def run_command(command):
    """Run a command and return its output"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


# Print system information
print("\n===== SYSTEM INFORMATION =====")
print(f"Python version: {sys.version}")
print(f"Operating system: {os.name} - {sys.platform}")

# Check if NVIDIA drivers are installed
print("\n===== NVIDIA DRIVER CHECK =====")
if sys.platform == 'win32':
    nvidia_smi = run_command("where nvidia-smi")
    if nvidia_smi:
        driver_version = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        print(f"NVIDIA driver version: {driver_version}")

        # Get GPU information
        gpu_info = run_command(
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader")
        print(f"GPU information: {gpu_info}")
    else:
        print("NVIDIA drivers not found. Run 'where nvidia-smi' to verify installation.")
else:
    nvidia_smi = run_command("which nvidia-smi")
    if nvidia_smi:
        driver_version = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        print(f"NVIDIA driver version: {driver_version}")

        # Get GPU information
        gpu_info = run_command(
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader")
        print(f"GPU information: {gpu_info}")
    else:
        print("NVIDIA drivers not found. Run 'which nvidia-smi' to verify installation.")

# Check PyTorch and CUDA
print("\n===== PYTORCH & CUDA CHECK =====")
try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch CUDA is not available.")

        # Check for common issues
        if hasattr(torch, '_C'):
            if hasattr(torch._C, '_cuda_isDriverSufficient'):
                print(f"CUDA driver sufficient: {torch._C._cuda_isDriverSufficient()}")
            else:
                print("Cannot check CUDA driver status directly.")
        else:
            print("Cannot check CUDA driver status.")

except ImportError:
    print("PyTorch is not installed.")


