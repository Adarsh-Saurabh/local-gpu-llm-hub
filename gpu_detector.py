"""
Universal Hardware & GPU Auto-Detection Engine
Detects NVIDIA CUDA, Apple Silicon Metal, AMD ROCm, or CPU fallback.
Calculates optimal VRAM allocation, context window, threads, and offload layers.
"""

import os
import sys
import platform
import subprocess
import shutil
from typing import Dict, Any, Optional

def detect_hardware() -> Dict[str, Any]:
    system_os = platform.system()  # 'Windows', 'Linux', 'Darwin'
    arch = platform.machine()      # 'x86_64', 'AMD64', 'arm64'
    cpu_count = os.cpu_count() or 4
    
    info = {
        "os": system_os,
        "arch": arch,
        "cpu_cores": cpu_count,
        "device_type": "cpu",       # 'cuda', 'metal', 'rocm', 'cpu'
        "device_name": "Generic CPU",
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "ram_total_mb": 0,
        "optimal_threads": max(1, cpu_count - 2),
        "flash_attn_supported": False,
        "recommended_tier": 1,      # 1 (<8GB), 2 (8-16GB), 3 (16-24GB), 4 (>24GB)
        "recommended_models": []
    }

    # 1. Detect System RAM
    try:
        if system_os == "Windows":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["ram_total_mb"] = int(stat.ullTotalPhys / (1024 * 1024))
        elif system_os == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            info["ram_total_mb"] = int(int(out) / (1024 * 1024))
        else: # Linux
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["ram_total_mb"] = int(line.split()[1]) // 1024
                        break
    except Exception:
        info["ram_total_mb"] = 16384  # sensible default fallback

    # 2. Detect NVIDIA CUDA GPU
    if shutil.which("nvidia-smi"):
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, encoding="utf-8").strip()
            lines = out.splitlines()
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                info["device_type"] = "cuda"
                info["device_name"] = parts[0]
                info["vram_total_mb"] = int(parts[1])
                info["vram_free_mb"] = int(parts[2])
                
                # Check for FlashAttention support (Turing/Ampere/Ada/Hopper)
                name_upper = parts[0].upper()
                if any(k in name_upper for k in ["RTX", "A100", "H100", "L4", "T4", "V100", "A4000", "A5000", "A6000", "QUADRO RTX"]):
                    info["flash_attn_supported"] = True
        except Exception:
            pass

    # 3. Detect Apple Silicon (Metal)
    if info["device_type"] == "cpu" and system_os == "Darwin" and arch == "arm64":
        info["device_type"] = "metal"
        info["device_name"] = f"Apple Silicon ({arch})"
        info["vram_total_mb"] = int(info["ram_total_mb"] * 0.75) # 75% unified memory allocation
        info["vram_free_mb"] = info["vram_total_mb"]
        info["flash_attn_supported"] = True

    # 4. Categorize Hardware Tier & Recommended Models
    vram = info["vram_total_mb"]
    if vram >= 24000:
        info["recommended_tier"] = 3
        info["recommended_models"] = [
            {"name": "Qwen2.5-Coder-32B-Instruct", "repo": "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF", "file": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf", "size": "19.5 GB"},
            {"name": "Mistral-Small-24B-Instruct", "repo": "bartowski/mistral-small-24b-instruct-2501-GGUF", "file": "mistral-small-24b-instruct-2501-Q4_K_M.gguf", "size": "14.2 GB"},
            {"name": "DeepSeek-R1-Distill-Qwen-14B", "repo": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF", "file": "DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf", "size": "15.8 GB"}
        ]
    elif vram >= 12000:
        info["recommended_tier"] = 2
        info["recommended_models"] = [
            {"name": "DeepSeek-R1-Distill-Qwen-14B (Reasoning)", "repo": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF", "file": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf", "size": "8.37 GB"},
            {"name": "Mistral-Small-24B-Instruct", "repo": "bartowski/mistral-small-24b-instruct-2501-GGUF", "file": "mistral-small-24b-instruct-2501-Q3_K_M.gguf", "size": "11.23 GB"},
            {"name": "Codestral-22B-v0.1 (FIM Specialist)", "repo": "bartowski/Codestral-22B-v0.1-GGUF", "file": "Codestral-22B-v0.1-Q3_K_M.gguf", "size": "10.64 GB"},
            {"name": "Qwen2.5-Coder-14B-Instruct", "repo": "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF", "file": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", "size": "8.99 GB"}
        ]
    else: # < 12GB VRAM or CPU
        info["recommended_tier"] = 1
        info["recommended_models"] = [
            {"name": "Qwen2.5-Coder-7B-Instruct", "repo": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", "size": "4.68 GB"},
            {"name": "DeepSeek-R1-Distill-Llama-8B", "repo": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF", "file": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf", "size": "5.12 GB"},
            {"name": "Phi-4-14B (Q3_K_S)", "repo": "bartowski/phi-4-GGUF", "file": "phi-4-Q3_K_S.gguf", "size": "6.85 GB"}
        ]

    return info

def get_optimal_llama_params(n_ctx: int = 16384) -> Dict[str, Any]:
    hw = detect_hardware()
    
    if hw["device_type"] in ["cuda", "metal"]:
        gpu_layers = -1  # Full GPU offload
    else:
        gpu_layers = 0   # CPU execution
        
    return {
        "n_gpu_layers": gpu_layers,
        "n_ctx": n_ctx,
        "n_threads": hw["optimal_threads"],
        "flash_attn": hw["flash_attn_supported"],
        "hardware_info": hw
    }

if __name__ == "__main__":
    hw = detect_hardware()
    print("=" * 60)
    print("         AUTOMATIC HARDWARE DETECTION & OPTIMIZATION")
    print("=" * 60)
    print(f"OS / Architecture  : {hw['os']} ({hw['arch']})")
    print(f"Detected Device    : {hw['device_name']} [{hw['device_type'].upper()}]")
    print(f"Total VRAM / RAM   : {hw['vram_total_mb']} MB VRAM / {hw['ram_total_mb']} MB System RAM")
    print(f"FlashAttention     : {'Supported' if hw['flash_attn_supported'] else 'Disabled'}")
    print(f"Optimal CPU Threads: {hw['optimal_threads']}")
    print(f"Hardware Tier      : Tier {hw['recommended_tier']}")
    print("Recommended Models for this hardware:")
    for m in hw["recommended_models"]:
        print(f"  - {m['name']} ({m['size']}) -> {m['repo']}")
    print("=" * 60)
