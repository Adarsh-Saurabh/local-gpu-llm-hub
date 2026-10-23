"""
Model Optimizer & Smart Auto-Tuning Engine
Reads GGUF metadata (layers, architecture, context) and calculates the exact
optimal GPU layer offload and KV cache quantization based on physical hardware VRAM.
Prevents Windows Shared GPU Memory spillover while maximizing tokens/sec.
"""

import os
import struct
from typing import Dict, Any, Optional

def parse_gguf_metadata(filepath: str) -> Dict[str, Any]:
    """
    Fast, zero-dependency binary reader for GGUF model headers.
    Extracts architecture, layer count (block_count), native context length, and KV heads.
    """
    if not os.path.exists(filepath):
        return {}
        
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return {}
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]
            
            meta = {
                "file_size_gb": round(os.path.getsize(filepath) / (1024**3), 2),
                "architecture": "llama",
                "block_count": 0,
                "context_length": 32768,
                "embedding_length": 4096,
                "head_count": 32,
                "head_count_kv": 8
            }
            
            def read_str():
                slen = struct.unpack('<Q', f.read(8))[0]
                return f.read(slen).decode('utf-8', errors='replace')
                
            def skip_val(vtype):
                if vtype in (0, 1, 7): f.seek(1, os.SEEK_CUR)
                elif vtype in (2, 3): f.seek(2, os.SEEK_CUR)
                elif vtype in (4, 5, 6): f.seek(4, os.SEEK_CUR)
                elif vtype in (10, 11, 12): f.seek(8, os.SEEK_CUR)
                elif vtype == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.seek(slen, os.SEEK_CUR)
                elif vtype == 9: # array
                    atype = struct.unpack('<I', f.read(4))[0]
                    alen = struct.unpack('<Q', f.read(8))[0]
                    if atype in (0, 1, 7): f.seek(alen * 1, os.SEEK_CUR)
                    elif atype in (2, 3): f.seek(alen * 2, os.SEEK_CUR)
                    elif atype in (4, 5, 6): f.seek(alen * 4, os.SEEK_CUR)
                    elif atype in (10, 11, 12): f.seek(alen * 8, os.SEEK_CUR)
                    elif atype == 8:
                        for _ in range(alen):
                            slen = struct.unpack('<Q', f.read(8))[0]
                            f.seek(slen, os.SEEK_CUR)

            def read_val(vtype):
                if vtype == 0: return struct.unpack('<B', f.read(1))[0]
                elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
                elif vtype == 8: return read_str()
                elif vtype in (10, 11): return struct.unpack('<Q', f.read(8))[0]
                else:
                    skip_val(vtype)
                    return None
                
            for _ in range(kv_count):
                key = read_str()
                vtype = struct.unpack('<I', f.read(4))[0]
                if key == 'general.architecture':
                    meta['architecture'] = read_val(vtype)
                elif key.endswith('.block_count'):
                    meta['block_count'] = read_val(vtype)
                elif key.endswith('.context_length'):
                    meta['context_length'] = read_val(vtype)
                elif key.endswith('.embedding_length'):
                    meta['embedding_length'] = read_val(vtype)
                elif key.endswith('.attention.head_count_kv'):
                    meta['head_count_kv'] = read_val(vtype)
                elif key.endswith('.attention.head_count'):
                    meta['head_count'] = read_val(vtype)
                else:
                    skip_val(vtype)
            return meta
    except Exception as e:
        print(f"[ModelOptimizer] Note parsing GGUF: {e}")
        return {}

def calculate_optimal_settings(
    filepath: str,
    total_vram_mb: int = 16384,
    target_ctx: int = 32768,
    preferred_kv: str = "auto"
) -> Dict[str, Any]:
    """
    Computes optimal GPU offload layers and KV quantization for a given model
    to prevent shared GPU memory thrashing on Windows.
    """
    file_size_gb = round(os.path.getsize(filepath) / (1024**3), 2) if os.path.exists(filepath) else 8.0
    meta = parse_gguf_metadata(filepath)
    
    # Safe defaults if metadata parsing didn't find specific keys
    if not meta or not meta.get("block_count"):
        # Heuristic estimation based on parameter sizes
        est_layers = 32 if file_size_gb < 7 else (40 if file_size_gb < 13 else 64)
        meta = {
            "file_size_gb": file_size_gb,
            "architecture": "llama",
            "block_count": est_layers,
            "context_length": 32768,
            "embedding_length": 4096,
            "head_count": 32,
            "head_count_kv": 8
        }
    else:
        meta["file_size_gb"] = file_size_gb

    file_size_mb = meta["file_size_gb"] * 1024
    total_layers = meta["block_count"] or 32
    head_dim = (meta["embedding_length"] // (meta["head_count"] or 32)) or 128
    n_head_kv = meta["head_count_kv"] or 8

    # Windows OS / DWM / Display overhead reservation: ~1000 MB
    # CUDA runtime & GGML scratch memory buffer reservation: ~600 MB
    base_reserve_mb = 1600

    # Auto KV cache precision
    if preferred_kv == "auto" or not preferred_kv or preferred_kv == "default":
        if target_ctx >= 24576 or meta["file_size_gb"] >= 10.0:
            kv_quant = "q4_0"
        elif target_ctx >= 16384:
            kv_quant = "q8_0"
        else:
            kv_quant = "f16"
    else:
        kv_quant = preferred_kv

    # Calculate exact KV cache footprint in MB
    bytes_per_elem = 0.56 if kv_quant == "q4_0" else (1.06 if kv_quant in ("q8_0", "8bit") else 2.0)
    kv_size_bytes = 2 * total_layers * n_head_kv * head_dim * target_ctx * bytes_per_elem
    kv_size_mb = int(kv_size_bytes / (1024 * 1024))

    usable_vram_for_weights = total_vram_mb - base_reserve_mb - kv_size_mb

    if usable_vram_for_weights <= 0:
        return {
            "gpu_layers": 0,
            "kv_quant": "q4_0",
            "context_window": min(target_ctx, 8192),
            "fits_fully_in_gpu": False,
            "layers_on_gpu": 0,
            "layers_on_cpu": total_layers,
            "total_layers": total_layers,
            "kv_cache_mb": kv_size_mb,
            "estimated_vram_used_mb": int(base_reserve_mb + kv_size_mb),
            "free_headroom_mb": 0,
            "summary_text": "CPU Execution (KV Cache exceeds GPU VRAM)",
            "meta": meta
        }

    weight_per_layer = file_size_mb / total_layers

    if file_size_mb <= usable_vram_for_weights:
        # Full offload fits safely
        est_used = int(base_reserve_mb + kv_size_mb + file_size_mb)
        headroom = int(total_vram_mb - est_used)
        return {
            "gpu_layers": -1,
            "kv_quant": kv_quant,
            "context_window": target_ctx,
            "fits_fully_in_gpu": True,
            "layers_on_gpu": total_layers,
            "layers_on_cpu": 0,
            "total_layers": total_layers,
            "kv_cache_mb": kv_size_mb,
            "estimated_vram_used_mb": est_used,
            "free_headroom_mb": headroom,
            "summary_text": f"100% GPU Offload (All {total_layers} layers in VRAM, ~{headroom}MB headroom)",
            "meta": meta
        }
    else:
        # Calculate safe partial offload layers
        safe_gpu_layers = int(usable_vram_for_weights / weight_per_layer)
        # 1-layer safety margin against driver fragmentation
        safe_gpu_layers = max(0, min(total_layers - 1, safe_gpu_layers - 1))
        cpu_layers = total_layers - safe_gpu_layers
        est_used = int(base_reserve_mb + kv_size_mb + (safe_gpu_layers * weight_per_layer))
        headroom = int(total_vram_mb - est_used)

        return {
            "gpu_layers": safe_gpu_layers,
            "kv_quant": kv_quant,
            "context_window": target_ctx,
            "fits_fully_in_gpu": False,
            "layers_on_gpu": safe_gpu_layers,
            "layers_on_cpu": cpu_layers,
            "total_layers": total_layers,
            "kv_cache_mb": kv_size_mb,
            "estimated_vram_used_mb": est_used,
            "free_headroom_mb": headroom,
            "summary_text": f"Optimal Hybrid Split: {safe_gpu_layers}/{total_layers} layers on GPU, {cpu_layers} on RAM (~{headroom}MB VRAM buffer)",
            "meta": meta
        }
