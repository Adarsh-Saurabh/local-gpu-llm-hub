import os
import sys
import glob
import time
import json
import gc
import uuid
import subprocess
import threading
import requests
import ctypes
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from huggingface_hub import HfApi, hf_hub_url
from gpu_detector import detect_hardware, get_optimal_llama_params
from model_optimizer import calculate_optimal_settings, parse_gguf_metadata

# Setup CUDA DLL directories for Windows Conda/Virtual environment
conda_prefix = sys.prefix
for dll_path in [
    os.path.join(conda_prefix, "bin"),
    os.path.join(conda_prefix, "Library", "bin"),
    os.path.join(conda_prefix, "Lib", "site-packages", "llama_cpp", "lib")
]:
    if os.path.exists(dll_path):
        try:
            os.add_dll_directory(dll_path)
        except Exception:
            pass

from llama_cpp import Llama

app = FastAPI(title="Universal Local GPU LLM Hub", version="4.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Hardware Profile
HARDWARE_PROFILE = detect_hardware()

# Global Real-time Download State
download_state = {
    "status": "idle", # idle, downloading, completed, error
    "repo_id": "",
    "filename": "",
    "downloaded_gb": 0.0,
    "total_gb": 0.0,
    "percent": 0.0,
    "speed_mb": 0.0,
    "eta": "",
    "message": ""
}

# Real-time Inference Telemetry State
inference_stats = {
    "tokens_per_second": 0.0,
    "last_tokens_generated": 0,
    "last_duration_seconds": 0.0,
    "is_generating": False
}

def get_gpu_telemetry() -> Dict[str, Any]:
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, encoding="utf-8").strip()
        parts = [p.strip() for p in output.split(",")]
        return {
            "memory_used_mb": int(parts[0]),
            "memory_total_mb": int(parts[1]),
            "temperature_c": int(parts[2]),
            "utilization_pct": int(parts[3])
        }
    except Exception:
        return {
            "memory_used_mb": 0,
            "memory_total_mb": HARDWARE_PROFILE.get("vram_total_mb", 16384),
            "temperature_c": 0,
            "utilization_pct": 0
        }

class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint), ("dwHighDateTime", ctypes.c_uint)]

class _MEMORYSTATUSEX(ctypes.Structure):
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

_prev_cpu_idle = 0
_prev_cpu_kernel = 0
_prev_cpu_user = 0

def get_system_telemetry() -> Dict[str, Any]:
    global _prev_cpu_idle, _prev_cpu_kernel, _prev_cpu_user
    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_mb = int(stat.ullTotalPhys / (1024 * 1024))
        avail_mb = int(stat.ullAvailPhys / (1024 * 1024))
        used_mb = total_mb - avail_mb
        ram_pct = int(stat.dwMemoryLoad)

        idle = _FILETIME()
        kernel = _FILETIME()
        user = _FILETIME()
        cpu_pct = 0
        if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            idle_int = (idle.dwHighDateTime << 32) + idle.dwLowDateTime
            kernel_int = (kernel.dwHighDateTime << 32) + kernel.dwLowDateTime
            user_int = (user.dwHighDateTime << 32) + user.dwLowDateTime
            if _prev_cpu_kernel > 0:
                idle_diff = idle_int - _prev_cpu_idle
                kernel_diff = kernel_int - _prev_cpu_kernel
                user_diff = user_int - _prev_cpu_user
                total = kernel_diff + user_diff
                if total > 0:
                    cpu_pct = int(((total - idle_diff) / total) * 100)
                    cpu_pct = max(0, min(100, cpu_pct))
            _prev_cpu_idle, _prev_cpu_kernel, _prev_cpu_user = idle_int, kernel_int, user_int

        return {
            "ram_used_mb": used_mb,
            "ram_total_mb": total_mb,
            "ram_pct": ram_pct,
            "cpu_pct": cpu_pct
        }
    except Exception:
        return {
            "ram_used_mb": 0,
            "ram_total_mb": 65216,
            "ram_pct": 0,
            "cpu_pct": 0
        }

class ModelManager:
    def __init__(self):
        self.active_llm: Optional[Llama] = None
        self.active_model_name: Optional[str] = None
        self.active_gpu_layers: int = -1
        self.active_ctx: int = 32768
        self.active_kv_quant: str = "f16"
        self.lock = threading.RLock()
        self.last_active_time: float = time.time()
        self.idle_timeout_seconds: int = 1800

    def list_models(self) -> List[Dict[str, Any]]:
        with self.lock:
            files = glob.glob(os.path.join(MODELS_DIR, "*.gguf"))
            result = []
            total_vram = HARDWARE_PROFILE.get("vram_total_mb", 16384)
            for f in files:
                size_gb = round(os.path.getsize(f) / (1024 ** 3), 2)
                name = os.path.basename(f)
                is_loaded = (name == self.active_model_name and self.active_llm is not None)
                opt = calculate_optimal_settings(
                    f,
                    total_vram_mb=total_vram,
                    target_ctx=self.active_ctx if is_loaded else 32768,
                    preferred_kv=self.active_kv_quant if is_loaded else "auto"
                )
                result.append({
                    "name": name,
                    "size_gb": f"{size_gb} GB",
                    "is_loaded": is_loaded,
                    "optimal_settings": opt
                })
            return result

    def resolve_model_name(self, requested_name: Optional[str]) -> Optional[str]:
        models = self.list_models()
        if not models:
            return None
        
        if not requested_name or requested_name.strip() in ["", "default", "*", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet", "claude-3-sonnet", "gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
            if self.active_model_name:
                return self.active_model_name
            return models[0]["name"]

        for m in models:
            if m["name"] == requested_name:
                return m["name"]

        clean_req = requested_name.lower().replace(".gguf", "")
        for m in models:
            if clean_req in m["name"].lower():
                return m["name"]

        return self.active_model_name or models[0]["name"]

    def _close_active_llm(self) -> None:
        """Safely release the active Llama instance from memory and CUDA pool."""
        if self.active_llm is not None:
            old_model = self.active_model_name
            print(f"\n[GPU Server] Freeing memory for '{old_model}'...")
            try:
                self.active_llm.close()
            except Exception as e:
                print(f"[GPU Server] Note during LLM close: {e}")
            self.active_llm = None
            self.active_model_name = None
            gc.collect()
            print("[GPU Server] VRAM freed successfully.")

    def unload(self) -> Dict[str, str]:
        with self.lock:
            if self.active_llm is not None:
                old_model = self.active_model_name
                self._close_active_llm()
                return {"status": "unloaded", "model": old_model or "none", "message": "GPU VRAM successfully cleared."}
            return {"status": "idle", "message": "No model was currently loaded in VRAM."}

    def _load_internal(self, model_name: str, n_ctx: int = 32768, n_gpu_layers: Union[int, str] = -1, kv_quant: str = "f16") -> None:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in ./models/")

        total_vram = HARDWARE_PROFILE.get("vram_total_mb", 16384)
        opt = calculate_optimal_settings(
            model_path,
            total_vram_mb=total_vram,
            target_ctx=n_ctx,
            preferred_kv=kv_quant
        )

        # Handle auto-tuning
        if n_gpu_layers in (-2, "auto", "-2"):
            n_gpu_layers = opt["gpu_layers"]
            if kv_quant in ("auto", "default", None):
                kv_quant = opt["kv_quant"]
            print(f"[GPU Server Auto-Tuner] Applied optimal configuration: {opt['summary_text']}")
        else:
            try:
                n_gpu_layers = int(n_gpu_layers)
            except Exception:
                n_gpu_layers = -1
            if n_gpu_layers == -1 and not opt["fits_fully_in_gpu"]:
                print(f"\n[GPU Server Warning] Model '{model_name}' ({opt['meta']['file_size_gb']} GB) exceeds safe VRAM limit!")
                print(f"[GPU Server Warning] Full GPU offload (-1) will spill into slow Windows Shared Memory (1-2 tok/s).")
                print(f"[GPU Server Warning] Safe recommended GPU layer offload is: {opt['gpu_layers']} layers.")

        if (self.active_model_name == model_name and 
            self.active_llm is not None and 
            self.active_gpu_layers == n_gpu_layers and 
            self.active_ctx == n_ctx and 
            self.active_kv_quant == kv_quant):
            self.last_active_time = time.time()
            return

        if self.active_llm is not None:
            print(f"\n[GPU Server] Dynamic Switch: Freeing '{self.active_model_name}'...")
            self._close_active_llm()

        flash_attn = HARDWARE_PROFILE.get("flash_attn_supported", True)
        threads = HARDWARE_PROFILE.get("optimal_threads", 8)

        kv_map = {
            "f16": None,
            "fp16": None,
            "q8_0": 8,  # GGML_TYPE_Q8_0
            "8bit": 8,
            "q4_0": 2,  # GGML_TYPE_Q4_0
            "4bit": 2
        }
        tk = kv_map.get(kv_quant.lower(), None)
        tv = tk

        print(f"\n[GPU Server] Loading '{model_name}'...")
        print(f"[GPU Server] GPU Offload Layers: {n_gpu_layers} (-1 = Full GPU)")
        print(f"[GPU Server] Context Window: {n_ctx} tokens | KV Cache Quant: {kv_quant.upper()} | Threads: {threads} | FlashAttn: {flash_attn}")
        
        self.active_llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=threads,
            type_k=tk,
            type_v=tv,
            flash_attn=flash_attn,
            verbose=False
        )
        self.active_model_name = model_name
        self.active_gpu_layers = n_gpu_layers
        self.active_ctx = n_ctx
        self.active_kv_quant = kv_quant
        self.last_active_time = time.time()
        print(f"[GPU Server] Successfully loaded '{model_name}' into memory!\n")

    def load(self, model_name: str, n_ctx: int = 32768, n_gpu_layers: int = -1, kv_quant: str = "f16") -> None:
        with self.lock:
            self._load_internal(model_name, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, kv_quant=kv_quant)

    def delete(self, model_name: str) -> Dict[str, str]:
        with self.lock:
            if self.active_model_name == model_name and self.active_llm is not None:
                print(f"[GPU Server] Unloading '{model_name}' before deletion...")
                self._close_active_llm()

            model_path = os.path.join(MODELS_DIR, model_name)
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"[GPU Server] Deleted model file: {model_path}")
                return {"status": "deleted", "message": f"Model '{model_name}' deleted successfully from disk."}
            else:
                raise HTTPException(status_code=404, detail="Model file not found on disk.")

    def get_llm(self, requested_model: Optional[str] = None, n_ctx: Optional[int] = None, n_gpu_layers: Optional[int] = None, kv_quant: Optional[str] = None) -> Llama:
        with self.lock:
            target_model = self.resolve_model_name(requested_model)
            if not target_model:
                raise HTTPException(status_code=404, detail="No models found on server. Please download a model first.")

            # If a model is ALREADY loaded
            if self.active_llm is not None:
                # If requested model matches what's currently active (or was omitted / "default")
                if target_model == self.active_model_name:
                    # Only reload if the caller explicitly requested a different setting
                    need_reload = False
                    if n_ctx is not None and n_ctx != self.active_ctx:
                        need_reload = True
                    if n_gpu_layers is not None and n_gpu_layers != self.active_gpu_layers:
                        need_reload = True
                    if kv_quant is not None and kv_quant != self.active_kv_quant:
                        need_reload = True
                    
                    if not need_reload:
                        self.last_active_time = time.time()
                        return self.active_llm

            # Otherwise load or switch model using current or requested preferences
            ctx_to_load = n_ctx if n_ctx is not None else self.active_ctx
            layers_to_load = n_gpu_layers if n_gpu_layers is not None else self.active_gpu_layers
            quant_to_load = kv_quant if kv_quant is not None else self.active_kv_quant

            self._load_internal(target_model, n_ctx=ctx_to_load, n_gpu_layers=layers_to_load, kv_quant=quant_to_load)
            self.last_active_time = time.time()
            return self.active_llm

manager = ModelManager()

# Background idle auto-unload
def idle_checker():
    while True:
        time.sleep(60)
        if manager.idle_timeout_seconds > 0 and manager.active_llm is not None:
            if time.time() - manager.last_active_time > manager.idle_timeout_seconds:
                print("[GPU Server] Idle timeout reached. Auto-unloading model from VRAM.")
                manager.unload()

threading.Thread(target=idle_checker, daemon=True).start()

# --- Real-Time Streaming Downloader with ETA & Speed ---
def background_download_task(repo_id: str, filename: str):
    global download_state
    temp_path = ""
    target_path = os.path.join(MODELS_DIR, filename)
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        print(f"\n[Downloader] Requesting: {url}")
        
        download_state["status"] = "downloading"
        download_state["repo_id"] = repo_id
        download_state["filename"] = filename
        download_state["percent"] = 0.0
        download_state["downloaded_gb"] = 0.0
        download_state["total_gb"] = 0.0
        download_state["speed_mb"] = 0.0
        download_state["eta"] = "Calculating..."
        download_state["message"] = f"Connecting to Hugging Face..."

        response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Hugging Face returned HTTP {response.status_code}")

        total_bytes = int(response.headers.get('content-length', 0))
        total_gb = round(total_bytes / (1024 ** 3), 2)
        download_state["total_gb"] = total_gb

        temp_path = os.path.join(MODELS_DIR, f"{filename}.download")
        downloaded_bytes = 0
        start_time = time.time()
        last_calc_time = start_time
        bytes_at_last_calc = 0
        chunk_size = 4 * 1024 * 1024

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded_bytes += len(chunk)
                
                now = time.time()
                elapsed = now - last_calc_time
                if elapsed >= 0.5:
                    downloaded_gb = round(downloaded_bytes / (1024 ** 3), 2)
                    pct = round((downloaded_bytes / total_bytes * 100) if total_bytes > 0 else 0, 1)
                    
                    speed_bps = (downloaded_bytes - bytes_at_last_calc) / elapsed if elapsed > 0 else 0
                    speed_mb = round(speed_bps / (1024 ** 2), 1)
                    
                    rem_bytes = max(0, total_bytes - downloaded_bytes)
                    eta_sec = int(rem_bytes / speed_bps) if speed_bps > 0 else 0
                    eta_str = f"{eta_sec // 60}m {eta_sec % 60}s remaining" if eta_sec > 0 else "Finishing up..."

                    download_state["downloaded_gb"] = downloaded_gb
                    download_state["percent"] = pct
                    download_state["speed_mb"] = speed_mb
                    download_state["eta"] = eta_str
                    download_state["message"] = f"{downloaded_gb} / {total_gb} GB ({pct}%) • {speed_mb} MB/s • {eta_str}"

                    last_calc_time = now
                    bytes_at_last_calc = downloaded_bytes

        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)

        download_state["status"] = "completed"
        download_state["percent"] = 100.0
        download_state["downloaded_gb"] = total_gb
        download_state["eta"] = "0s"
        download_state["message"] = f"Successfully downloaded {filename} ({total_gb} GB)"
        print(f"[Downloader] Finished saving to: {target_path}")

    except Exception as e:
        print(f"[Downloader Error] {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        download_state["status"] = "error"
        download_state["message"] = str(e)

# --- Pydantic Schemas ---
class DownloadRequest(BaseModel):
    repo_id: str
    filename: str

class DeleteRequest(BaseModel):
    name: str

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]]

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    gpu_layers: Optional[int] = None
    kv_quant: Optional[str] = None

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    gpu_layers: Optional[int] = None
    kv_quant: Optional[str] = None

class AnthropicMessageRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    system: Optional[Union[str, List[Any]]] = None
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.6
    stream: Optional[bool] = False
    gpu_layers: Optional[int] = None
    kv_quant: Optional[str] = None

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(content={
            "status": "online",
            "device": HARDWARE_PROFILE["device_name"],
            "active_model_in_vram": manager.active_model_name or "None",
            "active_gpu_layers": manager.active_gpu_layers,
            "downloaded_models": manager.list_models()
        })
    
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Local GPU LLM Hub Running</h1>")

@app.get("/api/hardware")
def api_hardware():
    return HARDWARE_PROFILE

@app.get("/api/status")
def api_status():
    gpu_stats = get_gpu_telemetry()
    sys_stats = get_system_telemetry()
    models = manager.list_models()
    return {
        "status": "online",
        "hardware": HARDWARE_PROFILE,
        "gpu": gpu_stats,
        "system": sys_stats,
        "inference": inference_stats,
        "active_model": manager.active_model_name,
        "active_gpu_layers": manager.active_gpu_layers,
        "active_ctx": manager.active_ctx,
        "active_kv_quant": manager.active_kv_quant,
        "downloaded_models": models,
        "idle_timeout_seconds": manager.idle_timeout_seconds,
        "download_task": download_state
    }

@app.get("/api/hf-files")
def get_hf_repo_files(repo_id: str = Query(...)):
    try:
        api = HfApi()
        items = api.list_repo_tree(repo_id)
        gguf_files = []
        for item in items:
            path = getattr(item, "path", "")
            if path.endswith(".gguf") and not path.endswith(".part"):
                size_bytes = getattr(item, "size", None)
                if size_bytes:
                    size_gb = f"{round(size_bytes / (1024 ** 3), 2)} GB"
                else:
                    size_gb = "Unknown size"
                gguf_files.append({"filename": path, "size_gb": size_gb})
        return {"repo_id": repo_id, "files": gguf_files}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/download")
def start_model_download(req: DownloadRequest):
    global download_state
    if download_state["status"] == "downloading":
        raise HTTPException(status_code=400, detail="Another download is currently in progress.")
    
    threading.Thread(target=background_download_task, args=(req.repo_id.strip(), req.filename.strip()), daemon=True).start()
    return {"status": "started", "repo_id": req.repo_id, "filename": req.filename}

@app.post("/api/delete")
def delete_model_endpoint(req: DeleteRequest):
    return manager.delete(req.name)

@app.get("/v1/models")
@app.get("/api/tags")
def list_models():
    models = manager.list_models()
    return {
        "object": "list",
        "data": [{"id": m["name"], "object": "model", "owned_by": "local-gpu", "size": m["size_gb"], "loaded": m["is_loaded"]} for m in models],
        "models": [{"name": m["name"], "model": m["name"], "details": {"family": "gguf", "size": m["size_gb"]}} for m in models]
    }

@app.get("/api/model-optimizer")
def api_model_optimizer(model: str = Query(...), ctx: int = Query(32768), kv_quant: str = Query("auto")):
    resolved = manager.resolve_model_name(model)
    if not resolved:
        raise HTTPException(status_code=404, detail="Model not found")
    model_path = os.path.join(MODELS_DIR, resolved)
    total_vram = HARDWARE_PROFILE.get("vram_total_mb", 16384)
    return calculate_optimal_settings(model_path, total_vram_mb=total_vram, target_ctx=ctx, preferred_kv=kv_quant)

@app.get("/load")
@app.post("/load")
@app.post("/v1/models/load")
def load_endpoint(model: str = Query(None), ctx: int = Query(32768), gpu_layers: Union[int, str] = Query(-1), kv_quant: str = Query("f16")):
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' parameter")
    resolved = manager.resolve_model_name(model)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found in ./models")
    parsed_layers = -2 if str(gpu_layers).lower() in ("auto", "-2") else (int(gpu_layers) if str(gpu_layers).lstrip("-").isdigit() else -1)
    manager.load(resolved, n_ctx=ctx, n_gpu_layers=parsed_layers, kv_quant=kv_quant)
    return {
        "status": "loaded",
        "active_model": manager.active_model_name,
        "context_window": manager.active_ctx,
        "n_gpu_layers": manager.active_gpu_layers,
        "kv_quant": manager.active_kv_quant
    }

@app.get("/unload")
@app.post("/unload")
@app.post("/v1/models/unload")
@app.post("/api/unload")
def unload_endpoint():
    return manager.unload()

# --- OpenAI Chat API ---
@app.post("/v1/chat/completions")
@app.post("/api/chat")
async def chat_completions(req: ChatCompletionRequest):
    llm = manager.get_llm(req.model, n_gpu_layers=req.gpu_layers, kv_quant=req.kv_quant)
    formatted_messages = []
    for m in req.messages:
        text = m.content if isinstance(m.content, str) else json.dumps(m.content)
        formatted_messages.append({"role": m.role, "content": text})

    t_start = time.time()
    try:
        if req.stream:
            def sse_generator():
                inference_stats["is_generating"] = True
                tokens_count = 0
                t_first = None
                try:
                    stream_resp = llm.create_chat_completion(
                        messages=formatted_messages,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        max_tokens=req.max_tokens,
                        stop=req.stop,
                        stream=True
                    )
                    for chunk in stream_resp:
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            tokens_count += 1
                            if t_first is None:
                                t_first = time.time()
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    err = {"error": str(e)}
                    yield f"data: {json.dumps(err)}\n\n"
                finally:
                    duration = max(0.001, time.time() - (t_first or t_start))
                    tps = round(tokens_count / duration, 1) if tokens_count > 0 else 0.0
                    inference_stats["tokens_per_second"] = tps
                    inference_stats["last_tokens_generated"] = tokens_count
                    inference_stats["last_duration_seconds"] = round(duration, 2)
                    inference_stats["is_generating"] = False
                    print(f"[Speed Telemetry] {tps} tok/s ({tokens_count} tokens in {duration:.2f}s)")

            return StreamingResponse(sse_generator(), media_type="text/event-stream")
        else:
            inference_stats["is_generating"] = True
            try:
                resp = llm.create_chat_completion(
                    messages=formatted_messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    stop=req.stop,
                    stream=False
                )
                duration = max(0.001, time.time() - t_start)
                tokens_count = resp.get("usage", {}).get("completion_tokens", 0)
                tps = round(tokens_count / duration, 1) if tokens_count > 0 else 0.0
                inference_stats["tokens_per_second"] = tps
                inference_stats["last_tokens_generated"] = tokens_count
                inference_stats["last_duration_seconds"] = round(duration, 2)
                print(f"[Speed Telemetry] {tps} tok/s ({tokens_count} tokens in {duration:.2f}s)")
                return JSONResponse(content=resp)
            finally:
                inference_stats["is_generating"] = False
    except ValueError as ve:
        err_msg = str(ve)
        print(f"\n[GPU Server Warning] ValueError: {err_msg}")
        raise HTTPException(status_code=400, detail=f"Generation Error: {err_msg}")
    except Exception as e:
        err_msg = str(e)
        print(f"\n[GPU Server Warning] Model execution error: {err_msg}")
        if any(crit in err_msg.lower() for crit in ["cuda", "memory", "out of memory", "pointer", "assert", "device"]):
            print("[GPU Server Auto-Recovery] Resetting GPU state to recover cleanly...")
            manager._close_active_llm()
        raise HTTPException(status_code=500, detail=f"Model Error: {err_msg}. Server remains online and will reload on next request.")

# --- OpenAI Completions API ---
@app.post("/v1/completions")
@app.post("/api/generate")
async def completions(req: CompletionRequest):
    llm = manager.get_llm(req.model, n_gpu_layers=req.gpu_layers, kv_quant=req.kv_quant)
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]

    t_start = time.time()
    try:
        if req.stream:
            def sse_generator():
                inference_stats["is_generating"] = True
                tokens_count = 0
                t_first = None
                try:
                    stream_resp = llm.create_completion(
                        prompt=prompt,
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                        stop=req.stop,
                        stream=True
                    )
                    for chunk in stream_resp:
                        text = chunk.get("choices", [{}])[0].get("text", "")
                        if text:
                            tokens_count += 1
                            if t_first is None:
                                t_first = time.time()
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    err = {"error": str(e)}
                    yield f"data: {json.dumps(err)}\n\n"
                finally:
                    duration = max(0.001, time.time() - (t_first or t_start))
                    tps = round(tokens_count / duration, 1) if tokens_count > 0 else 0.0
                    inference_stats["tokens_per_second"] = tps
                    inference_stats["last_tokens_generated"] = tokens_count
                    inference_stats["last_duration_seconds"] = round(duration, 2)
                    inference_stats["is_generating"] = False

            return StreamingResponse(sse_generator(), media_type="text/event-stream")
        else:
            inference_stats["is_generating"] = True
            try:
                resp = llm.create_completion(
                    prompt=prompt,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stop=req.stop,
                    stream=False
                )
                duration = max(0.001, time.time() - t_start)
                tokens_count = resp.get("usage", {}).get("completion_tokens", 0)
                tps = round(tokens_count / duration, 1) if tokens_count > 0 else 0.0
                inference_stats["tokens_per_second"] = tps
                inference_stats["last_tokens_generated"] = tokens_count
                inference_stats["last_duration_seconds"] = round(duration, 2)
                return JSONResponse(content=resp)
            finally:
                inference_stats["is_generating"] = False
    except ValueError as ve:
        err_msg = str(ve)
        print(f"\n[GPU Server Warning] ValueError: {err_msg}")
        raise HTTPException(status_code=400, detail=f"Generation Error: {err_msg}")
    except Exception as e:
        err_msg = str(e)
        print(f"\n[GPU Server Warning] Model execution error: {err_msg}")
        if any(crit in err_msg.lower() for crit in ["cuda", "memory", "out of memory", "pointer", "assert", "device"]):
            print("[GPU Server Auto-Recovery] Resetting GPU state to recover cleanly...")
            manager._close_active_llm()
        raise HTTPException(status_code=500, detail=f"Model Error: {err_msg}. Server remains online and will reload on next request.")

# --- Anthropic Messages API (Claude Code) ---
@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessageRequest):
    llm = manager.get_llm(req.model, n_gpu_layers=req.gpu_layers, kv_quant=req.kv_quant)
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    
    formatted_messages = []
    if req.system:
        sys_text = req.system if isinstance(req.system, str) else json.dumps(req.system)
        formatted_messages.append({"role": "system", "content": sys_text})
        
    for m in req.messages:
        text = m.content if isinstance(m.content, str) else json.dumps(m.content)
        formatted_messages.append({"role": m.role, "content": text})

    try:
        if req.stream:
            def anthropic_sse_generator():
                try:
                    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': manager.active_model_name, 'usage': {'input_tokens': 10, 'output_tokens': 0}}})}\n\n"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    
                    stream_resp = llm.create_chat_completion(
                        messages=formatted_messages,
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                        stream=True
                    )
                    out_tokens = 0
                    for chunk in stream_resp:
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            out_tokens += 1
                            event_data = {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": content}
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n"
                    
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                except Exception as e:
                    err = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"event: error\ndata: {json.dumps(err)}\n\n"

            return StreamingResponse(anthropic_sse_generator(), media_type="text/event-stream")
        else:
            resp = llm.create_chat_completion(
                messages=formatted_messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=False
            )
            content_text = resp["choices"][0]["message"]["content"]
            out_tokens = resp.get("usage", {}).get("completion_tokens", len(content_text) // 4)
            in_tokens = resp.get("usage", {}).get("prompt_tokens", 10)
            
            return JSONResponse(content={
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content_text}],
                "model": manager.active_model_name,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens}
            })
    except ValueError as ve:
        err_msg = str(ve)
        print(f"\n[GPU Server Warning] ValueError: {err_msg}")
        raise HTTPException(status_code=400, detail=f"Generation Error: {err_msg}")
    except Exception as e:
        err_msg = str(e)
        print(f"\n[GPU Server Warning] Model execution error: {err_msg}")
        if any(crit in err_msg.lower() for crit in ["cuda", "memory", "out of memory", "pointer", "assert", "device"]):
            print("[GPU Server Auto-Recovery] Resetting GPU state to recover cleanly...")
            manager._close_active_llm()
        raise HTTPException(status_code=500, detail=f"Model Error: {err_msg}. Server remains online and will reload on next request.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Universal Local GPU LLM Hub")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=11434, help="Port to bind")
    parser.add_argument("--idle-timeout", type=int, default=1800, help="Idle timeout in seconds (0 = disable)")
    args = parser.parse_args()

    manager.idle_timeout_seconds = args.idle_timeout

    print("=" * 65)
    print(f" Universal GPU LLM Hub running on http://{args.host}:{args.port}")
    print(f" Detected Hardware: {HARDWARE_PROFILE['device_name']} [{HARDWARE_PROFILE['device_type'].upper()}]")
    print(f" Open http://{args.host}:{args.port} in any browser on your network!")
    print("=" * 65)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
