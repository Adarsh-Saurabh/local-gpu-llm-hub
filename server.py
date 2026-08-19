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
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from huggingface_hub import HfApi, hf_hub_url
from gpu_detector import detect_hardware, get_optimal_llama_params

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

class ModelManager:
    def __init__(self):
        self.active_llm: Optional[Llama] = None
        self.active_model_name: Optional[str] = None
        self.active_gpu_layers: int = -1
        self.active_ctx: int = 16384
        self.lock = threading.Lock()
        self.last_active_time: float = time.time()
        self.idle_timeout_seconds: int = 1800

    def list_models(self) -> List[Dict[str, Any]]:
        files = glob.glob(os.path.join(MODELS_DIR, "*.gguf"))
        result = []
        for f in files:
            size_gb = round(os.path.getsize(f) / (1024 ** 3), 2)
            name = os.path.basename(f)
            result.append({
                "name": name,
                "size_gb": f"{size_gb} GB",
                "is_loaded": (name == self.active_model_name and self.active_llm is not None)
            })
        return result

    def resolve_model_name(self, requested_name: Optional[str]) -> Optional[str]:
        models = self.list_models()
        if not models:
            return None
        
        if not requested_name or requested_name.strip() in ["", "default", "*", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet", "claude-3-sonnet"]:
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

    def unload(self) -> Dict[str, str]:
        with self.lock:
            if self.active_llm is not None:
                old_model = self.active_model_name
                print(f"\n[GPU Server] Unloading '{old_model}' and clearing VRAM...")
                del self.active_llm
                self.active_llm = None
                self.active_model_name = None
                gc.collect()
                print("[GPU Server] VRAM freed successfully.")
                return {"status": "unloaded", "model": old_model or "none", "message": "GPU VRAM successfully cleared."}
            return {"status": "idle", "message": "No model was currently loaded in VRAM."}

    def load(self, model_name: str, n_ctx: int = 16384, n_gpu_layers: int = -1) -> None:
        with self.lock:
            if self.active_model_name == model_name and self.active_llm is not None and self.active_gpu_layers == n_gpu_layers and self.active_ctx == n_ctx:
                self.last_active_time = time.time()
                return

            if self.active_llm is not None:
                print(f"\n[GPU Server] Dynamic Switch: Unloading '{self.active_model_name}'...")
                del self.active_llm
                self.active_llm = None
                gc.collect()

            model_path = os.path.join(MODELS_DIR, model_name)
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in ./models/")

            flash_attn = HARDWARE_PROFILE.get("flash_attn_supported", True)
            threads = HARDWARE_PROFILE.get("optimal_threads", 8)

            print(f"\n[GPU Server] Loading '{model_name}'...")
            print(f"[GPU Server] GPU Offload Layers: {n_gpu_layers} (-1 = Full GPU)")
            print(f"[GPU Server] Context Window: {n_ctx} tokens | Threads: {threads} | FlashAttn: {flash_attn}")
            
            self.active_llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=threads,
                flash_attn=flash_attn,
                verbose=False
            )
            self.active_model_name = model_name
            self.active_gpu_layers = n_gpu_layers
            self.active_ctx = n_ctx
            self.last_active_time = time.time()
            print(f"[GPU Server] Successfully loaded '{model_name}' into memory!\n")

    def delete(self, model_name: str) -> Dict[str, str]:
        with self.lock:
            if self.active_model_name == model_name and self.active_llm is not None:
                print(f"[GPU Server] Unloading '{model_name}' before deletion...")
                del self.active_llm
                self.active_llm = None
                self.active_model_name = None
                gc.collect()

            model_path = os.path.join(MODELS_DIR, model_name)
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"[GPU Server] Deleted model file: {model_path}")
                return {"status": "deleted", "message": f"Model '{model_name}' deleted successfully from disk."}
            else:
                raise HTTPException(status_code=404, detail="Model file not found on disk.")

    def get_llm(self, requested_model: Optional[str] = None, n_ctx: int = 16384, n_gpu_layers: int = -1) -> Llama:
        target_model = self.resolve_model_name(requested_model)
        if not target_model:
            raise HTTPException(status_code=404, detail="No models found on server. Please download a model first.")

        if self.active_model_name != target_model or self.active_llm is None:
            self.load(target_model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        
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
    gpu_layers: Optional[int] = -1

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    gpu_layers: Optional[int] = -1

class AnthropicMessageRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    system: Optional[Union[str, List[Any]]] = None
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.6
    stream: Optional[bool] = False
    gpu_layers: Optional[int] = -1

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
    models = manager.list_models()
    return {
        "status": "online",
        "hardware": HARDWARE_PROFILE,
        "gpu": gpu_stats,
        "active_model": manager.active_model_name,
        "active_gpu_layers": manager.active_gpu_layers,
        "active_ctx": manager.active_ctx,
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

@app.get("/load")
@app.post("/load")
@app.post("/v1/models/load")
def load_endpoint(model: str = Query(None), ctx: int = Query(16384), gpu_layers: int = Query(-1)):
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' parameter")
    resolved = manager.resolve_model_name(model)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found in ./models")
    manager.load(resolved, n_ctx=ctx, n_gpu_layers=gpu_layers)
    return {"status": "loaded", "active_model": resolved, "context_window": ctx, "n_gpu_layers": gpu_layers}

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
    gpu_layers = req.gpu_layers if req.gpu_layers is not None else -1
    llm = manager.get_llm(req.model, n_gpu_layers=gpu_layers)
    formatted_messages = []
    for m in req.messages:
        text = m.content if isinstance(m.content, str) else json.dumps(m.content)
        formatted_messages.append({"role": m.role, "content": text})

    if req.stream:
        def sse_generator():
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
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err = {"error": str(e)}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")
    else:
        resp = llm.create_chat_completion(
            messages=formatted_messages,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop,
            stream=False
        )
        return JSONResponse(content=resp)

# --- OpenAI Completions API ---
@app.post("/v1/completions")
@app.post("/api/generate")
async def completions(req: CompletionRequest):
    gpu_layers = req.gpu_layers if req.gpu_layers is not None else -1
    llm = manager.get_llm(req.model, n_gpu_layers=gpu_layers)
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]

    if req.stream:
        def sse_generator():
            try:
                stream_resp = llm.create_completion(
                    prompt=prompt,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stop=req.stop,
                    stream=True
                )
                for chunk in stream_resp:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err = {"error": str(e)}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")
    else:
        resp = llm.create_completion(
            prompt=prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stop=req.stop,
            stream=False
        )
        return JSONResponse(content=resp)

# --- Anthropic Messages API (Claude Code) ---
@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessageRequest):
    gpu_layers = req.gpu_layers if req.gpu_layers is not None else -1
    llm = manager.get_llm(req.model, n_gpu_layers=gpu_layers)
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    
    formatted_messages = []
    if req.system:
        sys_text = req.system if isinstance(req.system, str) else json.dumps(req.system)
        formatted_messages.append({"role": "system", "content": sys_text})
        
    for m in req.messages:
        text = m.content if isinstance(m.content, str) else json.dumps(m.content)
        formatted_messages.append({"role": m.role, "content": text})

    if req.stream:
        def anthropic_sse_generator():
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
