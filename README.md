# ⚡ Local GPU LLM Hub & Inference Server

A high-performance, stateless, self-hosted LLM inference server and web control dashboard. Built with native support for **NVIDIA CUDA**, **Apple Silicon Metal**, and **CPU fallback** — providing unified **OpenAI**, **Anthropic (Claude Code)**, and **Ollama** API compatibility.

---

## ✨ Features

* **🤖 Auto-Hardware Detection & Optimization:** Automatically probes GPU vendor, total VRAM, and system RAM to configure optimal `n_gpu_layers`, FlashAttention-2, context lengths, and CPU threads.
* **🎛️ Dynamic GPU Offload Layer Control:** Adjust GPU layers on the fly via UI slider or API (`-1` for 100% full GPU VRAM, `0` for CPU, or `1..64` for partial offloading).
* **🌐 Universal API Compatibility:**
  * **OpenAI API:** `/v1/chat/completions`, `/v1/completions`, `/v1/models` (VS Code Continue, Cline, Roo Code, SillyTavern).
  * **Anthropic Messages API:** `/v1/messages` (Native zero-proxy support for **Claude Code CLI**).
  * **Ollama API:** `/api/tags`, `/api/generate`, `/api/chat`.
* **📥 In-Browser Model Downloader:** Search any Hugging Face GGUF repository directly from the dashboard, see exact file sizes in GB, and stream downloads with live speed (MB/s) and dynamic ETA.
* **🔒 100% Stateless & Zero Data Logging:** No database, no chat logs saved to disk. Prompts and KV cache live strictly in volatile VRAM/RAM during inference and are purged immediately.
* **🧹 Remote VRAM Flush & Idle Sleep:** 1-click VRAM purge button and configurable idle auto-unload timer to keep your GPU cool and power consumption low.

---

## 🚀 Quickstart (1-Click Setup)

### 🪟 Windows Setup
```cmd
git clone https://github.com/your-username/local-gpu-llm-hub.git
cd local-gpu-llm-hub
install.bat
start.bat
```

### 🍎 macOS & 🐧 Linux Setup
```bash
git clone https://github.com/your-username/local-gpu-llm-hub.git
cd local-gpu-llm-hub
chmod +x install.sh start.sh
./install.sh
./start.sh
```

The Web Control Dashboard will automatically launch in your browser at:
$$\Large\mathbf{\text{http://localhost:11434/}}$$

*(If accessing from another device across your local network or Tailscale, use `http://<SERVER_IP>:11434/`)*.

---

## 💻 Client Integration Guides

### 1. Claude Code CLI (Zero-Proxy Native Support)
On your client terminal, run:

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_BASE_URL = "http://<SERVER_IP>:11434"
$env:ANTHROPIC_API_KEY = "sk-local"
claude
```

**macOS / Linux (Bash):**
```bash
export ANTHROPIC_BASE_URL="http://<SERVER_IP>:11434"
export ANTHROPIC_API_KEY="sk-local"
claude
```

---

### 2. SillyTavern
1. Open SillyTavern and click **API Connections** (plug icon 🔌).
2. Set **API:** `Chat Completion`
3. Set **Chat Completion Source:** `Custom (OpenAI-compatible)`
4. Set **Custom Endpoint (Base URL):** `http://<SERVER_IP>:11434/v1`
5. Set **API Key:** `none`
6. Click **Connect** (or refresh 🔄), select your model, and start chatting!

---

### 3. VS Code (Continue.dev)
Add to your `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "Local GPU LLM Hub",
      "provider": "openai",
      "model": "default",
      "apiBase": "http://<SERVER_IP>:11434/v1",
      "apiKey": "none"
    }
  ]
}
```

---

### 4. Cline / Roo Code / Aider
* **Provider:** `OpenAI-Compatible`
* **Base URL:** `http://<SERVER_IP>:11434/v1`
* **API Key:** `none`
* **Model ID:** `default` (or specific `.gguf` name)

---

## 📊 Recommended Hardware & Model Sizing

| Hardware Tier | Detected Hardware | Recommended Models | Quantization | Optimal Context |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | `< 8 GB VRAM` / CPU | Qwen2.5-Coder-7B, DeepSeek-R1-8B | `Q4_K_M` | 4,096 – 8,192 |
| **Tier 2** | `8 – 16 GB VRAM` (RTX 5000, 4070) | DeepSeek-R1-14B, Mistral-Small-24B | `Q3_K_M` / `Q4_K_M` | 16,384 – 32,768 |
| **Tier 3** | `16 – 24 GB+ VRAM` (RTX 3090/4090) | Qwen2.5-Coder-32B, Codestral-22B | `Q4_K_M` / `Q5_K_M` | 32,768+ |
| **Tier 4** | Apple Mac Studio (32GB–128GB Unified) | Qwen 32B, DeepSeek MoE, 70B | `Q4_K_M` / `Q8_0` | 32,768+ |

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web Control Dashboard (or JSON status if `Accept: application/json`) |
| `GET` | `/api/status` | Real-time GPU telemetry, active model, and download progress |
| `GET` | `/api/hardware` | Auto-detected hardware profile and recommended model tiers |
| `GET` | `/api/hf-files?repo_id=<repo>` | Searches Hugging Face repository for `.gguf` files with exact sizes in GB |
| `POST` | `/api/download` | Starts background streaming model download with live speed and ETA |
| `POST` | `/api/delete` | Deletes a `.gguf` model file from disk |
| `GET/POST` | `/load?model=<name>&gpu_layers=-1` | Loads or dynamically switches the active model in GPU VRAM |
| `GET/POST` | `/unload` | Flushes model from memory and completely frees VRAM |
| `POST` | `/v1/chat/completions` | Standard OpenAI chat completions endpoint (streaming supported) |
| `POST` | `/v1/messages` | Anthropic Messages API (native Claude Code compatibility) |

---

## 🛡️ Security & Privacy

* **GGUF Safe Tensor Loading:** Unlike PyTorch pickle files (`.pt`/`.bin`), GGUF files contain raw tensors and cannot execute arbitrary malicious code.
* **Ephemeral Computation:** Prompts and responses exist solely in volatile GPU VRAM and are discarded immediately upon request completion.
* **Encrypted Remote Access:** Recommended for deployment behind **Tailscale** or private VPNs to prevent public port scanning.

---

## 📄 License
Released under the [MIT License](LICENSE).
