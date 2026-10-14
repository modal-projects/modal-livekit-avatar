"""Qwen3-TTS voice clone, OpenAI-compatible `/v1/audio/speech`.

Set the reference clip and transcript below. The voice is registered at startup;
the agent requests `CHARLES_VOICE`.

    modal deploy qwen3_tts/qwen3_tts_server.py
"""

import modal

app = modal.App("qwen3-tts-charles")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_cache_path = "/root/.cache/huggingface"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
vllm_cache_path = "/root/.cache/vllm"

qwen3_tts_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "libsox-fmt-all", "sox")
    .uv_pip_install("vllm==0.30.0", extra_options="--torch-backend=cu130")
    .uv_pip_install(
        "vllm-omni==0.30.0",
        "sox",
    )
    .uv_pip_install("requests")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_CACHE_PATH": vllm_cache_path,
            "TORCH_CPP_LOG_LEVEL": "FATAL",
        }
    )
    .entrypoint([])
)

with qwen3_tts_image.imports():
    import subprocess
    import time

    import requests

# Base checkpoint: clone from a reference clip + transcript. Match CHARLES_VOICE in the agent.
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PUBLIC_PORT = 8000
MAX_CONCURRENT_REQUESTS = 10

# Reference clip and a transcript of that clip.
CHARLES_REF_AUDIO = "https://modal-cdn.com/charles.wav"
CHARLES_REF_TEXT = (
    "A vision talk, uhh and, uh, a talk about what, you know, what could be done. "
    "Umm, and it's also a talk about an analogy."
)
CHARLES_VOICE = "charles"
# Required by POST /v1/audio/voices. Any string; stored as metadata.
CHARLES_CONSENT = "modal-charles-dev-rel-consent"


@app.server(
    image=qwen3_tts_image,
    gpu="L4",
    volumes={
        hf_cache_path: hf_cache_vol,
        vllm_cache_path: vllm_cache_vol,
    },
    min_containers=1,
    startup_timeout=60 * 60,
    port=PUBLIC_PORT,
    target_concurrency=MAX_CONCURRENT_REQUESTS,
    compute_region="us-west",
    routing_region="us-west",
    exit_grace_period=3600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    unauthenticated=True,
)
class Qwen3TTS:
    @modal.enter()
    def start_container(self):
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(PUBLIC_PORT),
            "--max-num-seqs",
            str(MAX_CONCURRENT_REQUESTS),
            "--gpu-memory-utilization",
            "0.9",
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--omni",
        ]
        print("launching vLLM-Omni:", " ".join(cmd))
        self.vllm_proc = subprocess.Popen(cmd)
        self._wait_for_vllm()
        self._register_charles()

    def _wait_for_vllm(self, timeout: int = 60 * 30) -> None:
        start = time.time()
        while time.time() - start < timeout:
            try:
                if (
                    requests.get(f"http://localhost:{PUBLIC_PORT}/health", timeout=5).status_code
                    == 200
                ):
                    print("vLLM-Omni server is ready!")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("vLLM-Omni server failed to start within timeout period")

    def _register_charles(self) -> None:
        print(f"downloading reference audio: {CHARLES_REF_AUDIO}")
        audio_resp = requests.get(CHARLES_REF_AUDIO, timeout=60)
        audio_resp.raise_for_status()
        audio_bytes = audio_resp.content

        url = f"http://localhost:{PUBLIC_PORT}/v1/audio/voices"
        files = {
            "audio_sample": (f"{CHARLES_VOICE}.wav", audio_bytes, "audio/wav"),
        }
        data = {
            "name": CHARLES_VOICE,
            "consent": CHARLES_CONSENT,
            "ref_text": CHARLES_REF_TEXT,
        }
        print(f"registering voice '{CHARLES_VOICE}' at {url}")
        resp = requests.post(url, files=files, data=data, timeout=120)
        print(f"voice registration response [{resp.status_code}]: {resp.text}")
        if not resp.ok:
            raise RuntimeError(f"voice registration failed [{resp.status_code}]: {resp.text}")
        print(f"voice '{CHARLES_VOICE}' registered successfully")
