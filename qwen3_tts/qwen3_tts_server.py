"""Modal deployment for Qwen3-TTS served via vLLM-Omni.

vLLM-Omni exposes an OpenAI-compatible ``POST /v1/audio/speech`` route directly,
so we serve the vLLM-Omni OpenAI server straight on the public port and let the
stock LiveKit ``openai.TTS`` plugin talk to it with no proxy in between.

The stock plugin can only send ``input``/``model``/``voice``/``response_format``,
which is not enough to describe a voice-clone request. Instead of adapting each
request in a proxy, we register the "Charles" clone **once at container startup**
by uploading the reference audio to ``POST /v1/audio/voices`` (the runtime-upload
path). vLLM-Omni then treats ``voice="charles"`` as an uploaded voice and
automatically selects the ``Base`` (voice-clone) task for every request, so the
plugin just sends ``voice="charles"`` like any predefined speaker.

Registration runs on every cold start because Modal's filesystem is ephemeral;
this is intended and idempotent (re-uploading the same name overwrites in place).

    Deploy from the repository root with ``modal deploy qwen3_tts/qwen3_tts_server.py``.
"""

import modal

# the app name + served class name are the integration contract agent_worker.py
# uses to resolve the server URL
app = modal.App("qwen3-tts-charles")

# Volumes to cache HF weights and vLLM's compiled artifacts across cold starts.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_cache_path = "/root/.cache/huggingface"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
vllm_cache_path = "/root/.cache/vllm"

# Image mirrors the validated vLLM-Omni blueprint.
qwen3_tts_image = (
    # CUDA 13.0 toolkit matches torch's cu130 backend below so the compiled-CUDA
    # versions agree
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "libsox-fmt-all", "sox")
    .uv_pip_install("vllm==0.30.0", extra_options="--torch-backend=cu130")
    .uv_pip_install(
        "vllm-omni==0.30.0",
        "sox",
    )
    # no separate flash-attn install: vllm 0.30.0 uses flashinfer, and vllm-omni
    # only needs flash-attn under its optional `fa4` extra, which we don't enable.
    # `requests` (startup health-wait + voice registration) ships transitively, but
    # we pin it explicitly so cold start never breaks.
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

# imports for remote execution
with qwen3_tts_image.imports():
    import subprocess
    import time

    import requests

# Model + serving config. The `Base` checkpoint is the voice-clone variant; it
# clones from reference audio + transcript rather than predefined speakers.
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PUBLIC_PORT = 8000  # what @app.server exposes / the agent connects to / vLLM serves
# vLLM-Omni auto-discovers the deploy config from the checkpoint's model_type,
# so no explicit --deploy-config path is needed.
MAX_CONCURRENT_REQUESTS = 10
# Qwen3-TTS codec runs at 24 kHz mono s16le PCM.
SAMPLE_RATE = 24000

# Default reference audio + transcript for voice cloning, our dev rel Charles!
# ADD YOUR OWN VOICE REFERENCE AUDIO FILE HERE
CHARLES_REF_AUDIO = "https://modal-cdn.com/charles.wav"
# PROVIDE THE TRANSCRIPT OF THE VOICE REFERENCE AUDIO FILE HERE
CHARLES_REF_TEXT = (
    "A vision talk, uhh and, uh, a talk about what, you know, what could be done. "
    "Umm, and it's also a talk about an analogy."
)
# The agent's `voice="charles"` maps to the reference above. We register this as
# an uploaded voice at startup, so vLLM-Omni auto-selects the Base clone task.
CHARLES_VOICE = "charles"
# `POST /v1/audio/voices` requires a consent recording ID; this is a free-form
# string stored as metadata (we own the reference audio, so a static marker is fine).
CHARLES_CONSENT = "modal-charles-dev-rel-consent"


# The model is served by a vLLM subprocess whose CUDA context and child process
# don't survive a snapshot restore, so we rely on `min_containers=1` for warm starts.
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
    """Serves Qwen3-TTS directly via vLLM-Omni's OpenAI-compatible server."""

    @modal.enter()
    def start_container(self):
        """Launch vLLM-Omni on the public port, wait, then register Charles."""
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
            "--omni",  # required: selects the vLLM-Omni pipeline
        ]
        print("launching vLLM-Omni:", " ".join(cmd))
        self.vllm_proc = subprocess.Popen(cmd)
        self._wait_for_vllm()
        self._register_charles()

    def _wait_for_vllm(self, timeout: int = 60 * 30) -> None:
        """Block until the vLLM-Omni server answers /health."""
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
        """Upload the Charles reference audio as an uploaded voice at startup.

        Uses vLLM-Omni's runtime upload path (`POST /v1/audio/voices`,
        multipart/form-data). Once registered, `voice="charles"` auto-selects the
        Base voice-clone task for every `/v1/audio/speech` request, so the stock
        `openai.TTS` plugin needs no clone-specific fields.

        Runs on every cold start (Modal fs is ephemeral); re-uploading the same
        name overwrites in place, so this is idempotent.
        """
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
