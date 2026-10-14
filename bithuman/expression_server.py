"""BitHuman Expression GPU container. Modal proxies its `/launch` and `/ready` ports.

A sticky session keeps the GPU up; video goes out over WebRTC.

    modal deploy bithuman/expression_server.py

https://docs.bithuman.ai/guides/deploy-self-hosted/
"""

import logging
import subprocess
import time

import modal

logger = logging.getLogger("bithuman-expression-server")
logging.basicConfig(level=logging.INFO)

# Pin the digest. `:latest` can leave Modal on a cached build with bad video.
EXPRESSION_IMAGE = (
    "sgubithuman/expression-avatar@sha256:"
    "e9325ab35468be968eb41c4132b642775a45f83def2e11ee2f6ed5297fa696b3"
)
PORT = 8089
LOCAL_READY_URL = f"http://127.0.0.1:{PORT}/ready"

MAX_CONCURRENT_SESSIONS = 20

app = modal.App(name="modal-bithuman-expression")

gpu_image = modal.Image.from_registry(EXPRESSION_IMAGE, add_python="3.12").uv_pip_install("httpx")

model_volume = modal.Volume.from_name("bithuman-expression-models", create_if_missing=True)


@app.server(
    image=gpu_image,
    gpu="H100",
    secrets=[modal.Secret.from_name("bithuman")],
    volumes={"/data/models": model_volume},
    min_containers=1,
    scaledown_window=120,
    startup_timeout=60 * 60,
    port=PORT,
    max_concurrency=MAX_CONCURRENT_SESSIONS,
    compute_region="us-west",
    routing_region="us-west",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.sessioned()
class ExpressionAvatar:
    @modal.enter(snap=True)
    def start_server(self) -> None:
        # Image venv. add_python shadows `python` on PATH.
        self._proc = subprocess.Popen(["/opt/venv/bin/python", "-m", "src.dispatcher"], cwd="/app")

        import httpx

        deadline = time.time() + 60 * 60
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"expression server exited early (code {self._proc.returncode})")
            try:
                if httpx.get(LOCAL_READY_URL, timeout=5).status_code == 200:
                    logger.info("expression server ready")
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("expression server did not become ready in time")
