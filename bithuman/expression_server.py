"""Host BitHuman's self-hosted Expression (GPU) avatar worker on Modal.

This runs the published `sgubithuman/expression-avatar` Docker image, a GPU worker
that joins a LiveKit room and streams 25 FPS lip-synced video. The image already
exposes a `/launch` and `/ready` HTTP API on port 8089, so we don't write our own
renderer — Modal proxies that port directly.

Video streams out-of-band over WebRTC, so an idle HTTP server would let Modal
scale the GPU down mid-call. `@modal.sessioned()` holds the container for as long
as the agent keeps a sticky session open. The agent starts that session, POSTs
`/launch`, and terminates the session when the LiveKit job ends.

See https://docs.bithuman.ai/guides/deploy-self-hosted/

Deploy from the repository root:

    modal deploy bithuman/expression_server.py
"""

import logging
import subprocess
import time

import modal

logger = logging.getLogger("bithuman-expression-server")
logging.basicConfig(level=logging.INFO)

# Pin the digest, not `:latest`. `:latest` moves and Modal caches whatever digest
# it resolved at first build, so a stale `:latest` can leave us on an old build
# whose WebRTC publish-preset cap produces laggy/black video on the first turn.
# See https://docs.bithuman.ai/guides/deploy-self-hosted#troubleshooting-black-or-laggy-video
EXPRESSION_IMAGE = (
    "sgubithuman/expression-avatar@sha256:"
    "e9325ab35468be968eb41c4132b642775a45f83def2e11ee2f6ed5297fa696b3"
)
PORT = 8089
LOCAL_READY_URL = f"http://127.0.0.1:{PORT}/ready"

# one H100 serves this many concurrent conversations; a further session start
# scales out another container
MAX_CONCURRENT_SESSIONS = 20

app = modal.App(name="modal-bithuman-expression")

# add_python gives Modal its own interpreter; the bithuman server keeps using its
# own venv at /opt/venv (we invoke it explicitly below). httpx is only for the
# startup /ready poll.
gpu_image = modal.Image.from_registry(EXPRESSION_IMAGE, add_python="3.12").uv_pip_install("httpx")

# Encrypted weights (~5 GB) download into /data/models on first run and are cached
# here for later containers.
model_volume = modal.Volume.from_name("bithuman-expression-models", create_if_missing=True)


@app.server(
    image=gpu_image,
    gpu="H100",
    secrets=[modal.Secret.from_name("bithuman")],
    volumes={"/data/models": model_volume},
    min_containers=1,
    buffer_containers=1,
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
        """Start the container's own avatar server and wait until it's ready."""
        # use the image's venv python so the baked-in deps resolve (add_python may
        # have shadowed `python` on PATH)
        self._proc = subprocess.Popen(["/opt/venv/bin/python", "-m", "src.dispatcher"], cwd="/app")

        import httpx

        deadline = time.time() + 60 * 60  # first start downloads weights + warms TRT
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
