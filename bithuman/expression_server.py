"""Host BitHuman's self-hosted Expression (GPU) avatar worker on Modal.

This runs the published `sgubithuman/expression-avatar` Docker image, a GPU worker
that joins a LiveKit room and streams 25 FPS lip-synced video. The image already
exposes a `/launch` and `/ready` HTTP API on port 8089, so we don't write our own
renderer — we run the container's server and dispatch to it over localhost.

Video streams out-of-band over WebRTC, so a fire-and-forget launch would let Modal
scale the GPU container down mid-call. Instead `launch` runs on the container,
tells the local server to join, then blocks until the human leaves the room, which
keeps the container alive for exactly the duration of the conversation.

See https://docs.bithuman.ai/guides/deploy-self-hosted/

Deploy with:

    modal deploy expression_server.py
"""

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

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
LOCAL_LAUNCH_URL = f"http://127.0.0.1:{PORT}/launch"
LOCAL_READY_URL = f"http://127.0.0.1:{PORT}/ready"

# how long a single conversation may run (also the Modal call timeout)
MAX_SESSION_SECONDS = 60 * 60
# give up waiting if no human ever joins (e.g. launch failed)
HUMAN_JOIN_TIMEOUT = 120

# the default avatar face is baked into the image so the agent worker never has
# to ship image bytes over the launch call
DEFAULT_AVATAR_PATH = "/assets/avatar.jpg"

app = modal.App(name="modal-bithuman-expression")

# The bithuman container image plus the small client deps our dispatch code needs.
# add_python gives Modal its own interpreter; the bithuman server keeps using its
# own venv at /opt/venv (we invoke it explicitly below).
gpu_image = (
    modal.Image.from_registry(EXPRESSION_IMAGE, add_python="3.12")
    .uv_pip_install("httpx", "livekit-agents==1.7.1")
    .add_local_file(Path(__file__).parent / "avatar.jpg", DEFAULT_AVATAR_PATH, copy=True)
)

# Encrypted weights (~5 GB) download into /data/models on first run and are cached
# here for later containers.
model_volume = modal.Volume.from_name("bithuman-expression-models", create_if_missing=True)


@app.cls(
    image=gpu_image,
    gpu="H100",
    secrets=[modal.Secret.from_name("livekit-agent"), modal.Secret.from_name("bithuman")],
    volumes={"/data/models": model_volume},
    min_containers=1,
    buffer_containers=1,
    scaledown_window=120,
    timeout=MAX_SESSION_SECONDS,
    startup_timeout=60 * 60,
    region="us-west",
    # routing_region="us-west",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=20)
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

    @modal.method()
    async def launch(
        self,
        livekit_url: str,
        livekit_token: str,
        room_name: str,
        avatar_image: bytes | None = None,
    ) -> None:
        """Tell the local server to join the room, then block until the call ends.

        `avatar_image` is an optional per-session override; when omitted we use the
        default face baked into the image at DEFAULT_AVATAR_PATH.
        """
        import httpx

        if avatar_image is None:
            avatar_image = Path(DEFAULT_AVATAR_PATH).read_bytes()

        data = {
            "livekit_url": livekit_url,
            "livekit_token": livekit_token,
            "room_name": room_name,
        }
        files = {"avatar_image": ("avatar.jpg", avatar_image, "image/jpeg")}
        headers = {"Authorization": f"Bearer {os.environ['BITHUMAN_API_SECRET']}"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                LOCAL_LAUNCH_URL, data=data, files=files, headers=headers, timeout=None
            )
            resp.raise_for_status()

        await self._wait_until_session_done(livekit_url, room_name)
        logger.info("session ended for room %s", room_name)

    async def _wait_until_session_done(self, livekit_url: str, room_name: str) -> None:
        """Block while a human is in the room; return once everyone has left.

        Keeping this coroutine running keeps the Modal call (and therefore the GPU
        container) alive for the duration of the conversation.
        """
        from livekit import api
        from livekit.protocol import models

        lkapi = api.LiveKitAPI(
            livekit_url, os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
        )
        try:
            seen_human = False
            waited_for_join = 0.0
            poll_interval = 3.0
            while True:
                try:
                    res = await lkapi.room.list_participants(
                        api.ListParticipantsRequest(room=room_name)
                    )
                    participants = res.participants
                except Exception:
                    # room no longer exists -> treat as ended
                    participants = []

                humans = [p for p in participants if p.kind == models.ParticipantInfo.Kind.STANDARD]
                if humans:
                    seen_human = True
                elif seen_human:
                    return  # everyone left
                else:
                    waited_for_join += poll_interval
                    if waited_for_join >= HUMAN_JOIN_TIMEOUT:
                        logger.warning(
                            "no human joined room %s within %ss, stopping",
                            room_name,
                            HUMAN_JOIN_TIMEOUT,
                        )
                        return

                await asyncio.sleep(poll_interval)
        finally:
            await lkapi.aclose()
