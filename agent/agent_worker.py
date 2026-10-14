"""LiveKit voice agent. OpenAI Realtime emits text; Qwen3-TTS speaks it; BitHuman renders video.

From the repo root:

    modal deploy bithuman/expression_server.py
    modal deploy qwen3_tts/qwen3_tts_server.py
    uv run agent/agent_worker.py dev
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import modal
from dotenv import load_dotenv
from hold_music import hold_beats
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    JobContext,
    TurnHandlingOptions,
    cli,
    room_io,
)
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.plugins import openai

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger("bithuman-expression-example")

server = AgentServer()

AVATAR_IDENTITY = "bithuman-avatar-agent"
AVATAR_AUDIO_SETTLE_DELAY = 0.75

EXPRESSION_APP_NAME = "modal-bithuman-expression"
EXPRESSION_SERVER_NAME = "ExpressionAvatar"
EXPRESSION_SERVER = modal.Server.from_name(EXPRESSION_APP_NAME, EXPRESSION_SERVER_NAME)
# Released on job shutdown. Also the cap if that callback never runs.
AVATAR_SESSION_IDLE_TIMEOUT = 60 * 60
LAUNCH_TIMEOUT = aiohttp.ClientTimeout(total=120)
DEFAULT_AVATAR_PATH = Path(__file__).parent / "avatar.jpg"

QWEN3_APP_NAME = "qwen3-tts-charles"
QWEN3_SERVER_NAME = "Qwen3TTS"
QWEN3_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def resolve_tts_base_url() -> str:
    """Deployed Qwen server URL, or `QWEN3_TTS_BASE_URL` if that lookup fails."""
    try:
        server_url = modal.Server.from_name(QWEN3_APP_NAME, QWEN3_SERVER_NAME).get_url()
        if server_url:
            return f"{server_url.rstrip('/')}/v1"
    except Exception as e:
        logger.warning("could not resolve Qwen3-TTS URL via Modal: %s", e)

    if override := os.getenv("QWEN3_TTS_BASE_URL"):
        return f"{override.rstrip('/')}/v1"

    raise RuntimeError(
        "Could not resolve the Qwen3-TTS server URL. Deploy the server "
        "(`modal deploy qwen3_tts/qwen3_tts_server.py`) or set QWEN3_TTS_BASE_URL."
    )


TTS_BASE_URL = resolve_tts_base_url()


async def launch_avatar(ctx: JobContext, avatar_identity: str) -> None:
    """Start a sticky session and POST BitHuman `/launch`. The session keeps the GPU up."""
    if not (bithuman_secret := os.getenv("BITHUMAN_API_SECRET")):
        raise RuntimeError(
            "BITHUMAN_API_SECRET is required to authorize POST /launch on the Expression server."
        )

    token = (
        api.AccessToken()
        .with_kind("agent")
        .with_identity(avatar_identity)
        .with_name("BitHuman Avatar")
        .with_grants(api.VideoGrants(room_join=True, room=ctx.room.name))
        .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: ctx.local_participant_identity})
        .to_jwt()
    )

    logger.info("starting Expression avatar session")
    avatar_session = await EXPRESSION_SERVER.sessions.start.aio(
        idle_timeout=AVATAR_SESSION_IDLE_TIMEOUT
    )

    async def _end_avatar_session() -> None:
        logger.info("terminating Expression avatar session")
        try:
            await EXPRESSION_SERVER.sessions.terminate.aio(avatar_session.token)
        except Exception:
            logger.warning("failed to terminate Expression avatar session", exc_info=True)

    ctx.add_shutdown_callback(_end_avatar_session)

    server_url = await EXPRESSION_SERVER.get_url.aio()
    form = aiohttp.FormData()
    form.add_field("livekit_url", os.environ["LIVEKIT_URL"])
    form.add_field("livekit_token", token)
    form.add_field("room_name", ctx.room.name)
    form.add_field(
        "avatar_image",
        DEFAULT_AVATAR_PATH.read_bytes(),
        filename="avatar.jpg",
        content_type="image/jpeg",
    )
    # Modal-Authorization selects the sticky session. Authorization is the BitHuman secret.
    headers = {
        "Modal-Authorization": f"Bearer {avatar_session.token}",
        "Authorization": f"Bearer {bithuman_secret}",
    }
    launch_url = f"{server_url.rstrip('/')}/launch"
    async with aiohttp.ClientSession(timeout=LAUNCH_TIMEOUT) as http:
        async with http.post(launch_url, data=form, headers=headers) as resp:
            body = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"POST /launch failed [{resp.status}]: {body}")
    logger.info("Expression avatar launch accepted for room %s", ctx.room.name)


async def wait_for_avatar_ready(
    ctx: JobContext, avatar_identity: str, timeout: float = 30.0
) -> bool:
    """Wait until the avatar is publishing audio and video. Greet anyway on timeout."""

    def is_ready(participant: rtc.RemoteParticipant) -> bool:
        kinds = {pub.kind for pub in participant.track_publications.values()}
        return rtc.TrackKind.KIND_AUDIO in kinds and rtc.TrackKind.KIND_VIDEO in kinds

    ready = asyncio.Event()

    def _check(participant: rtc.RemoteParticipant) -> None:
        if participant.identity == avatar_identity and is_ready(participant):
            ready.set()

    for p in ctx.room.remote_participants.values():
        _check(p)

    @ctx.room.on("track_published")
    def _on_published(pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant) -> None:
        _check(participant)

    @ctx.room.on("track_subscribed")
    def _on_subscribed(
        track: rtc.Track, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        _check(participant)

    try:
        await asyncio.wait_for(ready.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning("avatar not fully ready within %.0fs; greeting anyway", timeout)
        return False


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    launch_task = asyncio.create_task(launch_avatar(ctx, AVATAR_IDENTITY))

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            modalities=["text"],
            turn_detection=ServerVad(
                type="server_vad",
                threshold=0.7,
                silence_duration_ms=800,
                prefix_padding_ms=300,
                create_response=True,
            ),
        ),
        tts=openai.TTS(
            base_url=TTS_BASE_URL,
            api_key="unused",
            model=QWEN3_MODEL_NAME,
            voice="charles",
            response_format="wav",
        ),
        turn_handling=TurnHandlingOptions(interruption={"resume_false_interruption": False}),
    )

    session.output.audio = DataStreamAudioOutput(
        ctx.room,
        destination_identity=AVATAR_IDENTITY,
        wait_remote_track=rtc.TrackKind.KIND_VIDEO,
    )

    await session.start(
        agent=Agent(
            instructions=(
                "You are Charles, a friendly dev rel who works at Modal. Keep your "
                "responses concise and use short sentences."
            ),
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(audio_output=False),
    )

    bg_audio = BackgroundAudioPlayer()
    await bg_audio.start(room=ctx.room, agent_session=session)

    @asynccontextmanager
    async def hold_music() -> AsyncIterator[None]:
        handle = bg_audio.play(AudioConfig(source=hold_beats(), fade_in=1.0, fade_out=0.4))
        try:
            yield
        finally:
            handle.stop()

    async with hold_music():
        await launch_task
        await wait_for_avatar_ready(ctx, AVATAR_IDENTITY)
        await asyncio.sleep(AVATAR_AUDIO_SETTLE_DELAY)

    session.generate_reply(
        instructions="Introduce yourself as Charles from Modal in one or two short sentences."
    )


if __name__ == "__main__":
    cli.run_app(server)
