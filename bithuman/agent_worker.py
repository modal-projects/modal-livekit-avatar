"""Voice agent: OpenAI Realtime (text) + Qwen3-TTS → BitHuman Expression avatar.

OpenAI Realtime handles audio input, turn detection, and reasoning in text-output
mode (`modalities=["text"]`). The session voices that text with the Modal-hosted
Qwen3-TTS server via the stock `openai.TTS` plugin, and routes the audio to the
BitHuman avatar with `DataStreamAudioOutput`. The avatar's GPU container renders
the video itself, so we just mint a token and dispatch to it on Modal.

Hold music plays over the ~28s avatar cold start via `BackgroundAudioPlayer`,
which publishes its own room track (independent of the session speech we redirect
to the avatar) and stops right before the greeting.

Deploy both containers first (`modal deploy expression_server.py` and
`modal deploy ../qwen3_tts/qwen3_tts_server.py`), then run:

    python agent_worker.py dev
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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

load_dotenv()

logger = logging.getLogger("bithuman-expression-example")

server = AgentServer()

# identity the BitHuman container joins the room as
AVATAR_IDENTITY = "bithuman-avatar-agent"

# settle time after the avatar is ready before greeting, so its DataStream audio
# channel is live when the first segment arrives (a cold start can otherwise drop it)
AVATAR_AUDIO_SETTLE_DELAY = 0.75

# resolve the Modal class reference once at import, off the per-session path
EXPRESSION_AVATAR = modal.Cls.from_name("modal-bithuman-expression", "ExpressionAvatar")

# deployed Modal app + `@app.server` class that serves the Qwen3-TTS HTTP API
QWEN3_APP_NAME = "qwen3-tts-charles"
QWEN3_SERVER_NAME = "Qwen3TTS"
# The vLLM OpenAI server validates the `model` field against the served model.
QWEN3_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def resolve_tts_base_url() -> str:
    """Resolve the OpenAI-compatible base URL for the deployed Qwen3-TTS server.

    Prefers Modal's `Server.get_url()` for the deployed `@app.server` class, and
    falls back to `QWEN3_TTS_BASE_URL` if the lookup fails (e.g. before deploy).
    """
    try:
        server_url = modal.Server.from_name(QWEN3_APP_NAME, QWEN3_SERVER_NAME).get_url()
        if server_url:
            return f"{server_url.rstrip('/')}/v1"
    except Exception as e:  # noqa: BLE001 - fall back to the env override
        logger.warning("could not resolve Qwen3-TTS URL via Modal: %s", e)

    if override := os.getenv("QWEN3_TTS_BASE_URL"):
        return f"{override.rstrip('/')}/v1"

    raise RuntimeError(
        "Could not resolve the Qwen3-TTS server URL. Deploy the server "
        "(`modal deploy ../qwen3_tts/qwen3_tts_server.py`) or set QWEN3_TTS_BASE_URL."
    )


# resolve once at import so only startup pays the Modal lookup
TTS_BASE_URL = resolve_tts_base_url()


async def launch_avatar(ctx: JobContext, avatar_identity: str) -> None:
    """Mint an avatar token and dispatch the Modal-hosted Expression avatar."""
    token = (
        api.AccessToken()
        .with_kind("agent")
        .with_identity(avatar_identity)
        .with_name("BitHuman Avatar")
        .with_grants(api.VideoGrants(room_join=True, room=ctx.room.name))
        # let the avatar publish audio/video on behalf of this agent
        .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: ctx.local_participant_identity})
        .to_jwt()
    )

    logger.info("dispatching Expression avatar")
    await EXPRESSION_AVATAR().launch.spawn.aio(
        livekit_url=os.getenv("LIVEKIT_URL"),
        livekit_token=token,
        room_name=ctx.room.name,
    )


async def wait_for_avatar_ready(
    ctx: JobContext, avatar_identity: str, timeout: float = 30.0
) -> bool:
    """Block until the avatar is publishing both its audio and video tracks.

    Waiting for both means the avatar's DataStream audio channel is fully set up, so
    the first greeting segment is acked rather than dropped. Returns False on timeout
    (we greet anyway).
    """

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

    # dispatch the avatar concurrently while the session starts
    launch_task = asyncio.create_task(launch_avatar(ctx, AVATAR_IDENTITY))

    session = AgentSession(
        # Realtime handles audio input + VAD but emits text only; the session
        # voices it with Qwen3-TTS below.
        llm=openai.realtime.RealtimeModel(
            modalities=["text"],
            # tuned to reduce false barge-ins: a higher threshold ignores room noise,
            # a longer silence_duration avoids jumping in on short pauses
            turn_detection=ServerVad(
                type="server_vad",
                threshold=0.7,
                silence_duration_ms=800,
                prefix_padding_ms=300,
                create_response=True,
            ),
        ),
        # response_format="wav" so the WAV header carries the model's sample rate
        # (otherwise the plugin assumes 24 kHz PCM)
        tts=openai.TTS(
            base_url=TTS_BASE_URL,
            api_key="unused",
            model=QWEN3_MODEL_NAME,
            voice="charles",
            response_format="wav",
        ),
        # DataStreamAudioOutput doesn't support pause, so disable false-interruption resume
        turn_handling=TurnHandlingOptions(interruption={"resume_false_interruption": False}),
    )

    # gate audio forwarding on the avatar's video track (published last, so the
    # strongest single "fully ready" signal)
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
        # audio is forwarded to the avatar, so disable room audio output
        room_options=room_io.RoomOptions(audio_output=False),
    )

    # Hold music over the avatar cold start. BackgroundAudioPlayer publishes its own
    # room track, independent of the session speech we redirect to the avatar, so the
    # user hears it even though no avatar track exists yet. Stopped before the greeting.
    bg_audio = BackgroundAudioPlayer()
    await bg_audio.start(room=ctx.room, agent_session=session)

    @asynccontextmanager
    async def hold_music() -> AsyncIterator[None]:
        handle = bg_audio.play(AudioConfig(source=hold_beats(), fade_in=1.0, fade_out=0.4))
        try:
            yield
        finally:
            handle.stop()

    # Greet only once the avatar is ready to play audio, then settle briefly: greeting
    # the instant the video track appears can race the cold GPU start and drop the first
    # segment. The hold music covers this whole window.
    async with hold_music():
        await launch_task
        await wait_for_avatar_ready(ctx, AVATAR_IDENTITY)
        await asyncio.sleep(AVATAR_AUDIO_SETTLE_DELAY)

    session.generate_reply(
        instructions="Introduce yourself as Charles from Modal in one or two short sentences."
    )


if __name__ == "__main__":
    cli.run_app(server)
