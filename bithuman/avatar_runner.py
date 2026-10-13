from __future__ import annotations

import asyncio
import logging

from bithuman import AsyncBithuman

from livekit import rtc
from livekit.agents import utils
from livekit.agents.types import TOPIC_TRANSCRIPTION
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
)
from livekit.plugins.bithuman.avatar import BithumanGenerator

logger = logging.getLogger("bithuman-avatar-example")


@utils.log_exceptions(logger=logger)
async def main(api_url: str, api_token: str, model_path: str, api_secret: str) -> None:
    # load the bithuman runtime and wrap it in the plugin's video generator
    runtime = await AsyncBithuman.create(model_path=model_path, api_secret=api_secret)
    video_gen = BithumanGenerator(runtime)

    # connect to the room
    room = rtc.Room()
    await room.connect(api_url, api_token)

    def on_transcription_received(reader: rtc.TextStreamReader, participant_identity: str):
        # ignore transcription from the room
        pass

    room.register_text_stream_handler(TOPIC_TRANSCRIPTION, on_transcription_received)

    should_stop = asyncio.Event()

    # stop when we disconnect from the room or the agent disconnects
    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant):
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            logger.info("agent %s disconnected, stopping worker", participant.identity)
            should_stop.set()

    @room.on("disconnected")
    def _on_disconnected():
        logger.info("room disconnected, stopping worker")
        should_stop.set()

    # define the avatar options from the runtime and start the runner
    output_width, output_height = video_gen.video_resolution
    avatar_options = AvatarOptions(
        video_width=output_width,
        video_height=output_height,
        video_fps=video_gen.video_fps,
        audio_sample_rate=video_gen.audio_sample_rate,
        audio_channels=1,
    )
    runner = AvatarRunner(
        room, audio_recv=DataStreamAudioReceiver(room), video_gen=video_gen, options=avatar_options
    )
    tasks: list[asyncio.Task] = []
    try:
        await runner.start()

        # run until stopped or the runner is complete/failed
        tasks = [
            asyncio.create_task(runner.wait_for_complete()),
            asyncio.create_task(should_stop.wait()),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        await utils.aio.cancel_and_wait(*tasks)
        await runner.aclose()
        await video_gen.stop()
        await room.disconnect()
        logger.info("avatar runner stopped")


if __name__ == "__main__":
    import os

    from livekit.agents.cli.log import setup_logging

    setup_logging("DEBUG", devmode=True, console=True)

    livekit_url = os.getenv("LIVEKIT_URL")
    livekit_token = os.getenv("LIVEKIT_TOKEN")
    model_path = os.getenv("BITHUMAN_MODEL_PATH")
    api_secret = os.getenv("BITHUMAN_API_SECRET")
    assert livekit_url and livekit_token, "LIVEKIT_URL and LIVEKIT_TOKEN must be set"
    assert model_path, "BITHUMAN_MODEL_PATH must be set"
    assert api_secret, "BITHUMAN_API_SECRET must be set"

    asyncio.run(main(livekit_url, livekit_token, model_path, api_secret))
