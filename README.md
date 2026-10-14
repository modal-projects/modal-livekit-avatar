# Deploy a conversational avatar with LiveKit and Modal

A voice agent with a photorealistic, talking-head
avatar, where the **avatar runs on GPU on [Modal](https://modal.com)** while the
voice agent runs on LiveKit Cloud (or locally during development). Speech is a
voice clone of a short reference clip, synthesized by the open-weights
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) Base model, also on Modal.

The avatar is [BitHuman](https://bithuman.io)'s published Expression GPU container
(`sgubithuman/expression-avatar`), which is itself a LiveKit avatar worker
exposing a `/launch` HTTP API. We host that container on a Modal GPU
(`expression_server.py`) and dispatch to it from the agent — there is no custom
avatar runner to write. See BitHuman's
[self-hosted Expression guide](https://docs.bithuman.ai/guides/deploy-self-hosted/).

## Architecture

```
┌─────────────────┐
│  Human client   │
│  (browser)      │
└────────┬────────┘
         │  WebRTC: microphone in, avatar video out
         ▼
┌────────────────────────────────────────────────────────────┐
│  LiveKit room                                              │
└────────▲───────────────────────────────────────▲───────────┘
         │ join                                  │ join
         │ speech audio via data stream          │ publish lip-sync video
         │                                       │
┌────────┴──────────────────────┐                │
│  Agent worker                 │                │
│  LiveKit Cloud or local       │                │
│                               │ session.start  │
│  OpenAI Realtime (text only)  │ POST /launch   │
└────────┬──────────────────────┘────────────────┤
         │                                       │
         │ POST /v1/audio/speech                 ▼
         ▼                          ┌────────────┬───────────────┐
┌────────┬──────────────┐           │  ExpressionAvatar          │
│  Qwen3-TTS            │           │  Modal GPU                 │
│  Modal, voice clone   │           │  sticky session holds GPU  │
└───────────────────────┘           └────────────────────────────┘
```

- **Agent Worker** (`agent/agent_worker.py`): the main LiveKit voice agent.
  Runs on LiveKit Cloud (or your machine in `dev`/`console` mode). OpenAI
  Realtime listens, detects turns, and writes the reply as text
  (`modalities=["text"]`). The worker voices that text with the Modal
  [Qwen3-TTS](#voice-cloning-with-qwen3-tts) server through the stock
  `openai.TTS` plugin, mints an avatar token, and starts a sticky session on
  the Expression server (`sessions.start`). It then `POST`s `/launch` with
  that session's token. Session audio is routed to the avatar via
  `DataStreamAudioOutput`, and the greeting waits until the avatar is
  publishing video. When the LiveKit job ends, a shutdown callback
  terminates the Modal session.

- **Expression Server** (`bithuman/expression_server.py`): a Modal Server that
  runs the BitHuman GPU container and proxies its port. On container start
  (`@modal.enter`) it launches the image's own avatar server and waits for
  `/ready`. `@modal.sessioned()` keeps the GPU up for every live session, which
  is what holds the container for the call — WebRTC traffic is invisible to
  Modal. `max_concurrency=20` is 20 sessions per H100.

- **Qwen3-TTS** (`qwen3_tts/qwen3_tts_server.py`): a Modal server that runs the
  open-weights Base checkpoint through vLLM-Omni. At startup it uploads a
  reference clip as a named voice, then the agent requests that voice like any
  other speaker. See [Voice cloning with Qwen3-TTS](#voice-cloning-with-qwen3-tts).

## Prerequisites

- A [Modal](https://modal.com) account with the `modal` CLI installed and
  authenticated
- A [LiveKit](https://livekit.io) Cloud project (or a self-hosted server)
- An OpenAI API key (for the Realtime model). Speech itself is open-weights
  Qwen3-TTS, so there is no separate TTS API key
- A [BitHuman](https://bithuman.io) API secret

## Setup

### 1. Create the Modal secrets

The Expression container reads the BitHuman API secret from a Modal secret (it
uses that to license the model). The agent worker sends the same secret as the
`Authorization` bearer on `POST /launch`, and reads LiveKit and OpenAI
credentials from its own environment (step 5):

```bash
modal secret create bithuman \
  BITHUMAN_API_SECRET=your-bithuman-secret
```

### 2. Deploy the Expression server to Modal

From the repository root:

```bash
modal deploy bithuman/expression_server.py
```

The container downloads ~5 GB of weights into the `bithuman-expression-models`
volume on first start, cached for later containers.

### 3. Deploy the Qwen3-TTS server

```bash
modal deploy qwen3_tts/qwen3_tts_server.py
```

The agent resolves the URL with
`modal.Server.from_name("qwen3-tts-charles", "Qwen3TTS").get_url()`. The
shipped voice is a clone of Charles; see
[Voice cloning with Qwen3-TTS](#voice-cloning-with-qwen3-tts) to use a
different reference clip. Weights land in the `huggingface-cache` and
`vllm-cache` volumes on first start.

### 4. Set the avatar portrait (optional)

The avatar face is `agent/avatar.jpg`. The agent uploads it on each
`POST /launch`. To use a different face, replace that file and restart the
agent worker. The Expression server does not need a redeploy for a new face.

### 5. Configure the local environment

The agent worker is a normal LiveKit agent. Put a `.env` at the repository
root (the worker loads that file, not one next to the script). It needs the
standard `LIVEKIT_*` variables, `OPENAI_API_KEY` for the Realtime model, and
`BITHUMAN_API_SECRET` for the `/launch` bearer (the same value as the Modal
`bithuman` secret).

### 6. Run the agent worker

Install the dependencies, then run the worker like any other LiveKit agent — it
does not run on Modal:

From the repository root:

```bash
uv sync

# local development
uv run agent/agent_worker.py dev

# or deploy to LiveKit Cloud
uv run agent/agent_worker.py start
```

### 7. Test with the Agent Playground

- Open the "Agents" tab in the LiveKit dashboard for your project.
- Click "Launch Console" to open the Playground. If you don't see it, wait a
  moment and refresh — the console needs to detect the registered agent.
- Click "Start Session" to join the room and connect the agent.

## Voice cloning with Qwen3-TTS

The agent's voice is
[`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base),
the open-weights **Base** checkpoint of Qwen3-TTS. Base is the voice-clone
variant: it copies a speaker from a short reference clip plus a transcript of
that clip (the model is built for about a 3-second sample). It does not pick
from a fixed speaker list. CustomVoice and VoiceDesign are different
checkpoints and cannot serve a clone request.

[`qwen3_tts/qwen3_tts_server.py`](qwen3_tts/qwen3_tts_server.py) serves that
checkpoint on an L4 with
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) (`vllm serve ... --omni`),
which exposes an OpenAI-compatible `POST /v1/audio/speech`. The LiveKit
`openai.TTS` plugin talks to it directly. The plugin can only send `input`,
`model`, `voice`, and `response_format`, which is not enough to describe a
clone on every request. The server registers the clone **once at container
startup** instead:

1. Download the reference WAV (`CHARLES_REF_AUDIO`).
2. `POST /v1/audio/voices` (multipart) with the clip, its transcript
   (`ref_text`), a voice `name`, and a `consent` string. vLLM-Omni requires
   `consent`; it is stored as metadata. `CHARLES_CONSENT` is a static marker
   because we own the shipped clip.
3. After that, `voice="charles"` selects the Base clone task automatically.
   The agent sets that name in `agent/agent_worker.py`.

Registration runs on every cold start — the container filesystem is ephemeral —
and uploading the same name again overwrites it in place. `min_containers=1`
keeps one GPU warm so a conversation does not pay that startup.

The shipped reference is Charles, at `https://modal-cdn.com/charles.wav`. To
clone a different voice, edit the constants at the top of
`qwen3_tts/qwen3_tts_server.py`:

- `CHARLES_REF_AUDIO` — URL of a WAV of the voice to clone
- `CHARLES_REF_TEXT` — a transcript that matches that clip. The Base model uses
  the transcript (not speaker embedding alone); a mismatch hurts the clone
- `CHARLES_VOICE` — the name the agent will request

Point `voice=` in `agent/agent_worker.py` at the same name, then redeploy
the TTS server:

```bash
modal deploy qwen3_tts/qwen3_tts_server.py
```

## Notes

- **Lifecycle / cost.** `expression_server.py` defaults to `min_containers=1`,
  which keeps one GPU warm for instant joins (and continuous billing). Set it to
  `0` to bill per session instead. A live sticky session keeps that GPU up until
  the agent terminates it (or until `AVATAR_SESSION_IDLE_TIMEOUT`, one hour, if
  the agent never does). The first session after scaledown pays the cold start
  (weight download/decrypt + TRT warm).
- **Ending a call.** The agent registers a LiveKit shutdown callback that calls
  `sessions.terminate`. That releases the session as soon as the job ends.
  `AVATAR_SESSION_IDLE_TIMEOUT` in `agent/agent_worker.py` is the backstop when the
  callback never runs.
- **GPU sizing.** One `H100` with `max_concurrency=20` handles 20 sticky
  sessions. A further `sessions.start` scales out another container. Adjust
  `gpu=` and `max_concurrency=` for your load and budget.
- **Decrypted weights.** The image expects a tmpfs to keep decrypted weights out
  of the writable layer. Modal has no tmpfs flag, so the container prints a
  `SECURITY:` warning; on Modal the container filesystem is ephemeral and
  isolated, so this is acceptable for the example.
- **Container startup command.** `expression_server.py` launches the image's
  baked-in server (`/opt/venv/bin/python -m src.dispatcher` in `/app`). If a
  future image tag changes this, update the `subprocess.Popen(...)` call.
- **Logs that look like failures.** A few messages show up on a healthy call:
  - `no request_id provided for TTS` — vLLM-Omni does not send the
    `x-request-id` header the OpenAI plugin reads. It substitutes `"unknown"`
    and still synthesizes audio.
  - `flush audio emitter due to slow audio generation` — a debug line. Qwen
    returns one WAV per segment, so the emitter releases that segment in a
    single flush instead of streaming frames.
  - `failed to perform clear buffer rpc` / `RpcError: Failed to send`, followed
    by `playback_finished called more times than playback segments were
    captured` — the console room disconnected while LiveKit was sending
    `lk.clear_buffer` to the avatar. The participant was already gone, so the
    RPC could not be delivered. LiveKit then marks that playout finished on a
    timer, and a real playback-finished event can arrive as well. The extra
    call is ignored. A following session that logs `playback finished event
    received` from `bithuman-avatar-agent` means the avatar path is working.
  - `FfiHandle.__del__` / `AssertionError` after `process exiting` — LiveKit
    drops a native handle after its FFI client is already shut down. It shows
    up on room disconnect and Ctrl-C and does not affect the call.
