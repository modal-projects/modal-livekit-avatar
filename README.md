# Deploy a conversational avatar with LiveKit and Modal

A voice agent with a photorealistic avatar.

The **avatar runs on [Modal](https://modal.com)** while the
voice agent runs on LiveKit Cloud (or locally during development). Text-to-speech is
also deployed on Modal using the open-weights [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 
model served with vLLM-Omni. OpenAI Realtime is used for Specch-to-text and LLM generation.

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
  `openai.TTS` plugin. The agent also mints an avatar session token, and starts 
  a sticky session on the Expression server (`sessions.start`). 
  It then `POST`s `/launch` with that session's token. Session audio is routed 
  to the avatar via `DataStreamAudioOutput`, and the greeting waits until the avatar is
  publishing video. When the LiveKit job ends, a shutdown callback
  terminates the Modal session.

- **Expression Server** (`bithuman/expression_server.py`): a Modal Server that
  runs the BitHuman GPU container and proxies its port. On container start
  (`@modal.enter`) it launches the image's own avatar server. 
  `@modal.sessioned()` enables Modal Sessions which
  is what holds the container for the call — WebRTC traffic is invisible to
  Modal. `max_concurrency=20` is the estimated concurrency when running on an H100.

- **Qwen3-TTS** (`qwen3_tts/qwen3_tts_server.py`): a Modal server that runs the
  open-weights TTS model using vLLM-Omni. At startup it uploads a
  reference clip as a named voice, then the agent requests that voice like any
  other speaker. See [Voice cloning with Qwen3-TTS](#voice-cloning-with-qwen3-tts).

## Prerequisites

- A [Modal](https://modal.com) account. Step 1 installs the `modal` CLI; authenticate
  it there if you have not already
- A [LiveKit](https://livekit.io) Cloud project (or a self-hosted server)
- An OpenAI API key (for the Realtime model). Speech itself is open-weights
  Qwen3-TTS, so there is no separate TTS API key
- A [BitHuman](https://bithuman.io) API secret

## Setup

### 1. Install dependencies

From the repository root:

```bash
uv sync
```

This installs the agent dependencies, including `modal` and `dotenv`. Later
steps call that CLI with `uv run modal`. If it is not authenticated yet:

```bash
uv run modal setup
```

### 2. Create the Modal secrets

The Expression container reads the BitHuman API secret from a Modal secret (it
uses that to license the model). The agent worker sends the same secret as the
`Authorization` bearer on `POST /launch`, and reads LiveKit and OpenAI
credentials from its own environment (step 6):

```bash
uv run modal secret create bithuman \
  BITHUMAN_API_SECRET=your-bithuman-secret
```

### 3. Deploy the Expression server to Modal

From the repository root:

```bash
uv run modal deploy bithuman/expression_server.py
```

The container downloads ~5 GB of weights into the `bithuman-expression-models`
volume on first start, cached for later containers.

### 4. Deploy the Qwen3-TTS server

```bash
uv run modal deploy qwen3_tts/qwen3_tts_server.py
```

The agent resolves the URL with
`modal.Server.from_name("qwen3-tts-charles", "Qwen3TTS").get_url()`. The
shipped voice is a clone of Charles; see
[Voice cloning with Qwen3-TTS](#voice-cloning-with-qwen3-tts) to use a
different reference clip. Weights land in the `huggingface-cache` and
`vllm-cache` volumes on first start.

### 5. Set the avatar portrait (optional)

The avatar face is `agent/avatar.jpg`. The agent uploads it on each
`POST /launch`. To use a different face, replace that file and restart the
agent worker. The Expression server does not need a redeploy for a new face.

### 6. Configure the local environment

The agent worker is a normal LiveKit agent. Put a `.env` at the repository
root (the worker loads that file, not one next to the script). It needs the
standard `LIVEKIT_*` variables, `OPENAI_API_KEY` for the Realtime model, and
`BITHUMAN_API_SECRET` for the `/launch` bearer (the same value as the Modal
`bithuman` secret).

### 7. Run the agent worker

Run the worker like any other LiveKit agent — it does not run on Modal:

From the repository root:

```bash
# local development
uv run agent/agent_worker.py dev

# or deploy to LiveKit Cloud
uv run agent/agent_worker.py start
```

### 8. Test with the Agent Playground

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
uv run modal deploy qwen3_tts/qwen3_tts_server.py
```