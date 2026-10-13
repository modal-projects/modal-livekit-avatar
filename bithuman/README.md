# BitHuman Expression Avatar on Modal

A voice agent with a photorealistic [BitHuman](https://bithuman.io) talking-head
avatar, where the **avatar runs on GPU on [Modal](https://modal.com)** while the
voice agent runs on LiveKit Cloud (or locally during development).

The avatar is BitHuman's published Expression GPU container
(`sgubithuman/expression-avatar`), which is itself a LiveKit avatar worker
exposing a `/launch` HTTP API. We host that container on a Modal GPU
(`expression_server.py`) and dispatch to it from the agent — there is no custom
avatar runner to write. See BitHuman's
[self-hosted Expression guide](https://docs.bithuman.ai/guides/deploy-self-hosted/).

## Architecture

```
┌─────────────────────┐   modal.Cls    ┌──────────────────────────────┐
│   Agent Worker      │ ──.spawn()────▶│  ExpressionAvatar.launch      │
│   (LiveKit Cloud)   │                │  (Modal, GPU)                 │
│                     │                │                              │
│  mints avatar token,│                │  @enter: run container's      │
│  spawns the Modal   │                │   server, wait /ready         │
│  GPU class directly │                │  POST localhost:8089/launch   │
│                     │                │  → avatar joins room (WebRTC) │
└─────────────────────┘                │  block until human leaves     │
                                       └──────────────────────────────┘
```

- **Agent Worker** (`agent_worker.py`): the main LiveKit voice agent.
  Runs on LiveKit Cloud (or your machine in `dev`/`console` mode). It starts an
  OpenAI Realtime session, mints an avatar token, and dispatches the Modal GPU
  class directly with `modal.Cls.from_name(...).launch.spawn(...)`. Session audio
  is routed to the avatar via `DataStreamAudioOutput`, and the greeting waits
  until the avatar is publishing video.

- **Expression Server** (`expression_server.py`): a Modal app that runs the
  BitHuman GPU container. On container start (`@modal.enter`) it launches the
  image's own avatar server and waits for `/ready`. Each `launch` call tells the
  local server to join the room, then blocks until the human leaves — keeping the
  GPU container alive for exactly the duration of the conversation.

## Prerequisites

- A [Modal](https://modal.com) account with the `modal` CLI installed and
  authenticated
- A [LiveKit](https://livekit.io) Cloud project (or a self-hosted server)
- An OpenAI API key (for the Realtime voice model)
- A [BitHuman](https://bithuman.io) API secret

## Setup

### 1. Create the Modal secrets

The Modal app reads credentials from two secrets — `livekit-agent` (LiveKit +
OpenAI) and `bithuman` (the BitHuman API secret):

```bash
modal secret create livekit-agent \
  LIVEKIT_URL=wss://your-project.livekit.cloud \
  LIVEKIT_API_KEY=your-api-key \
  LIVEKIT_API_SECRET=your-api-secret \
  OPENAI_API_KEY=your-openai-key

modal secret create bithuman \
  BITHUMAN_API_SECRET=your-bithuman-secret
```

### 2. Deploy the Expression server to Modal

```bash
cd examples/avatar_agents/modal_avatar/bithuman
modal deploy expression_server.py
```

The container downloads ~5 GB of weights into the `bithuman-expression-models`
volume on first start, cached for later containers.

### 3. Set the avatar portrait (optional)

The avatar face is `avatar.jpg` in this directory, baked into the Modal image at
deploy time (so the agent worker never ships image bytes on the launch call). To
use a different face, replace `avatar.jpg` and redeploy `expression_server.py`.

### 4. Configure the local environment

The agent worker is a normal LiveKit agent, so it reads credentials from the
shared `examples/.env` like every other example (see
[`examples/.env.example`](../../.env.example) and the
[examples README](../../README.md)). It needs the standard `LIVEKIT_*` variables
plus `OPENAI_API_KEY` for the Realtime model. The BitHuman secret is only used on
Modal, so it does not belong here.

### 5. Run the agent worker

Install the dependencies, then run the worker like any other LiveKit agent — it
does not run on Modal:

```bash
uv sync --all-extras --dev         # from the repository root (installs livekit-agents)
uv pip install -r requirements.txt # this example also needs modal + python-dotenv

# local development
uv run agent_worker.py dev

# or deploy to LiveKit Cloud
uv run agent_worker.py start
```

### 6. Test with the Agent Playground

- Open the "Agents" tab in the LiveKit dashboard for your project.
- Click "Launch Console" to open the Playground. If you don't see it, wait a
  moment and refresh — the console needs to detect the registered agent.
- Click "Start Session" to join the room and connect the agent.

## Notes

- **Lifecycle / cost.** `expression_server.py` defaults to `min_containers=1`,
  which keeps one GPU warm for instant joins (and continuous billing). Set it to
  `0` to bill per session instead — the launch-and-block design keeps the GPU up
  only while a call is in flight — at the cost of a cold start (weight
  download/decrypt + TRT warm) on the first session after scaledown.
- **Session-end detection.** `launch` polls the LiveKit server API and returns
  once no human participant remains. Tune `MAX_SESSION_SECONDS` /
  `HUMAN_JOIN_TIMEOUT` in `expression_server.py`; `MAX_SESSION_SECONDS` is also
  the Modal call timeout, so it caps the longest call.
- **GPU sizing.** One `H100` with `@modal.concurrent(max_inputs=20)` handles many
  concurrent sessions. Adjust `gpu=` and `max_inputs=` for your load and budget.
- **Decrypted weights.** The image expects a tmpfs to keep decrypted weights out
  of the writable layer. Modal has no tmpfs flag, so the container prints a
  `SECURITY:` warning; on Modal the container filesystem is ephemeral and
  isolated, so this is acceptable for the example.
- **Container startup command.** `expression_server.py` launches the image's
  baked-in server (`/opt/venv/bin/python -m src.dispatcher` in `/app`). If a
  future image tag changes this, update the `subprocess.Popen(...)` call.
