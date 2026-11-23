# Sparky Architecture

This document describes the target architecture for Sparky: one codebase in
which **every deployment talks to an OpenAI-compatible LLM server over HTTP**.
On the Jetson that server runs on the same machine over loopback; on the Pi it
runs on a LAN or cloud host. Only the server URL, model name, and credentials
differ between deployments. The app itself contains no in-process LLM.

This is a design document. Some modules described here are not yet implemented,
and the migration keeps a working in-process path until remote vision is
proven (see §14). See [`TASKS.md`](TASKS.md) for execution order and status.

## 1. Goals and Non-Goals

### Goals
- One inference code path: a single HTTP provider for all deployments.
- One test surface: contract tests against the HTTP API cover everything.
- ASR, TTS, camera capture, display, and real-time audio stay **local** on both
  deployments.
- The LLM server is a **first-class component** with its own build, version,
  health check, warm-up, restart policy, and logs.
- Responsive interaction: stream tokens, speak per sentence, keep the face
  animating during inference and server startup.
- New capabilities (piano, singing, stories, calls, movement) are pluggable
  skills.
- Child privacy: in-memory state only, no recordings by default, no content in
  logs.

### Non-goals
- No in-process LLM (no `llama-cpp-python`) in the robot app.
- No AI HAT+ / Hailo backend.
- No per-device configuration profiles.
- No `local`/`remote` backend switch — all deployments use HTTP.
- No local-model fallback when the server is unavailable.
- No cloud dependency by default.

> The `LLMProvider` interface remains as the extension seam. If an in-process
> provider is ever needed again, it can be added without changing callers.

## 2. Deployment Modes

### 2.1 Pi 5 client (LAN/cloud server)

```text
┌──────────────────────────── Raspberry Pi 5 ────────────────────────────┐
│  microphone ─► Whisper (local) ─► skill router ──────────────────┐     │
│                                                                  │     │
│  camera ─────► JPEG capture (local) ─────────────────────────────┤     │
│                                                                  ▼     │
│  display ◄──── face/text UI ◄──── response ◄──── HTTP client ───────────────► LAN / cloud
│                                                                  │     │      OpenAI-compatible
│  speaker ◄──── Piper + DSP (local) ◄──── sentence stream ◄───────┘     │      /v1/chat/completions
│                                                                        │
│  audio DSP (pitch/onset/AEC) — local, independent of the network       │
└────────────────────────────────────────────────────────────────────────┘
```

- Only LLM/VLM requests cross the network.
- No local model is loaded; failures produce a spoken error, not a fallback.
- The server binds to a LAN address only when an API key and TLS are enabled;
  otherwise it binds to loopback.

### 2.2 Jetson Orin Nano Super (standalone, server on device)

```text
┌──────────────────────── Jetson Orin Nano Super ────────────────────────┐
│                                                                        │
│  ┌── llama.cpp server (127.0.0.1:8080, CUDA -ngl 99) ─────────────┐    │
│  │  GGUF model + mmproj projector; /health; warm-up                │    │
│  └───────────────────────────▲────────────────────────────────────┘    │
│                              │ loopback HTTP                           │
│  microphone ─► Whisper ─► skill router ─► HTTP client ─────────────────┘
│  camera ─────► JPEG ─────────────────────► VLM request (image_url)
│  speaker ◄──── Piper + DSP ◄──────────── streamed tokens
│  display ◄──── face/text UI
│  audio DSP (pitch/onset/AEC) — local
└────────────────────────────────────────────────────────────────────────┘
```

- Fully offline. The only "remote" call is to localhost.
- GPU offload is configured once on the server (`-ngl 99`); the Python app has
  no GPU flags and keeps its CPU cores free for audio and UI.

## 3. Current Code Layout (as of this writing)

```text
offline_robot.py            entry point + main conversation loop
download_models.py          model downloader (Whisper, GGUF, mmproj, Piper)

sparky/
  asr/
    microphone.py           energy-based VAD recording (sounddevice)
    whisper.py              faster-whisper load + transcribe
  tts/
    speaker.py              Piper synthesis, robot DSP, playback
    voice.py                cached voice loading
    effects.py              pitch shift + robot DSP
    text.py                 text cleanup for speech
  llm/
    conversation.py         bounded in-memory conversation history
    loader.py               in-process GGUF + mmproj loading
    text_llm.py             text chat completion
    vision_llm.py           VLM chat completion
    prompts.py              system prompts
  vision/
    camera.py               still capture (picamera2 / OpenCV)
    turn.py                 one-shot photo turn
    check.py                projector diagnostic
  config/
    settings.py             env-driven runtime settings
    paths.py                model paths
    llm_registry.py         local GGUF registry
  commands.py               exit / "look and ..." parsing
  cli.py                    argument parsing and model listing

tests/
  test_conversation.py      (mocks llm.create_chat_completion)
  test_vision.py
```

Today everything is local-GGUF. The provider and server work below replaces
that path.

## 4. Target Module Layout

New or substantially changed modules are marked with `*`.

```text
sparky/
  config/
    runtime.py          * resolved RuntimeConfig (CLI → env → default)
    settings.py           low-level env helpers and defaults
    paths.py              model paths
    llm_registry.py       * GGUF/mmproj inventory for server provisioning
  llm/
    backends/
      base.py           * LLMProvider interface + capabilities
      remote_http.py    * OpenAI-compatible HTTP client (only provider)
      factory.py        * builds the single provider from RuntimeConfig
    conversation.py       unchanged (text-only memory)
    prompts.py            updated for dual speech/screen output
    text_llm.py         * calls provider.chat / stream_chat
    vision_llm.py       * calls provider.vision_chat
    loader.py             removed after the vision gate passes (§14)
  display/              * face + text rendering thread
  skills/               * intent router, event bus, skill base class
  audio/                * low-latency capture, pitch/onset detection, AEC
  vision/                 camera / turn / check (calls provider)
  navigation/           * optional motor + planner (conditional)
  diagnostics.py        * --status doctor command (incl. server health)

scripts/
  start-llm-server.sh   * llama.cpp server launch (flags, env, logs)
  sparky.service        * systemd unit example for the Jetson
```

## 5. Configuration

Resolved once at startup into a `RuntimeConfig`. Precedence:

**CLI flag → environment variable → built-in default.**

There is no backend switch and no device profile. Both deployments use the
same variables; only the URL, model name, and credentials differ. The
`ROBOT_REMOTE_*` prefix means "over HTTP": on the Jetson the server is on
loopback, so "remote" does not imply off-device.

### 5.1 Core settings

| Variable | Values | Default | Meaning |
|---|---|---|---|
| `ROBOT_REMOTE_URL` | URL | `http://127.0.0.1:8080/v1` | LLM server base URL |
| `ROBOT_REMOTE_MODEL` | string | — | Server-side model name |
| `ROBOT_REMOTE_API_KEY` | string | — | Bearer token (required for non-loopback) |
| `ROBOT_REMOTE_TIMEOUT` | seconds | `60` | Per-request timeout |
| `ROBOT_REMOTE_MAX_RETRIES` | int | `2` | Retries on transient failures |
| `ROBOT_REMOTE_VERIFY_TLS` | bool | `true` | TLS certificate verification |
| `ROBOT_LLM_MAX_TOKENS` | int | `96` | Spoken reply cap |
| `ROBOT_MEMORY_MAX_TURNS` | int | `8` | Conversation turns kept in RAM |
| `ROBOT_VISION_MAX_TOKENS` | int | `256` | Vision reply cap |
| `ROBOT_IMAGE_MAX_DIM` | int | `768` | Max image side sent to the VLM |

If `ROBOT_REMOTE_MODEL` is unset and the server advertises exactly one model at
`/v1/models`, the client may adopt that id automatically.

Removed from app config (they now belong to the server, §7):
`ROBOT_LLM_BACKEND`, `ROBOT_LLM_GPU_LAYERS`, `ROBOT_LLM_THREADS`.

### 5.2 Validation rules
- `ROBOT_REMOTE_MODEL` is required unless the client can auto-detect a single
  model from `/v1/models`.
- A non-loopback `ROBOT_REMOTE_URL` requires an API key and HTTPS. For a
  self-signed LAN certificate, install the CA on the client or terminate TLS at
  a reverse proxy. Ollama's auth is limited, so prefer llama.cpp server when
  binding beyond loopback.
- Secrets come from the environment and are never logged.
- Invalid values fail fast at startup with a clear message.

### 5.3 Client assets vs server assets
- **Pi client**: Whisper tiny.en + Piper voice only. The Pi never downloads or
  loads a GGUF model.
- **Jetson**: both sets — Whisper + Piper for the app, GGUF + mmproj for the
  on-device server.
- **Dedicated LAN server**: GGUF + mmproj only.
- `download_models.py` provisions whichever sets are needed; see §7.

## 6. LLM Provider

```python
# sparky/llm/backends/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

@dataclass(frozen=True)
class Capabilities:
    supports_vision: bool
    supports_streaming: bool

@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    detail: str = ""

class LLMProvider(ABC):
    name: str
    capabilities: Capabilities

    @abstractmethod
    def chat(self, messages, *, max_tokens, temperature, **kwargs) -> str: ...

    @abstractmethod
    def stream_chat(self, messages, *, max_tokens, temperature, **kwargs) -> Iterator[str]: ...

    @abstractmethod
    def vision_chat(self, messages, *, max_tokens, temperature, **kwargs) -> str: ...

    @abstractmethod
    def health(self) -> HealthStatus: ...

    def close(self) -> None:
        return None
```

Messages use the OpenAI chat shape already produced by
`sparky/llm/conversation.py` and `sparky/llm/vision_llm.py`. Vision messages use
`{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`.

### 6.1 RemoteHTTPProvider (the only provider)
- `POST {ROBOT_REMOTE_URL}/chat/completions` with `model`, `messages`,
  `max_tokens`, `temperature`, `stream`.
- Non-streaming: `choices[0].message.content`.
- Streaming: SSE `data: {...}` lines ending with `data: [DONE]`. The parser
  handles split multi-byte UTF-8, keep-alive comments, and `reasoning_content`
  fields. If a server ignores `stream=true`, fall back to a non-streaming
  request — this is a **transport** fallback, not a model fallback.
- Vision: same `image_url` data-URL messages as before.
- Auth: `Authorization: Bearer <key>` when configured.
- Retries transient errors (timeouts, 503 while loading, 5xx) with backoff;
  does not retry 4xx.
- Typed errors on final failure; no local model is ever loaded.

### 6.2 Factory
```python
def build_provider(config: RuntimeConfig) -> LLMProvider:
    return RemoteHTTPProvider(config)
```

## 7. LLM Server Provisioning & Lifecycle

This is a first-class component, not a detail of the app.

### 7.1 Primary server: llama.cpp server
- Build llama.cpp with `GGML_CUDA=on` on the Jetson; pin the commit/version.
- **Flag verification is mandatory**: server flags differ between llama.cpp
  versions. Confirm `--mmproj`, `--jinja`, `--parallel`, prompt-cache, and
  mmproj GPU placement against the pinned version before use.
- Example launch (Jetson; verify flags for your pinned version):
  ```bash
  llama-server \
    -m models/qwen3.5-4b-instruct.Q4_K_M.gguf \
    --mmproj models/qwen3.5-4b-mmproj-F16.gguf \
    --jinja -ngl 99 -c 4096 --parallel 1 \
    --host 127.0.0.1 --port 8080
  ```
- Tuning: context must cover `Conversation` + reply (align
  `Conversation.context_tokens` with `-c`); `--parallel 1` avoids batching
  overhead for a single robot client; enable prompt caching if the pinned
  version supports it; CUDA offload all layers.
- Do **not** bind to `0.0.0.0` on the Jetson. For a LAN server, require an API
  key and TLS.

### 7.2 Optional server: Ollama
- Convenient for development and LAN use; its `/v1` API is compatible.
- Vision depends on the model and Modelfile setup; validate in contract tests.
- Ollama's authentication options are limited; do not expose it beyond a
  trusted network. Use llama.cpp server when API-key protection is required.
- Not the Jetson target.

### 7.3 Lifecycle
- **Startup**: launch via `scripts/start-llm-server.sh` or a `systemd` unit with
  `Restart=on-failure`; start ordering before Sparky.
- **Readiness**: `llama.cpp server` returns 503 to `/health` while loading. The
  app polls with a bounded wait and shows a `waking` face state.
- **Warm-up**: send one tiny text completion (and one tiny image for vision)
  after startup so the first user turn is fast.
- **Cold start**: 4B + mmproj can take 10–30 s; budget for it in UX and SLOs.
- **Restarts**: if the server crashes, `systemd` restarts it; the app detects
  the outage and recovers without a model fallback.
- **Logs**: rotate server logs; disable request-content and prompt logging in
  both server and client (verify the pinned llama.cpp version's verbosity and
  prompt-log flags) so no voice text/images are persisted.
- **Version pinning**: record the llama.cpp commit and the GGUF/mmproj hashes
  in deployment docs.

## 8. Data Flows

### 8.1 Chat turn
1. `record_audio()` captures an utterance (local).
2. `transcribe_audio()` runs Whisper (local).
3. The skill router classifies the text (chat, math, piano, story, call, ...).
4. The active skill calls `provider.stream_chat(...)` (HTTP to local or remote
   server).
5. Tokens are buffered into sentences; each sentence goes to `speak()` and the
   text panel; the face shows `speaking`.
6. The exchange is appended to the in-memory `Conversation`.

### 8.2 Vision turn ("look and ...")
1. Router matches the look intent; `capture_jpeg()` runs locally.
2. The image becomes a base64 data URL.
3. `provider.vision_chat(...)` sends it to the server (same code on both
   deployments).
4. The answer is spoken and rendered on screen; only a short text summary is
   remembered (never image bytes).

### 8.3 Face state machine
```text
idle ──speech detected──► listening ──utterance done──► thinking
  ▲                                                         │
  │                                                 first sentence
  │                                                         ▼
  └──────── reply finished ◄──────── speaking ◄─────────────┘

thinking ──(server still loading)──► waking ──(ready)──► thinking
```
`waking` is a substate of `thinking`: while the server loads, the face shows
waking; it returns to thinking once the first tokens are requested. A call or
error moves to `error`/`thinking`. The face thread runs independently so
animation never blocks inference.

### 8.4 Real-time audio (piano, singing, calls)
- Lives in `sparky/audio/` and runs regardless of LLM state.
- Pitch/onset detection and echo cancellation are CPU DSP, local on both
  deployments, and never sent to the remote brain for timing-critical feedback.

## 9. Failure Handling & Degraded Mode

| Failure | Behavior |
|---|---|
| Server unreachable at startup | Retry with backoff up to a bounded wait; show `waking`, then `error`; speak a friendly message |
| Server returns 503 (loading) | Retry; do not treat as fatal until the wait is exhausted |
| Server returns 4xx | Do not retry; log status only (no content); friendly message |
| Request timeout | Abort; friendly message; keep local UI/audio alive |
| Stream interrupted | Try a non-streaming request for the same turn (transport fallback only) |
| Vision unsupported by server | Speak "I can't see right now"; take no photo |
| Network drops mid-call | End call gracefully; return to idle |
| Local ASR/TTS failure | Report the specific failure; stay in the loop |

No local model is loaded. Local ASR/TTS/display/audio continue to work during
an outage.

## 10. Privacy & Security

- The server binds to loopback by default; non-loopback requires an API key and
  TLS with certificate verification.
- For a self-signed LAN certificate, install the CA on the Pi or terminate TLS
  at a reverse proxy. Ollama's auth is limited; prefer llama.cpp server for any
  non-loopback bind.
- A LAN server is the recommended remote target; cloud is opt-in and documented
  as sending audio/photo data off-device.
- **Neither client nor server logs message content, images, or audio.** Log
  only latency, status codes, backend/server name, and model name.
- Secrets come from the environment and are never printed.
- Conversation memory is RAM-only and cleared on exit.
- Calls use an allowlist, parental controls, encryption, and no recording.
- Content shown to children is filtered server-side and/or locally.

## 11. Testing Strategy

- **Unit**: config resolution precedence and validation, provider factory,
  sentence splitting, face transitions, startup readiness states.
- **Remote client (mock HTTP server, runs in CI)**: success, streaming, 401,
  500, 503-then-success, connection-refused-then-success, timeout, malformed
  SSE, split multi-byte tokens, `reasoning_content`, `[DONE]`, `/v1/models`
  parsing, and non-streaming fallback. Assert no content is logged.
- **Existing tests**: rewrite `test_conversation.py` / `test_vision.py` mocks
  from `llm.create_chat_completion` to the `LLMProvider` interface.
- **Real contract test (manual/nightly, on the Jetson)**: run against a pinned
  llama.cpp server with the real GGUF + mmproj; verify chat, streaming, and a
  real image; this is the vision gate (§14).
- **Compatibility**: Ollama optional; vLLM/cloud if available.
- **Hardware smoke**: Pi 5 (ASR/TTS, memory, network failure, HTTP overhead)
  and Jetson (server build, 4B tokens/s, `--check-vision`).
- Pin the llama.cpp version used for contract tests.

### Compatibility matrix (target)

| Server | Chat | Streaming | Vision | Notes |
|---|---|---|---|---|
| llama.cpp server | yes | yes | must validate | Primary; same GGUF + mmproj; `/health` |
| Ollama | yes | yes | model-dependent | Optional dev/LAN; Modelfile for vision |
| vLLM | yes | yes | model-dependent | OpenAI-compatible |
| Cloud (OpenAI-style) | yes | yes | model-dependent | Opt-in only; TLS required |

## 12. Performance SLOs

| Metric | Pi 5 client | Jetson standalone |
|---|---|---|
| ASR (3 s utterance) | ≤ ~1 s | ≤ ~1 s |
| Server cold readiness | n/a (remote) | ≤ ~30 s |
| LLM time-to-first-token | server-dependent; LAN target ≤ ~1 s | ≤ ~0.5 s |
| Time-to-first-audio | ≤ ~3 s on LAN | ≤ ~2 s |
| TTS start | ≤ ~500 ms | ≤ ~500 ms |
| Vision turnaround | ≤ ~8 s on LAN | ≤ ~15 s |
| 4B decode | server-dependent | ~20–30 tok/s |

Time-to-first-audio is measured from the end of the user's utterance and
includes ASR. LAN figures assume a server at least as fast as the Jetson;
record the actual server class with results. HTTP/SSE overhead (including
base64 image payloads) is **measured** against the in-process baseline, not
assumed negligible.

## 13. Diagnostics

`offline_robot.py --status` reports in one screen:
- effective config (URL, model, timeouts) with secrets redacted,
- server health and model/vision capability,
- mic/camera availability,
- last known latency numbers.

`--list-models` queries `/v1/models` and accepts the OpenAI shape
`{"data": [{"id": ...}]}`; if the server is offline it reports that instead of
failing.

`--check-vision` is an HTTP probe: POST a tiny in-memory image and report one
of three outcomes — server unavailable (including model-load errors) / vision
unsupported by server / vision OK (with model name).

## 14. Migration Plan (gated)

The local in-process path is retained until remote vision is proven:

1. Documentation and baseline (this doc, mission, task list, tests).
2. `RuntimeConfig` and settings; no backend switch.
3. `LLMProvider` + factory.
4. Add `RemoteHTTPProvider` **alongside** the current local path.
5. Build and launch a pinned llama.cpp server on the Jetson (TASKS A4).
6. **GATE**: prove chat, streaming, and **vision** against that server with the
   real GGUF + mmproj. If vision fails, apply the abort criteria in
   `TASKS.md` A4.4 and keep the local path.
7. Wire callers behind the provider; rewrite existing test mocks.
8. Harden the server lifecycle (systemd, readiness, warm-up, tuning,
   version pin) — TASKS A5.
9. Only after the gate passes: remove `LocalGGUFProvider`,
   `llama-cpp-python`, and in-process loader code.
10. Validate on Pi and Jetson; update docs.
11. Add skills (piano, singing, stories, calls) behind the router.
12. Optional navigation.

See [`TASKS.md`](TASKS.md) for the ordered, checkable task list.
