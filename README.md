# Sparky — Offline Companion & Tutor Robot

Sparky is a fully offline, English-speaking companion robot for kids. Speech is
transcribed locally with `faster-whisper`, answered by an OpenAI-compatible LLM
server, and spoken back through `piper-tts` with a robot voice effect. ASR, TTS,
camera capture, and display stay on the robot; only LLM requests cross the
network.

```text
microphone → faster-whisper → LLM server (HTTP) → Piper TTS → speaker
                                   ▲
camera ────── JPEG data URL ───────┘
```

The app has **no in-process LLM** and no local-model fallback. On the Jetson or
Mac the server runs on loopback; on a Pi 5 it runs on a LAN or cloud host. Only
the URL, model name, and credentials differ.

For the full design, see [`ARCHITECTURE.md`](ARCHITECTURE.md) and
[`PROJECT_MISSION.md`](PROJECT_MISSION.md).

## Requirements

- Python 3.11+ (tested with 3.12.6)
- A microphone and speaker for the voice loop
- macOS on Apple Silicon, or 64-bit Linux (Raspberry Pi OS)
- An OpenAI-compatible LLM server: `llama.cpp` locally, or a reachable LAN host

## Quick start

Run the server and the app on the same machine first. Multi-device setups are
described below.

### 1. Install the client dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install faster-whisper piper-tts sounddevice numpy opencv-python-headless
```

`sounddevice` bundles PortAudio on macOS. On Linux or a Pi, install it first:

```bash
sudo apt install -y libportaudio2
```

### 2. Install llama.cpp

```bash
curl -LsSf https://llama.app/install.sh | sh
llama --version
```

Homebrew also works if you prefer it:

```bash
brew install llama.cpp
```

### 3. Download the models

```bash
python download_models.py
```

This downloads the Whisper STT model, the default LLM and its vision projector
(mmproj), and the English Piper voice into `models/`.

### 4. Start the LLM server

In a separate terminal:

```bash
./scripts/start-llm-server.sh qwen35-2b
```

The script starts `llama serve` with:

- model key → GGUF + mmproj resolved from `sparky/config/llm_registry.py`
- alias set to the model key (`qwen35-2b`)
- `127.0.0.1:8080`
- `-ngl 99` and `-c 4096`
- reasoning disabled (`-rea off`), so Qwen3.5 answers directly instead of
  spending the short spoken-reply budget on `reasoning_content`

Wait until `/health` responds:

```bash
curl -s http://127.0.0.1:8080/health
```

Other registry models: `qwen35-4b`, `gemma4-e2b`, `qwen25-1.5b`, `llama32-1b`.

### 5. Run Sparky

In the first terminal:

```bash
export ROBOT_REMOTE_URL=http://127.0.0.1:8080/v1
export ROBOT_REMOTE_MODEL=qwen35-2b

python offline_robot.py
```

Speak after `[Listening... Speak now]`. Say **"bye"**, **"exit"**, **"quit"**,
or **"goodbye"** to stop.

If exactly one model is advertised by the server, `ROBOT_REMOTE_MODEL` can be
omitted and Sparky will adopt that id.

## Verify without a microphone

```bash
# One tiny synthetic image through the server's vision path
python offline_robot.py --remote-model qwen35-2b --check-vision

# One camera photo, answer, speak, exit (needs camera permission)
python offline_robot.py --remote-model qwen35-2b --look "solve this math"
```

`--model KEY` is accepted as an alias for `--remote-model KEY`, but both now
mean the **server-side model id**, not a local file.

`--list-models` still lists the local download inventory (what is available
under `models/`); it does not query the server. Use `curl /v1/models` for the
server's ids.

## Deployments

### Standalone (Jetson or Mac)

Everything runs on one machine. The server binds to loopback, so HTTP is fine
and no API key is needed.

```text
Sparky app → 127.0.0.1:8080 → llama.cpp server
```

### Pi 5 client with a LAN server

On the LAN host, create an API-key file (one key per line; `#` starts a
comment) and start the server with TLS:

```bash
mkdir -p ~/.config/sparky
printf '%s\n' 'your-token' > ~/.config/sparky/llm-api-keys
chmod 600 ~/.config/sparky/llm-api-keys

LLAMA_ARG_HOST=0.0.0.0 \
LLAMA_ARG_API_KEY_FILE="$HOME/.config/sparky/llm-api-keys" \
LLAMA_ARG_SSL_CERT_FILE=/path/to/server.crt \
LLAMA_ARG_SSL_KEY_FILE=/path/to/server.key \
./scripts/start-llm-server.sh qwen35-2b
```

`llama.cpp` must be built with OpenSSL support (`-DLLAMA_OPENSSL=ON`) for the
TLS flags to work. The launcher refuses a non-loopback bind unless it has a
readable key file and a PEM cert/key pair; only the key-file path is passed to
the server, so the key itself never appears in the process list.

Then point the Pi at the server:

```bash
export ROBOT_REMOTE_URL=https://llm-host.local:8080/v1
export ROBOT_REMOTE_API_KEY=your-token
python offline_robot.py
```

A non-loopback URL requires **HTTPS and an API key**; Sparky fails fast
otherwise. The client key must match one line in the server's key file. For a
self-signed certificate, install the CA on the client or terminate TLS at a
reverse proxy. See `ARCHITECTURE.md` §10.

The Pi still downloads Whisper and Piper locally. It never loads the GGUF; the
GGUF and mmproj belong to the server.

## Models and assets

| Asset | Client (Pi) | Server (Jetson/Mac/LAN) |
|---|---|---|
| Whisper STT | yes | yes (for the app) |
| Piper voice | yes | yes (for the app) |
| GGUF LLM | no | yes |
| mmproj projector | no | yes, for vision |

```bash
python download_models.py                     # default set for this machine
python download_models.py --list-llms         # keys, sizes, download status
python download_models.py --llm qwen35-4b     # weights + mmproj
python download_models.py --mmproj qwen35-4b  # projector only
python download_models.py --llm all           # every registered model
```

| Key | Model | Size | Notes |
|---|---|---|---|
| `qwen35-2b` *(default)* | Qwen3.5 2B Instruct | 1.28 GB | Best speed/quality balance; vision. |
| `gemma4-e2b` | Gemma 4 E2B IT (QAT) | 2.62 GB | Highest Pi-tier quality; vision. |
| `qwen35-4b` | Qwen3.5 4B Instruct | 2.74 GB | Best tutoring quality; vision. |
| `qwen25-1.5b` | Qwen2.5 1.5B Instruct | 1.00 GB | Fast, text-only. |
| `llama32-1b` | Llama 3.2 1B Instruct | 0.81 GB | Lowest latency, text-only. |

## Configuration

Precedence is **CLI flag → environment variable → default**.

| Variable | CLI flag | Default | Meaning |
|---|---|---|---|
| `ROBOT_REMOTE_URL` | `--remote-url` | `http://127.0.0.1:8080/v1` | LLM server base URL |
| `ROBOT_REMOTE_MODEL` | `--remote-model` | auto-detect | Server-side model id |
| `ROBOT_REMOTE_API_KEY` | `--remote-api-key` | — | Bearer token |
| `ROBOT_REMOTE_TIMEOUT` | `--remote-timeout` | `60` | Request timeout (seconds) |
| `ROBOT_REMOTE_MAX_RETRIES` | `--remote-max-retries` | `2` | Retry count (reserved; retries are not implemented yet) |
| `ROBOT_REMOTE_VERIFY_TLS` | `--remote-verify-tls` / `--no-remote-verify-tls` | `true` | Verify TLS certificates |
| `ROBOT_LLM_MAX_TOKENS` | — | `96` | Spoken reply cap |
| `ROBOT_VISION_MAX_TOKENS` | — | `256` | Vision reply cap |
| `ROBOT_MEMORY_MAX_TURNS` | — | `8` | Conversation turns kept in RAM |
| `ROBOT_VOICE_STYLE` | — | `cute_companion` | Voice effect profile |
| `ROBOT_IMAGE_MAX_DIM` | — | `768` | Longest image side sent to the VLM |
| `ROBOT_IMAGE_JPEG_QUALITY` | — | `85` | JPEG quality for vision requests |
| `ROBOT_KEEP_CAPTURES` | — | `0` | Save captures under `captures/` for debugging |
| `ROBOT_CAMERA_INDEX` | — | `0` | OpenCV camera index |
| `ROBOT_CAMERA_BACKEND` | — | `auto` | `auto`, `picamera2`, or `opencv` |

Secrets come from the environment and are never logged.

### Voice style

```bash
ROBOT_VOICE_STYLE=robot_droid python offline_robot.py
```

| Style | Effect |
|---|---|
| `cute_companion` *(default)* | Pitch shift plus subtle robot resonance. |
| `robot_droid` | Pitch shift plus a stronger, more mechanical effect. |
| `natural` | Unmodified Piper voice, no pitch shift or DSP. |

### Conversation memory

Sparky remembers the most recent turns of the session, including "look and ..."
turns as short text questions and answers. Photos and image data are never
stored. Memory is RAM-only and cleared when the program exits. Set
`ROBOT_MEMORY_MAX_TURNS` to change the limit (default `8`).

### Camera (macOS only)

OpenCV needs camera permission before the **"look and ..."** feature works:

1. Open **System Settings → Privacy & Security → Camera**.
2. Enable access for your terminal app.
3. Quit and reopen the terminal.

Without permission, Sparky says *"I couldn't take a photo right now."*

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Could not reach the LLM server` | Start the server and check `curl http://127.0.0.1:8080/health`. |
| `Configuration error: ... HTTPS` | A non-loopback URL requires HTTPS and an API key. |
| HTTP 404 / model not found | Check `curl /v1/models`; set `ROBOT_REMOTE_MODEL` to an id the server advertises, or use the script's alias. |
| Replies are empty (`Robot:` with no text) | The model spent `max_tokens` on reasoning. Launch the server with `-rea off` (the start script does this). |
| `--check-vision` fails | Server offline, model is text-only, or the mmproj is missing on the server. |
| `port 8080 already in use` | Stop the old server or change `PORT` in `scripts/start-llm-server.sh`. |
| `--list-models` shows "not downloaded" | That is local inventory; it does not mean the server lacks the model. |
| TLS certificate error | Install the server's CA on the client, or use a reverse proxy with a trusted certificate. |

## Project layout

```text
sparky/
  asr/          microphone capture and faster-whisper transcription
  tts/          text cleanup, Piper voice, robot effects, playback
  llm/          HTTP provider, factory, text/vision inference
    backends/   LLMProvider interface and RemoteHTTPProvider
  vision/       camera capture, one-shot vision turn, server vision check
  config/       runtime config, paths, GGUF/mmproj registry
  commands.py   exit and "look and ..." intent parsing
  cli.py        command-line arguments and model inventory

offline_robot.py             application entry point and main loop
download_models.py           model downloader (client + server assets)
scripts/start-llm-server.sh  llama.cpp server launcher
```

## Quick reference

| Task | Command |
|---|---|
| Start server | `./scripts/start-llm-server.sh qwen35-2b` |
| Run voice loop | `ROBOT_REMOTE_MODEL=qwen35-2b python offline_robot.py` |
| One-shot vision | `python offline_robot.py --remote-model qwen35-2b --look "read this"` |
| Check server vision | `python offline_robot.py --remote-model qwen35-2b --check-vision` |
| List local assets | `python offline_robot.py --list-models` |
| List server models | `curl http://127.0.0.1:8080/v1/models` |
| Download models | `python download_models.py` |

Tested with: `faster-whisper 1.2.1`, `piper-tts 1.8.0`, `sounddevice 0.5.6`,
`numpy 2.5.4`, `opencv-python-headless 5.1.0`.
