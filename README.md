# Sparky — Offline Companion & Tutor Robot

> [!WARNING]
> **README is being rewritten.** This document still describes the old
> in-process `llama.cpp` design. Sparky is moving to a single HTTP-only,
> OpenAI-compatible LLM server (`ROBOT_REMOTE_*` settings), with no in-process
> fallback. For the current design, see:
>
> - [`PROJECT_MISSION.md`](PROJECT_MISSION.md) — goals, deployments, privacy
> - [`ARCHITECTURE.md`](ARCHITECTURE.md) — modules, config, provider, SLOs
> - [`TASKS.md`](TASKS.md) — roadmapped work
>
> A full README rewrite is tracked as **A13**.

Sparky is a fully offline, English-speaking robot. Speech is transcribed with
`faster-whisper`, answered by a local GGUF language model, and spoken back
through `piper-tts` with a robot voice effect.

```
microphone → faster-whisper → local LLM (llama.cpp) → Piper TTS → speaker
```

## Project layout

```text
sparky/
  asr/          microphone capture and faster-whisper transcription
  tts/          text cleanup, Piper voice, robot effects, and playback
  llm/          local text and vision inference
  vision/       camera capture, one-shot vision turn, projector check
  config/       model paths, LLM registry, and runtime settings
  commands.py   exit and "look and ..." intent parsing
  cli.py        command-line arguments and model listing

offline_robot.py    application entry point and main conversation loop
download_models.py  model downloader
```

## Before you start

- Python 3.11+ (tested with 3.12.6)
- A working microphone and speaker
- macOS on Apple Silicon, or a 64-bit Linux system (Raspberry Pi OS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# llama.cpp bindings are compiled locally, so this takes a few minutes.
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python    # macOS
# CMAKE_ARGS="-DGGML_NATIVE=on" pip install llama-cpp-python  # Raspberry Pi 5 / Linux

pip install faster-whisper piper-tts sounddevice numpy opencv-python-headless
```

`sounddevice` bundles PortAudio on macOS, so no extra install is needed there.
On Linux or a Raspberry Pi, install it first:

```bash
sudo apt install -y libportaudio2
```

### Camera permissions (macOS only)

On macOS, OpenCV needs camera access. Before using the **"look and ..."** feature,
grant your terminal (Terminal, iTerm2, VS Code, etc.) camera permission:

1. Open **System Settings → Privacy & Security → Camera**.
2. Enable camera access for your terminal app.
3. Quit and reopen the terminal so the permission takes effect.

Without permission, Sparky will say *"I couldn't take a photo right now."*

Tested with: `faster-whisper 1.2.1`, `llama-cpp-python 0.3.36`,
`piper-tts 1.8.0`, `sounddevice 0.5.6`, `numpy 2.5.4`,
`opencv-python-headless 5.1.0`.

---

## 1. Download models — `download_models.py`

Run this once before starting the robot. It downloads the Whisper STT model,
the default LLM, its vision projector, and the English Piper voice into
`models/`.

```bash
python download_models.py
```

### Options

| Command | What it does |
|---|---|
| `python download_models.py` | Download the default set (Whisper + default LLM + its mmproj + Piper voice). |
| `python download_models.py --list-llms` | Show available model keys, sizes, and download status. |
| `python download_models.py --llm qwen35-2b` | Download one specific LLM. |
| `python download_models.py --llm all` | Download every registered LLM and vision projector (about 10.8 GB). |

Already-downloaded files are skipped, so re-running is safe.

### Available LLMs

| Key | Model | Size | When to use |
|---|---|---|---|
| `qwen35-2b` *(default)* | Qwen3.5 2B Instruct | 1.28 GB | Best speed/quality balance on a Raspberry Pi 5. Supports vision with mmproj. |
| `gemma4-e2b` | Gemma 4 E2B IT (QAT) | 2.62 GB | Highest quality in the Pi tier; supports audio and vision. mmproj downloads automatically. |
| `qwen35-4b` | Qwen3.5 4B Instruct | 2.74 GB | Best tutoring quality, but slow on a Pi; ideal on a Mac GPU. Supports vision; mmproj downloads automatically. |
| `qwen25-1.5b` | Qwen2.5 1.5B Instruct | 1.00 GB | Fast fallback with strong math for its size. No vision. |
| `llama32-1b` | Llama 3.2 1B Instruct | 0.81 GB | Lowest latency; weakest at math. No vision. |

Model files are stored under `models/` (git-ignored).

Vision-capable models also need their vision projector (mmproj) downloaded.
`download_models.py` fetches it automatically when you download a vision-capable
LLM, or you can fetch it separately:

```bash
python download_models.py --llm gemma4-e2b      # LLM weights + mmproj
# or grab just the projector:
python download_models.py --mmproj gemma4-e2b   # vision projector only
```

---

## 2. Talk to the robot — `offline_robot.py`

```bash
python offline_robot.py
```

Speak after `[Listening... Speak now]`. Sparky listens until you stop talking,
then replies. Say **"bye"**, **"exit"**, **"quit"**, or **"goodbye"** to stop.

### Conversation memory

Sparky remembers the most recent turns of the current session, so it can use
things you told it earlier (like your name) in later replies. "Look and ..."
turns are remembered too, as short text questions and answers, so Sparky can
refer back to what it saw — no photo or image data is ever stored. Memory lives
only in RAM: it is cleared when the program stops and nothing is saved to disk.
Use `ROBOT_MEMORY_MAX_TURNS` to change how many turns are remembered
(default `8`).

### "Look and ..." vision feature

Say **"look and <task>"** and Sparky will take a photo with the camera and ask
the vision-capable LLM to do something with it. For example:

- "look and solve this math" → captures the problem and speaks the solution
- "look and read this" → reads text visible in the photo
- "look and tell me what this is" → identifies what's in the frame

**Requirements:**

- A working camera (OpenCV reads `ROBOT_CAMERA_INDEX`, default `0`; on the Pi
  a CSI camera is used via `picamera2` when available).
- A vision-capable LLM: `gemma4-e2b` or `qwen35-4b` (or `qwen35-2b`).
- The model's vision projector (mmproj) downloaded (see above).

If the current model does not support vision, Sparky replies **"I can't see
right now"** and takes no photo.

Photos are processed in memory and not saved by default. Set
`ROBOT_KEEP_CAPTURES=1` to save frames under `captures/` for debugging.

### One-shot vision mode (debugging)

```bash
python offline_robot.py --model gemma4-e2b --look "solve this math"
```

Captures one photo, answers, speaks, and exits. It does not load Whisper or
play the greeting, so it also works without a microphone.

### Check the vision projector

```bash
python offline_robot.py --model gemma4-e2b --check-vision
```

Loads the selected model, runs one tiny in-memory image through the projector,
prints the answer, and exits. Use this after changing models or downloading a
projector. It exits nonzero if vision is unavailable.

### Choose an LLM

```bash
# Command-line flag (highest priority)
python offline_robot.py --model gemma4-e2b

# Environment variable
ROBOT_LLM_MODEL=qwen25-1.5b python offline_robot.py

# List models and see which are downloaded
python offline_robot.py --list-models
```

Precedence: `--model` → `ROBOT_LLM_MODEL` → `qwen35-2b`.

On a Raspberry Pi 5, disable GPU offload (the Pi has no usable GPU backend):

```bash
ROBOT_LLM_GPU_LAYERS=0 python offline_robot.py
```

### Choose a voice style

```bash
ROBOT_VOICE_STYLE=robot_droid python offline_robot.py
```

| Style | Effect |
|---|---|
| `cute_companion` *(default)* | Pitch shift plus subtle robot resonance. |
| `robot_droid` | Pitch shift plus stronger, more mechanical robot effect. |
| `natural` | Unmodified Piper voice, no pitch shift or DSP. |

---

## Quick reference

| Setting | Flag / environment variable | Default |
|---|---|---|
| LLM model | `--model KEY` / `ROBOT_LLM_MODEL` | `qwen35-2b` |
| List models | `--list-models` | — |
| One-shot look | `--look TASK` | — |
| Check vision | `--check-vision` | — |
| Voice style | `ROBOT_VOICE_STYLE` | `cute_companion` |
| GPU layers | `ROBOT_LLM_GPU_LAYERS` | `-1` (Mac Metal); use `0` on a Pi |
| CPU threads | `ROBOT_LLM_THREADS` | `4` |
| Max reply tokens | `ROBOT_LLM_MAX_TOKENS` | `96` |
| Remembered turns | `ROBOT_MEMORY_MAX_TURNS` | `8` |
| Max vision tokens | `ROBOT_VISION_MAX_TOKENS` | `256` |
| Download one LLM | `download_models.py --llm KEY` | default model |
| Download all LLMs | `download_models.py --llm all` | — |
| List downloadable LLMs | `download_models.py --list-llms` | — |
| Camera index | `ROBOT_CAMERA_INDEX` | `0` |
| Camera backend | `ROBOT_CAMERA_BACKEND` | `auto` (`auto`, `picamera2`, `opencv`) |
| Capture size | `ROBOT_CAMERA_WIDTH` x `ROBOT_CAMERA_HEIGHT` | `1280` x `720` |
| Warmup frames | `ROBOT_CAMERA_WARMUP_FRAMES` | `5` |
| Capture directory | `ROBOT_CAPTURE_DIR` | `captures/` |
| Max image side | `ROBOT_IMAGE_MAX_DIM` | `768` |
| JPEG quality | `ROBOT_IMAGE_JPEG_QUALITY` | `85` |
| Keep captures | `ROBOT_KEEP_CAPTURES` | `0` (`1`/`true`/`yes`/`on`) |

For the project mission, hardware targets, and design guidelines, see
[`PROJECT_MISSION.md`](PROJECT_MISSION.md).
