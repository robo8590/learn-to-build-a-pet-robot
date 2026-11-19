# Sparky — Offline Companion & Tutor Robot

Sparky is a fully offline, English-speaking robot. Speech is transcribed with
`faster-whisper`, answered by a local GGUF language model, and spoken back
through `piper-tts` with a robot voice effect.

```
microphone → faster-whisper → local LLM (llama.cpp) → Piper TTS → speaker
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

Without permission, Sparky will say *"I couldn't take a photo right now."`

Tested with: `faster-whisper 1.2.1`, `llama-cpp-python 0.3.36`,
`piper-tts 1.8.0`, `sounddevice 0.5.6`, `numpy 2.5.4`,
`opencv-python-headless 5.1.0`.

---

## 1. Download models — `download_models.py`

Run this once before starting the robot. It downloads the Whisper STT model,
the default LLM, and the English Piper voice into `models/`.

```bash
python download_models.py
```

### Options

| Command | What it does |
|---|---|
| `python download_models.py` | Download the default set (Whisper + default LLM + Piper voice). |
| `python download_models.py --list-llms` | Show available model keys, sizes, and download status. |
| `python download_models.py --llm qwen35-2b` | Download one specific LLM. |
| `python download_models.py --llm all` | Download every registered LLM (about 8.5 GB). |

Already-downloaded files are skipped, so re-running is safe.

### Available LLMs

| Key | Model | Size | When to use |
|---|---|---|---|
| `qwen35-2b` *(default)* | Qwen3.5 2B Instruct | 1.28 GB | Best speed/quality balance on a Raspberry Pi 5. No vision. |
| `gemma4-e2b` | Gemma 4 E2B IT (QAT) | 2.62 GB | Highest quality in the Pi tier; supports audio and vision. Needs `--mmproj` download too. |
| `qwen35-4b` | Qwen3.5 4B Instruct | 2.74 GB | Best tutoring quality, but slow on a Pi; ideal on a Mac GPU. Supports vision. Needs `--mmproj` download too. |
| `qwen25-1.5b` | Qwen2.5 1.5B Instruct | 1.00 GB | Fast fallback with strong math for its size. No vision. |
| `llama32-1b` | Llama 3.2 1B Instruct | 0.81 GB | Lowest latency; weakest at math. No vision. |

Model files are stored under `models/` (git-ignored).

---

## 2. Talk to the robot — `offline_robot.py`

```bash
python offline_robot.py
```

Each turn records **5 seconds** of audio. Speak after `[Listening... Speak now]`.
Say **"bye"**, **"exit"**, **"quit"**, or **"goodbye"** to stop.

### "Look and ..." vision feature

Say **"look and <task>"** and Sparky will take a photo with the camera and ask the
vision-capable LLM to do something with it. For example:

- "look and solve this math" → captures the problem and speaks the solution
- "look and read this" → reads text visible in the photo
- "look and tell me what this is" → identifies what's in the frame

**Requirements:**
- A working camera (OpenCV reads `ROBOT_CAMERA_INDEX`, default `0`).
- A vision-capable LLM: `gemma4-e2b` or `qwen35-4b`.
- The model's vision projector (mmproj) downloaded:

```bash
python download_models.py --llm gemma4-e2b      # LLM weights
python download_models.py --mmproj gemma4-e2b   # vision projector
# or grab both at once:
python download_models.py --llm gemma4-e2b --mmproj gemma4-e2b
```

If the current model does not support vision (e.g. the default `qwen35-2b`),
Sparky replies **"I can't see right now"** and no photo is taken.

Photos are saved to the `captures/` directory (override with `ROBOT_CAPTURE_DIR`).

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
| Voice style | `ROBOT_VOICE_STYLE` | `cute_companion` |
| GPU layers | `ROBOT_LLM_GPU_LAYERS` | `-1` (Mac Metal); use `0` on a Pi |
| CPU threads | `ROBOT_LLM_THREADS` | `4` |
| Max reply tokens | `ROBOT_LLM_MAX_TOKENS` | `96` |
| Download one LLM | `download_models.py --llm KEY` | default model |
| Download all LLMs | `download_models.py --llm all` | — |
| List downloadable LLMs | `download_models.py --list-llms` | — |
| Camera index | `ROBOT_CAMERA_INDEX` | `0` |
| Capture directory | `ROBOT_CAPTURE_DIR` | `captures/` |

For the project mission, hardware targets, and design guidelines, see
[`PROJECT_MISSION.md`](PROJECT_MISSION.md).
