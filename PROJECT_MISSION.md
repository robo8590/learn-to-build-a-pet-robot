# Project Mission: Sparky — The Personal Companion & Tutor Robot

## 1. Vision & Purpose

Sparky is a personal, cute, and helpful companion robot for families. In its
first form it is a **stationary companion and tutor**: it stands in one place,
talks with family members, shows a friendly robot face and teaching text on a
small screen, and helps kids learn.

Capabilities grow over time: conversation, math tutoring, piano and singing
practice, reading books and stories, "look and ..." vision, and calling
approved family members. Movement may be added later, but the product is
designed to be useful while stationary.

## 2. Core Pillars & Roles

### A. Personality & Companion
- **Tone & Persona**: Warm, encouraging, playful, gentle, and child-friendly.
- **Safety**: Safe language, positive reinforcement, no harmful or scary
  responses.
- **Language Support**: English interaction.

### B. Educational Tutor for Kids
- **Math Mentorship**: Break problems into fun, bite-sized steps; use
  storytelling, analogies, and quizzes suitable for children.
- **Music & Piano Guidance**: Teach notes, rhythm, keys, and basic ear
  training; encourage practice with interactive games and positive feedback.
- **Singing**: Pitch and rhythm feedback, call-and-response, and sing-along.
- **Reading**: Read books and stories aloud with matching on-screen text.
- **General Advice**: Gentle life habits, curiosity prompts, and daily
  encouragement.

### C. Display & Expression
- A small screen shows a robot face with emotions (`idle`, `listening`,
  `thinking`, `waking`, `speaking`, `happy`, `encouraging`, `curious`,
  `error`) and on-screen text for lessons, stories, lyrics, and captions.
- The face is always animating, which hides inference and server-startup
  latency and makes waits feel intentional rather than broken.

### D. Vision & "Look and ..." Capability
- One-shot photo understanding when the user says "look and <task>", e.g.
  "look and solve this math" or "look and read this".
- Camera capture is always local; vision inference runs **inside the LLM
  server** (on-device on the Jetson, LAN/cloud on the Pi deployment).
- Photos are processed in memory and are not saved by default.

### E. Family Communication (planned)
- Kids can ask Sparky to call approved family members.
- Calls require an allowlist, parental controls, and encrypted transport; call
  audio is not stored.
- This feature requires network access. The rest of the AI stack remains
  local/offline; the call itself is the only network-dependent function.

### F. Physical Mobility (optional, future)
- Stationary first. If movement is added, a motor/servo HAT handles low-level
  actuation while path planning runs on the main computer.
- Perception acceleration for movement is a separate decision and is not part
  of the current plan.

## 3. Technical Constraints & Principles

### 3.1 Offline-First & Privacy
- Prefer fully local processing.
- The **Jetson deployment is fully offline**: ASR, TTS, and the LLM/VLM server
  all run on-device, with Sparky talking to the server over loopback HTTP.
- The **Pi 5 deployment sends only LLM/VLM requests** to a configured server.
  ASR, TTS, camera capture, display, and audio DSP stay local.
- A **local-network server is the recommended remote target**; cloud use is
  explicit opt-in and must be documented as sending voice/images off-device.
- No voice, photos, or room data is stored by default. Conversation memory
  lives in RAM only and is cleared when the process stops.
- Calls require an allowlist, encryption, and parental controls.

### 3.2 Target Hardware & Deployment

| Role | Hardware | Local work | Remote work | Notes |
|---|---|---|---|---|
| Pi 5 client | Raspberry Pi 5, **8 GB recommended**; 4 GB only for OS Lite + pygame + no navigation | Whisper ASR, Piper TTS, camera, face/text display, audio DSP, navigation/motor control | LLM/VLM over HTTP to a LAN or cloud server | No local LLM and **no local fallback**; network required to answer |
| Standalone brain | Jetson Orin Nano Super, 8 GB | Everything, including the local LLM server | None | Fully offline; llama.cpp server on `127.0.0.1` |

Shared requirements and decisions:
- **LLM server**: a version-pinned OpenAI-compatible HTTP server. `llama.cpp
  server` is the primary choice on the Jetson and for a local LAN server;
  Ollama is an optional dev/LAN convenience. The robot app contains **no
  in-process LLM**.
- **Storage**: 32 GB+ NVMe SSD via PCIe HAT, or USB 3.0 SSD. MicroSD only for
  bootstrap. The Pi needs local Whisper + Piper assets; the Jetson needs those
  **and** the server's GGUF model and mmproj projector.
- **Network (Pi deployment)**: wired Ethernet or strong Wi-Fi; the server is
  on the critical path. The server binds to loopback on the Jetson; any
  non-loopback bind requires an API key and TLS. For a self-signed LAN
  certificate, set `ROBOT_REMOTE_CA_FILE` on the client or terminate TLS at a
  reverse proxy. Ollama's auth is limited, so prefer llama.cpp server beyond
  loopback.
- **Cooling**: active cooling for sustained inference.
- **Audio**: USB microphone and USB speaker/sound card.
- **Camera**: CSI camera via `picamera2` on the Pi (OpenCV/V4L2 fallback).
- **Display**: HDMI or SPI/I²C screen for the face and teaching text.
- **Expansion**: GPIO/PCA9685 servos and a motor HAT if movement is added.
- **Not used** (after migration): AI HAT+ / Hailo accelerators, per-device
  config profiles, an in-process LLM in the robot app, or a `local`/`remote`
  backend switch. The in-process path exists only until the remote vision gate
  passes.

### 3.3 Memory Budgets

**Pi 5 client (ASR/TTS local, LLM remote):**

| Component | Approx. RAM |
|---|---|
| Raspberry Pi OS Lite + system | 0.5–0.8 GB |
| Python + dependencies | 0.3–0.5 GB |
| Whisper tiny.en (int8) | ~0.3 GB |
| Piper voice + audio buffers | 0.15–0.2 GB |
| Face/UI (pygame) | 0.1–0.2 GB |
| Camera capture buffers | 0.1–0.2 GB |
| Audio DSP (pitch/AEC) | 0.1–0.2 GB |
| Headroom | 0.3–0.5 GB |
| **Typical total** | **~1.8–2.8 GB** |

Additions: navigation stack 0.3–0.8 GB; browser kiosk UI 0.3–0.5 GB; full
desktop 0.5–0.8 GB. Worst case approaches 4 GB, so 4 GB is viable only for the
minimal configuration (OS Lite, pygame, no navigation); 8 GB is recommended.

**Jetson Orin Nano Super (fully local, two processes):**

| Component | Approx. RAM |
|---|---|
| JetPack/Ubuntu + system | 1.0–1.5 GB |
| Whisper tiny.en (int8) | ~0.3 GB |
| Piper voice | ~0.15 GB |
| LLM **server** process, 4B Q4_K_M | 2.74 GB |
| Server KV cache (n_ctx 4096) | 0.4–0.6 GB |
| Vision projector (mmproj) | ~0.68 GB |
| App process + HTTP client | 0.2–0.4 GB |
| CUDA buffers + headroom | ~1 GB |
| **Typical total** | **~6.5–7 GB** |

### 3.4 Performance Targets

- **Pi 5 client**: no local LLM decode target. Targets instead:
  - ASR: under ~1 s for a typical 3 s utterance.
  - TTS: audio starts within ~500 ms of receiving the first sentence.
  - Time-to-first-audio: under ~3 s on a LAN server, measured from the end of
    the user's utterance and including ASR.
  - Vision turnaround: under ~8 s on a LAN server.
  - Server startup: cold readiness under ~30 s; show a `waking` face state
    while the server loads.
  - LAN figures assume a server at least as fast as the Jetson; record the
    actual server class with benchmark results.
- **Jetson**: 4B Q4_K_M at ~20–30 tokens/s; 2B at ~40–60 tokens/s. Measure
  HTTP/SSE overhead rather than assuming it is negligible.
- **Both**: stream tokens and speak per sentence; cap spoken replies at
  ~64–96 tokens and put longer explanations on the screen.
- Benchmark every change on the target board, not only on the development Mac.

### 3.5 Core Tech Stack

- **STT**: `faster-whisper` tiny.en with int8, run **locally** on both
  deployments (`base.en` optional for child-speech accuracy).
- **TTS**: `piper-tts` en_US-lessac-medium, run **locally** on both
  deployments, with pitch shift and robot DSP.
- **LLM**: an OpenAI-compatible HTTP server on every deployment. `llama.cpp
  server` runs on the Jetson over loopback with CUDA offload, and on the LAN
  server for the Pi. The robot app uses only an HTTP client.
- **Vision**: VLM plus mmproj projector loaded by the LLM server, so vision
  works identically in both deployments.
- **Audio I/O**: `sounddevice` + `numpy`; 16 kHz capture, 22.05 kHz playback.
- **Configuration**: only the LLM server URL, model name, and credentials
  differ between deployments. No device profiles and no backend switch in the
  app.

### 3.6 Model Selection Guidance
- Default to the smallest model that meets the current capability need.
- **Jetson server**: `qwen35-2b` by default; `qwen35-4b` for the best tutoring
  quality when its slower replies are acceptable. Both support vision via
  mmproj.
- **Pi 5**: no local language model. The server-side model is chosen on the
  server; the Pi only needs its name and endpoint.
- `download_models.py` provisions **client assets** (Whisper + Piper) and
  **server assets** (GGUF + mmproj). The Pi needs client assets only; the
  Jetson needs both, because it is both client and server.
- Keep spoken answers short; use the screen for detailed steps and stories.

### 3.7 The LLM Server as a Component
- The LLM server is a first-class part of the system, not an implementation
  detail of the app: it has its own build, version pin, configuration, health
  endpoint, warm-up request, restart policy, and logs.
- The robot app waits for server readiness, handles restarts, and degrades
  gracefully. It never loads a model itself.
- Settings prefixed `ROBOT_REMOTE_*` describe the HTTP LLM server even when it
  runs on loopback; "remote" means "over HTTP," not necessarily "off-device."

## 4. Architectural Guidelines

- **Short & Conversational Speech**: 1–3 sentences for casual chat;
  step-by-step but concise for teaching. Longer structured content goes on
  screen.
- **Multi-Turn Context**: Keep recent conversation turns in RAM so lessons and
  interactions feel coherent. Nothing is persisted.
- **Extensible Skill Modules**: Route intents to pluggable skills (chat, math,
  piano, singing, reading, vision, calls, movement) through a priority-aware
  event bus. Each new function is added as a skill, not by editing the main
  loop.
- **Local Real-Time Paths**: Piano, singing, and call audio processing must
  stay local and must never depend on the network.
- **Deterministic Safety**: Obstacle/safety logic and call controls are
  deterministic and independent of the LLM.
- **Graceful Degradation**: If the LLM server is unreachable or still loading,
  Sparky says so clearly. No local substitute model is loaded.
- **One Codebase, Two Deployments**: The same app runs on the Pi client and
  the Jetson. Only the LLM server URL, model name, and credentials differ; on
  Jetson that server simply runs on the same machine.

For module-level details, data flows, and the server lifecycle, see
[`ARCHITECTURE.md`](ARCHITECTURE.md). Work proceeds from the current code and
hands-on (user-acceptance) testing: implement the happy path first, then update
these documents only as real gaps surface.
