# Project Mission: Sparky — The Personal Companion & Tutor Robot

## 1. Vision & Purpose
The mission of this project is to learn and build a personal, cute, and helpful companion robot ("Sparky") capable of navigating around the home, chatting with family members, offering thoughtful advice, and tutoring kids in subjects like math and piano.

---

## 2. Core Pillars & Roles

### A. Personality & Companion
- **Tone & Persona**: Warm, encouraging, playful, gentle, and child-friendly.
- **Safety**: Safe language, positive reinforcement, no harmful or scary responses.
- **Language Support**: English interaction.

### B. Educational Tutor for Kids
- **Math Mentorship**:
  - Breaks down math problems into fun, bite-sized steps.
  - Uses storytelling, analogies, and quizzes suitable for children.
- **Music & Piano Guidance**:
  - Teaches music fundamentals (notes, rhythm, keys, basic ear training).
  - Encourages practice habits with interactive games and positive feedback.
- **General Advice**:
  - Offers gentle life habits, curiosity prompts, and daily encouragement.

### C. Physical Mobility & Presence (Future Hardware Roadmap)
- Navigate and explore home environments safely (obstacle avoidance, patrolling).
- Expressive cues (audio chimes, cute voice inflections, screen/face expressions).

---

## 3. Technical Constraints & Principles

### 3.1 Fully Offline First
- Must run locally without requiring constant internet access.
- Preserves family and child privacy (voice and room data stay local).

### 3.2 Target Hardware & Deployment

| Role | Hardware | Notes |
|---|---|---|
| Development | Apple Silicon Mac (M3), 16 GB+ | Metal GPU offload via llama.cpp (`n_gpu_layers=-1`) for fast iteration. |
| Target robot brain | Raspberry Pi 5, 8 GB RAM | CPU-only inference, 64-bit Raspberry Pi OS (Bookworm/Trixie). |
| Storage | 32 GB+ NVMe SSD via PCIe HAT, or USB 3.0 SSD | microSD only for bootstrap; each model is 1–3 GB. |
| Power | Official 27 W USB-C PD supply (5 V / 5 A) | Avoids undervoltage throttling when USB peripherals are attached. |
| Cooling | Active cooler | Sustained inference reaches the 80–85 °C throttle range without it. |
| Audio input | USB microphone | Pi 5 has no 3.5 mm jack; an I2S mic HAT is a planned upgrade. |
| Audio output | USB speaker or USB sound card | Alternatively HDMI audio; a dedicated output device keeps Piper latency low. |
| Future expansion | CSI camera, GPIO/PCA9685 servos, optional AI HAT+ | The AI HAT+ does not help LLM decode (slower than the CPU) but can accelerate vision. |

**Memory budget (Raspberry Pi 5, 8 GB):**

| Component | Approx. RAM |
|---|---|
| Raspberry Pi OS + desktop | 0.8–1.2 GB |
| Whisper tiny.en (int8) | ~0.3 GB |
| Piper voice | ~0.15 GB |
| LLM, 1B–2B Q4_K_M | 1.4–2.2 GB |
| Headroom (KV cache, sensors, camera) | ≥ 3 GB |

**Portability rule:** the same code must run on both profiles. Only `ROBOT_LLM_GPU_LAYERS`, `ROBOT_LLM_THREADS`, and the selected model key should change between Mac and Pi.

**Performance targets:**
- 1B–2B Q4_K_M models at 6–15 tokens/second decode on the Pi 5.
- Short spoken replies: cap generation at ~64–96 tokens and stream TTS per sentence.
- Dense models above ~4B are out of scope for the Pi target (≤ 5 tokens/second).

### 3.3 Core Tech Stack
- **STT**: `faster-whisper` tiny.en with int8 quantization (`base.en` optional for better child-speech accuracy).
- **LLM**: local GGUF models through `llama-cpp-python`. Default **Qwen3.5 2B Q4_K_M**; quality option **Gemma 4 E2B IT (QAT)**; fallback **Llama 3.2 1B**. Selected with `ROBOT_LLM_MODEL` or `--model`; the registry lives in `model_config.py`.
- **TTS**: `piper-tts` en_US-lessac-medium, with pitch shift and robot DSP.
- **Audio I/O**: `sounddevice` + `numpy`; 16 kHz capture, 22.05 kHz playback.

### 3.4 Model Selection Guidance
- Default to the smallest model that meets the current capability need.
- Prefer 1B–2B Q4_K_M on the Pi; use 3B–4B only when a slower reply is acceptable or on the Mac.
- Benchmark every model change on the target board, not only on the Mac.

---

## 4. Architectural Guidelines for Future Updates
- **Short & Conversational Responses**: Keep spoken answers concise (1–3 sentences for casual chat; conversational step-by-step for teaching).
- **Multi-Turn Context**: Maintain memory of recent conversation turns so lessons and interactions feel coherent.
- **Extensible Skill Modules**: Structure functionality into pluggable modes (e.g., `chat_mode`, `math_tutor_mode`, `piano_tutor_mode`, `movement_mode`).
