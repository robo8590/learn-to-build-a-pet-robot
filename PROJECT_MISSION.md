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

1. **Fully Offline First**:
   - Must run locally without requiring constant internet access.
   - Preserves family and child privacy (voice and room data stay local).
2. **Target Hardware Deployment**:
   - **Development**: Apple Silicon (Mac M3) with Metal GPU acceleration.
   - **Target Robot Brain**: Raspberry Pi 5 / embedded Linux SBC.
   - Code must maintain portability (e.g., configurable GPU layer offload `n_gpu_layers`).
3. **Core Tech Stack**:
   - **STT**: `faster-whisper` (e.g., `whisper-tiny` with int8 quantization).
   - **LLM**: Local small language models via `llama-cpp-python` (e.g., Llama 3.2 1B / 3B GGUF).
   - **TTS**: `piper-tts` (low-latency ONNX models for English).
   - **Audio I/O**: `sounddevice`, `numpy`.

---

## 4. Architectural Guidelines for Future Updates
- **Short & Conversational Responses**: Keep spoken answers concise (1–3 sentences for casual chat; conversational step-by-step for teaching).
- **Multi-Turn Context**: Maintain memory of recent conversation turns so lessons and interactions feel coherent.
- **Extensible Skill Modules**: Structure functionality into pluggable modes (e.g., `chat_mode`, `math_tutor_mode`, `piano_tutor_mode`, `movement_mode`).
