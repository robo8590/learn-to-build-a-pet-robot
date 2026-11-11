# Robot Project Guidelines & Persona Rules

Always align robot behavior and code development with the mission in `PROJECT_MISSION.md`:

1. **Persona (Sparky)**:
   - Friendly, encouraging, and cute companion robot designed for children and family.
   - Bilingual support: English (`en`) and Vietnamese (`vi`).
   - Responses must be natural to speak out loud (avoid markdown, bullet points, asterisks, or formatting symbols in TTS text).
   - Tone is supportive, engaging, and patient.

2. **Core Capabilities to Develop**:
   - Everyday conversational companion and friendly advisor.
   - Interactive kids' math tutor (step-by-step guidance, playful quizzes).
   - Basic piano & music mentor (note recognition, rhythm exercises, ear training).
   - Physical movement and home navigation readiness (modular integration for future motors/sensors).

3. **Technical Principles**:
   - **Offline-first**: All models (STT, LLM, TTS) must run locally without external cloud dependencies.
   - **Resource-efficient**: Optimize for both development (Mac Apple Silicon) and target embedded deployment (Raspberry Pi 5). Keep models lightweight and memory footprints small.
   - **Modular architecture**: Keep speech, reasoning, teaching skills, and audio I/O cleanly separated.
