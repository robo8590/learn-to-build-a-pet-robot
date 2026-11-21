"""Environment-driven runtime settings shared by Sparky modules."""

import os


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to a default."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# Robot Voice Effect Profile:
#   - "cute_companion": pitch shift + subtle robot DSP (default, kid-friendly)
#   - "robot_droid": pitch shift + stronger robot DSP
#   - "natural": pure, unmodified Piper voice (no pitch shift, no DSP)
ROBOT_VOICE_STYLE = os.getenv("ROBOT_VOICE_STYLE", "cute_companion").strip().lower()

# LLM runtime tuning. On a Raspberry Pi 5, set ROBOT_LLM_GPU_LAYERS=0.
LLM_N_GPU_LAYERS = env_int("ROBOT_LLM_GPU_LAYERS", -1)
LLM_N_THREADS = env_int("ROBOT_LLM_THREADS", 4)

# Keep spoken replies short; see the latency targets in PROJECT_MISSION.md.
LLM_MAX_TOKENS = env_int("ROBOT_LLM_MAX_TOKENS", 96)

# In-memory conversation memory: number of recent user/assistant turns to keep
# so Sparky can refer back to earlier parts of the session. Nothing is saved.
LLM_MEMORY_MAX_TURNS = env_int("ROBOT_MEMORY_MAX_TURNS", 8)

# Vision replies (math step-by-step) need more room than casual chat.
VISION_MAX_TOKENS = env_int("ROBOT_VISION_MAX_TOKENS", 256)
