"""Local Piper voice loading and synthesis configuration."""

from piper import PiperVoice
from piper.config import SynthesisConfig

from sparky.config.paths import PIPER_VOICE_PATH

# Piper synthesis configuration (warm, friendly setting)
PIPER_SYNTHESIS_CONFIG = SynthesisConfig(length_scale=0.92, noise_scale=0.667, noise_w_scale=0.8)

_piper_voice = None


def get_piper_voice():
    """Load (and cache) the local English Piper voice."""
    global _piper_voice

    if not PIPER_VOICE_PATH.is_file():
        raise FileNotFoundError(
            f"Local Piper voice model not found at '{PIPER_VOICE_PATH}'. "
            "Run 'python download_models.py' to download required voices."
        )

    if _piper_voice is None:
        print("Loading local Piper voice...")
        _piper_voice = PiperVoice.load(str(PIPER_VOICE_PATH))

    if _piper_voice is None:
        raise RuntimeError("Piper voice failed to load.")
    return _piper_voice
