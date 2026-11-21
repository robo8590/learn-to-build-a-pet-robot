"""Local filesystem paths used by Sparky.

The project root is two levels above this file:

    <project>/sparky/config/paths.py
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_DIR
MODELS_DIR = BASE_DIR / "models"

WHISPER_MODEL_PATH = MODELS_DIR / "whisper" / "en"
PIPER_VOICE_PATH = MODELS_DIR / "piper" / "en_US-lessac-medium.onnx"
