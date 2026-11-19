"""Shared paths and local LLM registry for Sparky.

Both ``download_models.py`` and ``offline_robot.py`` import this module so that
model keys, file names, and download sources stay in sync.

Add a new model by appending an entry to ``LLM_MODELS``; no other code changes
are required.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

WHISPER_MODEL_PATH = MODELS_DIR / "whisper" / "en"
PIPER_VOICE_PATH = MODELS_DIR / "piper" / "en_US-lessac-medium.onnx"

DEFAULT_LLM_MODEL = "qwen35-2b"

# Each entry describes one GGUF model that can be downloaded and loaded locally.
#   label           Human-readable name shown in the UI and docs.
#   repo_id         Hugging Face repository that hosts the GGUF.
#   hf_filename     File name inside the repository.
#   local_filename  File name stored under models/.
#   n_ctx           Context window to allocate at load time.
#   size_gb         Approximate download size, for display only.
#   notes           Short guidance used by --list-models / --list-llms.
#   supports_vision Whether the model can process images (multimodal).
LLM_MODELS = {
    "qwen35-2b": {
        "label": "Qwen3.5 2B Instruct",
        "repo_id": "unsloth/Qwen3.5-2B-GGUF",
        "hf_filename": "Qwen3.5-2B-Q4_K_M.gguf",
        "local_filename": "qwen3.5-2b-instruct.Q4_K_M.gguf",
        "n_ctx": 4096,
        "size_gb": 1.28,
        "notes": "Default. Best speed/quality balance for Raspberry Pi 5.",
        "supports_vision": False,
    },
    "gemma4-e2b": {
        "label": "Gemma 4 E2B IT (QAT)",
        "repo_id": "unsloth/gemma-4-E2B-it-qat-GGUF",
        "hf_filename": "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        "local_filename": "gemma-4-e2b-it.qat.UD-Q4_K_XL.gguf",
        "n_ctx": 4096,
        "size_gb": 2.62,
        "notes": "Highest quality in the Pi tier; also supports audio and vision.",
        "supports_vision": True,
        # CLIP vision projector (mmproj) required by llama.cpp for image input.
        "mmproj_repo_id": "unsloth/gemma-4-E2B-it-qat-GGUF",
        "mmproj_filename": "mmproj-F32.gguf",
        "mmproj_local_filename": "gemma-4-e2b-mmproj-F32.gguf",
    },
    "qwen35-4b": {
        "label": "Qwen3.5 4B Instruct",
        "repo_id": "unsloth/Qwen3.5-4B-GGUF",
        "hf_filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "local_filename": "qwen3.5-4b-instruct.Q4_K_M.gguf",
        "n_ctx": 4096,
        "size_gb": 2.74,
        "notes": "Best tutoring quality, but slow on a Pi; ideal on a Mac GPU.",
        "supports_vision": True,
        # CLIP vision projector (mmproj) required by llama.cpp for image input.
        "mmproj_repo_id": "unsloth/Qwen3.5-4B-GGUF",
        "mmproj_filename": "mmproj-F16.gguf",
        "mmproj_local_filename": "qwen3.5-4b-mmproj-F16.gguf",
    },
    "qwen25-1.5b": {
        "label": "Qwen2.5 1.5B Instruct",
        "repo_id": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "hf_filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "local_filename": "qwen2.5-1.5b-instruct.Q4_K_M.gguf",
        "n_ctx": 2048,
        "size_gb": 1.00,
        "notes": "Fast, proven fallback with strong math for its size.",
        "supports_vision": False,
    },
    "llama32-1b": {
        "label": "Llama 3.2 1B Instruct",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "hf_filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "local_filename": "llama-3.2-1b-instruct.Q4_K_M.gguf",
        "n_ctx": 2048,
        "size_gb": 0.81,
        "notes": "Lowest latency; weakest at math. Kept as a fallback.",
        "supports_vision": False,
    },
}


def is_vision_capable(key=None):
    """Return True if the resolved LLM model supports vision (image input)."""
    return get_llm_config(key).get("supports_vision", False)


def available_llm_keys():
    """Return the registry keys in display order."""
    return list(LLM_MODELS.keys())


def resolve_llm_key(key=None):
    """Normalize and validate a model key.

    ``None`` or an empty string resolves to ``DEFAULT_LLM_MODEL``.
    """
    if not key or not str(key).strip():
        return DEFAULT_LLM_MODEL

    normalized = str(key).strip().lower()
    if normalized not in LLM_MODELS:
        valid = ", ".join(available_llm_keys())
        raise KeyError(
            f"Unknown LLM model '{key}'. Available models: {valid}"
        )
    return normalized


def get_llm_config(key=None):
    """Return the registry entry for ``key`` (or the default model)."""
    return LLM_MODELS[resolve_llm_key(key)]


def llm_path(key=None):
    """Return the local GGUF path for ``key`` (or the default model)."""
    return MODELS_DIR / get_llm_config(key)["local_filename"]


def is_llm_downloaded(key=None):
    return llm_path(key).is_file()


def mmproj_path(key=None):
    """Return the local mmproj (vision projector) path for ``key``.

    Only meaningful for vision-capable models; returns ``None`` otherwise.
    """
    config = get_llm_config(key)
    if not config.get("supports_vision", False):
        return None
    return MODELS_DIR / config["mmproj_local_filename"]


def is_mmproj_downloaded(key=None):
    """Return True if the mmproj projector file exists for ``key``."""
    path = mmproj_path(key)
    return path is not None and path.is_file()


def is_vision_ready(key=None):
    """Return True if the resolved model supports vision *and* its mmproj
    projector file is downloaded, so image input is actually usable."""
    return is_vision_capable(key) and is_mmproj_downloaded(key)


def resolve_llm_model_key(cli_value=None, env_value=None):
    """Resolve the active model: CLI flag, then env var, then default."""
    if cli_value:
        return resolve_llm_key(cli_value)
    if env_value:
        return resolve_llm_key(env_value)
    return resolve_llm_key(DEFAULT_LLM_MODEL)
