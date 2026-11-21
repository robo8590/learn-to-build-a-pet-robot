#!/usr/bin/env python3
"""
download_models.py
Download the local offline models used by Sparky:

1. Faster-Whisper (tiny.en) speech-to-text
2. A local GGUF LLM (default: Qwen3.5 2B, see sparky/config/llm_registry.py)
3. Piper TTS voice (en_US lessac)
4. Vision projector (mmproj) for vision-capable LLMs

By default the default LLM and its mmproj are downloaded. Use ``--llm`` to
fetch another model, or ``--llm all`` to fetch every registered model. mmproj
projectors are downloaded automatically alongside any vision-capable LLM.

Examples:
    python download_models.py
    python download_models.py --list-llms
    python download_models.py --llm gemma4-e2b
    python download_models.py --llm all

Requirements:
    pip install huggingface_hub faster-whisper
"""

import argparse
import shutil
from pathlib import Path

from sparky.config.llm_registry import (
    DEFAULT_LLM_MODEL,
    available_llm_keys,
    get_llm_config,
    is_llm_downloaded,
    is_mmproj_downloaded,
    is_vision_capable,
    llm_path,
    mmproj_path,
    resolve_llm_key,
)
from sparky.config.paths import MODELS_DIR, PIPER_VOICE_PATH, WHISPER_MODEL_PATH

PIPER_DIR = PIPER_VOICE_PATH.parent
WHISPER_EN_DIR = WHISPER_MODEL_PATH


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download the offline models used by Sparky.",
    )
    parser.add_argument(
        "--llm",
        metavar="KEY",
        help=(
            "LLM model key to download, or 'all' for every registered model. "
            f"Default: {DEFAULT_LLM_MODEL}."
        ),
    )
    parser.add_argument(
        "--list-llms",
        action="store_true",
        help="List available LLM models and exit.",
    )
    parser.add_argument(
        "--mmproj",
        metavar="KEY",
        help=(
            "Download the vision projector (mmproj) for a vision-capable model "
            "KEY, or 'all' for every vision-capable model. Required for the "
            "'look and ...' photo feature."
        ),
    )
    return parser.parse_args(argv)


def print_llms():
    print("Available LLM models:")
    for key in available_llm_keys():
        config = get_llm_config(key)
        marker = " *default" if key == DEFAULT_LLM_MODEL else ""
        if config.get("supports_vision", False):
            vision = " [vision]"
            if not is_mmproj_downloaded(key):
                vision += " (mmproj missing)"
        else:
            vision = " [text-only]"
        status = "downloaded" if is_llm_downloaded(key) else "not downloaded"
        print(f"  {key}{marker}{vision}")
        print(
            f"    {config['label']} | {config['size_gb']:.2f} GB | "
            f"{status} | {config['notes']}"
        )
    print("\nDownload with: python download_models.py --llm KEY")
    print("Download a vision projector with: python download_models.py --mmproj KEY")


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_EN_DIR.mkdir(parents=True, exist_ok=True)


def download_whisper():
    """
    Downloads the English-only faster-whisper tiny model.
    """
    print("\n--- Checking / Downloading Whisper STT (English) ---")

    model_bin = WHISPER_EN_DIR / "model.bin"
    if model_bin.is_file():
        print(f"[✓] English Whisper model already exists at: {WHISPER_EN_DIR}")
        return

    print("Downloading Whisper 'tiny.en' (English dedicated) via faster-whisper...")
    try:
        from faster_whisper import download_model
        download_model("tiny.en", output_dir=str(WHISPER_EN_DIR))
    except Exception:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Systran/faster-whisper-tiny.en",
            local_dir=str(WHISPER_EN_DIR),
        )
    print(f"[✓] English Whisper model saved to: {WHISPER_EN_DIR}")


def download_llm(model_key, with_mmproj=True):
    """Download a single registered LLM by key."""
    config = get_llm_config(model_key)
    target_path = llm_path(model_key)

    if target_path.is_file():
        print(f"[✓] {config['label']} already exists at: {target_path}")
    else:
        from huggingface_hub import hf_hub_download

        print(f"Downloading {config['label']} ({config['size_gb']:.2f} GB)...")
        downloaded = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["hf_filename"],
            local_dir=str(MODELS_DIR),
        )

        # Ensure the local filename matches the registry, regardless of how
        # huggingface_hub laid out the file.
        downloaded_path = Path(downloaded)
        if downloaded_path != target_path:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded_path), str(target_path))

        print(f"[✓] Saved: {target_path}")

    if with_mmproj and config.get("supports_vision", False):
        download_mmproj(model_key)


def download_mmproj(model_key):
    """Download the vision projector (mmproj) for a vision-capable model."""
    config = get_llm_config(model_key)

    if not config.get("supports_vision", False):
        print(
            f"[!] {config['label']} ({model_key}) is not vision-capable; "
            "no mmproj projector needed."
        )
        return

    target_path = mmproj_path(model_key)

    if target_path.is_file():
        print(f"[✓] mmproj for {config['label']} already exists at: {target_path}")
        return

    from huggingface_hub import hf_hub_download

    print(
        f"Downloading mmproj vision projector for {config['label']} "
        f"({config.get('mmproj_size_gb', 0):.2f} GB)..."
    )
    downloaded = hf_hub_download(
        repo_id=config["mmproj_repo_id"],
        filename=config["mmproj_filename"],
        local_dir=str(MODELS_DIR),
    )

    downloaded_path = Path(downloaded)
    if downloaded_path != target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded_path), str(target_path))

    print(f"[✓] Saved: {target_path}")


def download_piper_voices():
    print("\n--- Checking / Downloading Piper TTS Voice (English) ---")
    from huggingface_hub import hf_hub_download

    voices = [
        # English: lessac-medium
        (
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "en_US-lessac-medium.onnx",
        ),
        (
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            "en_US-lessac-medium.onnx.json",
        ),
    ]

    for hf_subpath, local_filename in voices:
        dest_file = PIPER_DIR / local_filename
        if dest_file.is_file():
            print(f"[✓] Piper voice file exists: {local_filename}")
            continue

        print(f"Downloading {local_filename}...")
        downloaded = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=hf_subpath,
            local_dir=str(PIPER_DIR),
        )
        # Move out of nested directory if huggingface_hub recreated full subpath
        downloaded_path = Path(downloaded)
        if downloaded_path != dest_file:
            shutil.move(str(downloaded_path), str(dest_file))
            # Clean up empty parent directories left behind
            parent = downloaded_path.parent
            while parent != PIPER_DIR and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        print(f"[✓] Saved: {dest_file}")


def main(argv=None):
    args = parse_args(argv)

    if args.list_llms:
        print_llms()
        return

    try:
        if args.llm and args.llm.strip().lower() == "all":
            llm_keys = available_llm_keys()
        else:
            llm_keys = [resolve_llm_key(args.llm)]
    except KeyError as e:
        raise SystemExit(str(e))

    print("=" * 50)
    print(" Sparky Robot - Offline Model Downloader")
    print("=" * 50)
    ensure_dirs()

    download_whisper()

    print("\n--- Checking / Downloading local LLM(s) ---")
    for key in llm_keys:
        download_llm(key)

    print("\n--- Checking / Downloading vision projector(s) (mmproj) ---")
    if args.mmproj and args.mmproj.strip().lower() == "all":
        mmproj_keys = [k for k in available_llm_keys() if is_vision_capable(k)]
    elif args.mmproj:
        mmproj_keys = [resolve_llm_key(args.mmproj)]
    else:
        mmproj_keys = []
    for key in mmproj_keys:
        download_mmproj(key)

    download_piper_voices()

    print("\n" + "=" * 50)
    print(" All requested models verified and ready for offline use!")
    print("=" * 50)


if __name__ == "__main__":
    main()
