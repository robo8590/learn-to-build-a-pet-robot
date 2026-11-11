#!/usr/bin/env python3
"""
download_models.py
Automated Python script to download local offline models for Sparky Robot:
1. Faster-Whisper (tiny) STT
2. LLaMA 3.2 1B Instruct (GGUF Q4_K_M)
3. Piper TTS Voices (en_US lessac & vi_VN vais1000)

Requirements:
    pip install huggingface_hub faster-whisper
"""

import os
import shutil
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PIPER_DIR = MODELS_DIR / "piper"
WHISPER_DIR = MODELS_DIR / "whisper-tiny"


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)


def download_whisper():
    print("\n--- [1/3] Checking / Downloading Faster-Whisper (tiny) ---")
    model_bin = WHISPER_DIR / "model.bin"
    if model_bin.is_file():
        print(f"[✓] Whisper model already exists at: {WHISPER_DIR}")
        return

    try:
        from faster_whisper import download_model

        print("Downloading Whisper 'tiny' via faster-whisper...")
        download_model("tiny", output_dir=str(WHISPER_DIR))
        print(f"[✓] Whisper model downloaded to {WHISPER_DIR}")
    except ImportError:
        print("[!] faster-whisper not installed. Falling back to huggingface_hub...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Systran/faster-whisper-tiny",
            local_dir=str(WHISPER_DIR),
        )
        print(f"[✓] Whisper model downloaded to {WHISPER_DIR}")


def download_llm():
    print("\n--- [2/3] Checking / Downloading LLaMA 3.2 1B Instruct (GGUF) ---")
    from huggingface_hub import hf_hub_download

    llm_file = MODELS_DIR / "llama-3.2-1b-instruct.Q4_K_M.gguf"
    if llm_file.is_file():
        print(f"[✓] LLM model already exists at: {llm_file}")
        return

    print("Downloading LLaMA 3.2 1B Instruct (Q4_K_M) from Hugging Face...")
    downloaded_path = hf_hub_download(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        local_dir=str(MODELS_DIR),
    )
    # Ensure standard filename matches code expectation
    target_path = MODELS_DIR / "llama-3.2-1b-instruct.Q4_K_M.gguf"
    if Path(downloaded_path) != target_path and not target_path.exists():
        shutil.move(downloaded_path, target_path)

    print(f"[✓] LLM saved to: {target_path}")


def download_piper_voices():
    print("\n--- [3/3] Checking / Downloading Piper TTS Voices (en & vi) ---")
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
        # Vietnamese: vais1000-medium
        (
            "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
            "vi_VN-vais1000-medium.onnx",
        ),
        (
            "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json",
            "vi_VN-vais1000-medium.onnx.json",
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


def main():
    print("=" * 50)
    print(" Sparky Robot - Offline Model Downloader (Python)")
    print("=" * 50)
    ensure_dirs()
    download_whisper()
    download_llm()
    download_piper_voices()
    print("\n" + "=" * 50)
    print(" All models verified and ready for offline use!")
    print("=" * 50)


if __name__ == "__main__":
    main()
