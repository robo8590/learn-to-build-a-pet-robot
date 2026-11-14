#!/usr/bin/env python3
"""
download_models.py
Automated Python script to download local offline models for Sparky Robot:
1. Faster-Whisper (tiny) STT
2. LLaMA 3.2 1B Instruct (GGUF Q4_K_M)
3. Piper TTS Voices (en_US lessac & vi_VN vais1000, csa-voice-v3 5-speaker, 25hours_single)

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
WHISPER_DIR = MODELS_DIR / "whisper"
WHISPER_EN_DIR = WHISPER_DIR / "en"
WHISPER_VI_DIR = WHISPER_DIR / "vi"


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_EN_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_VI_DIR.mkdir(parents=True, exist_ok=True)


def download_whisper(language="vi"):
    """
    Downloads dedicated, language-optimized Whisper models:
      - 'vi': PhoWhisper-base (fine-tuned by VinAI on 844h Vietnamese speech, CTranslate2 format)
      - 'en': faster-whisper-tiny.en (English-only tiny model)
    """
    print(f"\n--- [1/3] Checking / Downloading Whisper STT for language: '{language}' ---")

    if language == "vi":
        model_bin = WHISPER_VI_DIR / "model.bin"
        if model_bin.is_file():
            print(f"[✓] Vietnamese PhoWhisper model already exists at: {WHISPER_VI_DIR}")
            return

        print("Downloading PhoWhisper-base (Vietnamese-specialized CT2 model) from Hugging Face...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="quocphu/PhoWhisper-ct2-FasterWhisper",
            allow_patterns=["PhoWhisper-base-ct2-fasterWhisper/*"],
            local_dir=str(WHISPER_DIR),
        )

        # Move files from subfolder into WHISPER_VI_DIR
        subfolder = WHISPER_DIR / "PhoWhisper-base-ct2-fasterWhisper"
        if subfolder.exists():
            for item in subfolder.iterdir():
                dest = WHISPER_VI_DIR / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(subfolder, ignore_errors=True)

        print(f"[✓] Vietnamese Whisper model saved to: {WHISPER_VI_DIR}")

    elif language == "en":
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
        # Vietnamese: vais1000-medium (warm, clear female)
        (
            "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
            "vi_VN-vais1000-medium.onnx",
        ),
        (
            "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json",
            "vi_VN-vais1000-medium.onnx.json",
        ),
        # Vietnamese: csa-voice-piper-v3-medium (5 speakers, high quality, cute child-friendly)
        (
            "vi/vi_VN/csa-voice-piper-v3/medium/vi_VN-csa-voice-piper-v3-medium.onnx",
            "vi_VN-csa-voice-piper-v3-medium.onnx",
        ),
        (
            "vi/vi_VN/csa-voice-piper-v3/medium/vi_VN-csa-voice-piper-v3-medium.onnx.json",
            "vi_VN-csa-voice-piper-v3-medium.onnx.json",
        ),
        # Vietnamese: 25hours_single-low (male voice, lower quality)
        (
            "vi/vi_VN/25hours_single/low/vi_VN-25hours_single-low.onnx",
            "vi_VN-25hours_single-low.onnx",
        ),
        (
            "vi/vi_VN/25hours_single/low/vi_VN-25hours_single-low.onnx.json",
            "vi_VN-25hours_single-low.onnx.json",
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
    import argparse

    parser = argparse.ArgumentParser(description="Download offline models for Sparky Robot.")
    parser.add_argument(
        "--lang",
        choices=["vi", "en", "all"],
        default="vi",
        help="Target language for STT and voice assets (default: vi)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print(" Sparky Robot - Offline Model Downloader (Python)")
    print(f" Target Language: {args.lang.upper()}")
    print("=" * 50)
    ensure_dirs()

    if args.lang in ("vi", "all"):
        download_whisper("vi")
    if args.lang in ("en", "all"):
        download_whisper("en")

    download_llm()
    download_piper_voices()
    print("\n" + "=" * 50)
    print(" All requested models verified and ready for offline use!")
    print("=" * 50)


if __name__ == "__main__":
    main()
