"""Command-line interface for Sparky."""

import argparse

from sparky.config.llm_registry import (
    DEFAULT_LLM_MODEL,
    available_llm_keys,
    get_llm_config,
    is_llm_downloaded,
    is_mmproj_downloaded,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sparky - offline voice companion and tutor robot.",
    )
    parser.add_argument(
        "--model",
        metavar="KEY",
        help="LLM model to load (overrides the ROBOT_LLM_MODEL env var).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available LLM models and exit.",
    )
    parser.add_argument(
        "--look",
        metavar="TASK",
        help=(
            "One-shot vision mode: take a photo, ask the vision LLM TASK, "
            "speak the answer, and exit. Requires a vision-capable model."
        ),
    )
    parser.add_argument(
        "--check-vision",
        action="store_true",
        help=(
            "Load the selected model, run one tiny in-memory image through the "
            "vision projector, print the result, and exit."
        ),
    )
    return parser.parse_args(argv)


def print_models_table():
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
    print("\nDownload a model with: python download_models.py --llm KEY")
    print("Download a vision projector with: python download_models.py --mmproj KEY")
