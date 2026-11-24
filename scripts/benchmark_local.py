#!/usr/bin/env python3
"""Benchmark Sparky's current in-process llama.cpp path.

This records the pre-refactor baseline used by TASKS.md A1. It calls the real
application functions (``initialize_llm``, ``generate_response``,
``generate_vision_response``) so the numbers reflect actual chat and vision
turns, including the vision projector load.

Example:
    python3 scripts/benchmark_local.py \
        --model qwen35-2b --model qwen35-4b \
        --runs 3 --json docs/baseline.json

Results are printed as a Markdown table. A JSON artifact is written when
``--json`` is supplied so later HTTP/SSE comparisons (A10.4/A11.2) can be
diffed mechanically.
"""

import argparse
import gc
import glob
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sparky.llm.conversation import Conversation  # noqa: E402
from sparky.llm.loader import initialize_llm  # noqa: E402
from sparky.llm.text_llm import generate_response  # noqa: E402
from sparky.llm.vision_llm import generate_vision_response  # noqa: E402

CHAT_PROMPT = "What is two plus two? Answer in one short sentence."
VISION_PROMPT = "What is in this photo? Answer in one short sentence."

def synthetic_jpeg():
    """Return a small grey JPEG, used only when no camera capture exists."""
    import cv2
    import numpy as np

    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise SystemExit("Could not encode the synthetic benchmark image.")
    return buffer.tobytes()


class TimedLLM:
    """Transparent proxy that records one ``create_chat_completion`` call."""

    def __init__(self, llm):
        self._llm = llm
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def create_chat_completion(self, *args, **kwargs):
        started = time.perf_counter()
        response = self._llm.create_chat_completion(*args, **kwargs)
        elapsed = time.perf_counter() - started
        usage = response.get("usage") or {}
        self.calls.append(
            {
                "elapsed_seconds": elapsed,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "content": response["choices"][0]["message"]["content"],
            }
        )
        return response


def peak_rss_mb():
    """Return the process's peak resident set size in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def jpeg_dimensions(data):
    """Read width/height from a JPEG header without external dependencies."""
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        length = int.from_bytes(data[index : index + 2], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            return width, height
        index += length
    return None, None


def load_image(path):
    """Load a benchmark photo and describe it for the report."""
    if path:
        image_path = Path(path)
        if not image_path.is_file():
            raise SystemExit(f"Image not found: {image_path}")
    else:
        candidates = sorted(
            glob.glob(str(ROOT / "captures" / "*.jpg")),
            key=os.path.getmtime,
        )
        if candidates:
            image_path = Path(candidates[-1])
        else:
            image_path = None
            raw = synthetic_jpeg()
            width, height = jpeg_dimensions(raw)
            return {
                "path": None,
                "source": "synthetic fallback",
                "bytes": len(raw),
                "width": width,
                "height": height,
                "data": raw,
            }

    raw = image_path.read_bytes()
    width, height = jpeg_dimensions(raw)
    return {
        "path": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
        "source": "captures/",
        "bytes": len(raw),
        "width": width,
        "height": height,
        "data": raw,
    }


def summarise(runs):
    elapsed = [run["elapsed_seconds"] for run in runs]
    summary = {
        "runs": len(runs),
        "elapsed_seconds": {
            "median": statistics.median(elapsed),
            "min": min(elapsed),
            "max": max(elapsed),
        },
    }
    token_counts = [run["completion_tokens"] for run in runs if run["completion_tokens"]]
    if token_counts:
        summary["completion_tokens_median"] = statistics.median(token_counts)
        summary["tokens_per_second"] = (
            summary["completion_tokens_median"] / summary["elapsed_seconds"]["median"]
        )
    return summary


def run_chat(llm, runs):
    timed = TimedLLM(llm)
    cold_llm = TimedLLM(llm)
    generate_response(cold_llm, CHAT_PROMPT, Conversation())
    cold = cold_llm.calls[-1]["elapsed_seconds"]

    for _ in range(runs):
        # Fresh conversation per run so history length stays constant.
        generate_response(timed, CHAT_PROMPT, Conversation())

    summary = summarise(timed.calls)
    summary["cold_seconds"] = cold
    summary["replies"] = [call["content"].strip() for call in timed.calls]
    summary["completion_tokens"] = [
        call["completion_tokens"] for call in timed.calls
    ]
    return summary


def run_vision(llm, image_bytes, runs):
    timed = TimedLLM(llm)
    cold_llm = TimedLLM(llm)
    generate_vision_response(cold_llm, image_bytes, VISION_PROMPT, Conversation())
    cold = cold_llm.calls[-1]["elapsed_seconds"]

    for _ in range(runs):
        generate_vision_response(timed, image_bytes, VISION_PROMPT, Conversation())

    summary = summarise(timed.calls)
    summary["cold_seconds"] = cold
    summary["replies"] = [call["content"].strip() for call in timed.calls]
    summary["completion_tokens"] = [
        call["completion_tokens"] for call in timed.calls
    ]
    return summary


def environment():
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    try:
        import llama_cpp

        llama_cpp_version = getattr(llama_cpp, "__version__", "unknown")
    except Exception:  # pragma: no cover - informational only
        llama_cpp_version = "unknown"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit or None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "llama_cpp": llama_cpp_version,
    }


def benchmark_model(model_key, runs, image):
    print(f"\n=== {model_key} ===")
    gc.collect()

    load_started = time.perf_counter()
    llm, vision_enabled = initialize_llm(model_key)
    load_seconds = time.perf_counter() - load_started
    rss_after_load = peak_rss_mb()
    print(f"loaded in {load_seconds:.2f}s (vision={vision_enabled})")

    result = {
        "key": model_key,
        "load_seconds": load_seconds,
        "vision_enabled": vision_enabled,
        "peak_rss_after_load_mb": rss_after_load,
    }

    result["chat"] = run_chat(llm, runs)
    result["chat"]["peak_rss_mb"] = peak_rss_mb()

    if vision_enabled:
        result["vision"] = run_vision(llm, image["data"], runs)
        result["vision"]["peak_rss_mb"] = peak_rss_mb()
    else:
        result["vision"] = None

    result["peak_rss_mb"] = peak_rss_mb()
    del llm
    gc.collect()
    return result


def print_markdown(payload):
    image = payload["image"]
    if image["path"]:
        source = image["path"]
    else:
        source = image["source"]
    print("\n## Results\n")
    print(f"Image: `{source}` ({image['width']}x{image['height']}, {image['bytes']} bytes)\n")
    print("| Model | Load (s) | Chat median (s) | Chat tok/s | Vision median (s) | Vision tok/s | Peak RSS (MiB) |")
    print("|---|---|---|---|---|---|---|")
    for model in payload["models"]:
        chat = model["chat"]
        chat_tps = tokens_per_second(chat)
        vision = model.get("vision")
        if vision:
            vision_median = f"{vision['elapsed_seconds']['median']:.2f}"
            vision_tps = f"{tokens_per_second(vision):.1f}"
        else:
            vision_median = "n/a"
            vision_tps = "n/a"
        print(
            f"| {model['key']} | {model['load_seconds']:.2f} | "
            f"{chat['elapsed_seconds']['median']:.2f} | {chat_tps:.1f} | "
            f"{vision_median} | {vision_tps} | {model['peak_rss_mb']:.0f} |"
        )


def tokens_per_second(summary):
    return summary.get("tokens_per_second", 0.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model key to benchmark (repeatable). Default: qwen35-2b.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per turn.")
    parser.add_argument("--image", help="JPEG path. Default: newest captures/*.jpg.")
    parser.add_argument("--json", help="Write a JSON artifact to this path.")
    args = parser.parse_args(argv)

    models = args.models or ["qwen35-2b"]
    image = load_image(args.image)
    payload = {
        "environment": environment(),
        "image": {key: value for key, value in image.items() if key != "data"},
        "prompts": {"chat": CHAT_PROMPT, "vision": VISION_PROMPT},
        "models": [],
    }

    for model_key in models:
        payload["models"].append(benchmark_model(model_key, args.runs, image))

    print_markdown(payload)

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
