# Pre-refactor baseline (A1)

Baseline for the HTTP-only LLM migration. It records the current **in-process**
`llama-cpp-python` path so the provider refactor can be proven
behavior-preserving. HTTP/SSE overhead (including base64 images) must be
measured against these numbers in **A10.4** and **A11.2**, not assumed
negligible.

- Captured: 2026-11-24
- Commit: `00a0c83`
- Machine: MacBook Air M3, 8 cores, 16 GB RAM
- OS: macOS 27.2.1 (build 26B131)
- Python: 3.12.6
- `llama-cpp-python`: 0.3.36
- Backend: Metal (`n_gpu_layers=-1`), `n_threads=4`, model `n_ctx` as registered

## A1.1 Test suite

```text
$ python3 -m unittest discover -s tests -v
Ran 46 tests in 0.535s
OK
```

Wall time was ~2.1 s. The run emits harmless duplicate-`libavdevice` Objective-C
warnings from `av`/`opencv` on macOS; no test fails.

> **Historical note:** `scripts/benchmark_local.py` was removed along with the
> in-process GGUF path. The numbers below are the frozen baseline for future
> HTTP comparisons; a remote benchmark will replace the harness.

## Benchmark method

```bash
python3 scripts/benchmark_local.py \
    --model qwen35-2b --model qwen35-4b \
    --runs 3 --json docs/baseline.json
```

- Calls the real application functions: `initialize_llm()`,
  `generate_response()`, and `generate_vision_response()`.
- One warm-up call per turn type, then 3 measured runs with a fresh
  `Conversation` each time.
- Chat prompt: “What is two plus two? Answer in one short sentence.”
- Vision prompt: “What is in this photo? Answer in one short sentence.”
- Image: `captures/sparky_1795160189.jpg` (1280×720 JPEG, 182,819 bytes, a
  close-up photo). `captures/` is gitignored; re-run with `--image PATH` to use a
  fixed file.
- Peak RSS is `getrusage(RUSAGE_SELF).ru_maxrss`, so it is a whole-process peak
  and is cumulative across models in one run. Model weights are mmap-backed, so
  this is not an exact model-memory figure.

### Model hashes

| Model | GGUF SHA-256 | mmproj SHA-256 |
|---|---|---|
| `qwen35-2b` | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` | `7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7` |
| `qwen35-4b` | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` | `cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864` |

## Results

End-to-end latency is the time inside one chat/vision turn (the vision turn
includes image preprocessing, CLIP encode, prefill, and decode). Cold is the
first call after load; median/min/max are warm runs.

| Model | Load (s) | Chat cold (s) | Chat median (s) | Chat range (s) | Chat tok/s | Vision cold (s) | Vision median (s) | Vision range (s) | Peak RSS (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| `qwen35-2b` | 5.80 | 1.85 | 0.53 | 0.49–0.59 | 38.0 | 4.37 | 4.29 | 4.18–4.29 | 2,225 |
| `qwen35-4b` | 1.65 | 1.66 | 0.97 | 0.93–1.12 | 16.6 | 6.84 | 7.71 | 7.00–8.16 | 3,808 |

Peak RSS after load only: 1,513 MiB (`qwen35-2b`) and 3,100 MiB (`qwen35-4b`).
The 4B entry includes memory already high-water-marked by the 2B run.

### Observations

- Warm chat is well under the SLO; the first chat call costs ~1.3 s more than
  warm calls, consistent with Metal graph/shader warm-up.
- Vision latency is dominated by image encode/prefill (~3 s in the logs), so
  “tokens/s” there reflects decode over a mostly fixed cost, not image
  throughput.
- The 4B model loaded faster than the 2B model on its first attempt because it
  ran second (Metal initialized and GGUF paging already warm). Load times are
  order-dependent; treat them as directional, not a model comparison.
- 2B chat: 20 completion tokens median; 4B chat: 16. 4B vision answers were
  longer (43 median) because the prompt does not cap length tightly.

## Must-not-regress checklist (A1.3)

The provider refactor must preserve all of the following, verified against the
existing tests and behavior:

- **Prompt/message order**: system prompt first, then chronological turns, then
  the current user turn; vision uses `VISION_SYSTEM_PROMPT`.
- **Vision message shape**: image first as `image_url` with a
  `data:image/jpeg;base64,` URL, then the task text.
- **Conversation memory**: turns are appended only on success; failures leave
  history untouched; `max_turns` and token-budget eviction keep the newest turn
  and never drop the system message.
- **Vision history is text-only**: stored as `Photo question: <task>` plus the
  answer; no base64 or data URLs are retained.
- **Reply cleanup**: markdown characters `* # _ \`` are stripped before
  returning/speaking.
- **Error fallbacks**: chat returns “Oops! My circuits got tangled…”; vision
  returns “I had trouble looking at that photo…”; errors do not update history.
- **Model selection precedence**: `--model` → `ROBOT_LLM_MODEL` → default
  (`qwen35-2b`).
- **CLI behavior**: `--list-models`, `--check-vision`, and `--look` keep their
  current outputs and exit ordering (check-vision exits before loading Whisper).
- **Startup ordering**: `--look` and `--check-vision` skip Whisper and the
  greeting; vision-disabled sessions speak the fallback instead of failing.
- **Latency envelope**: warm chat and vision should remain within the same order
  of magnitude as the table above once HTTP overhead is added.
