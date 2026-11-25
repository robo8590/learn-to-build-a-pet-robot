#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?Usage: $0 MODEL}"

HOST=127.0.0.1
PORT=8080
NGL=99
CTX=4096

RESOLVED="$(PYTHONPATH="$ROOT" python3 - "$MODEL" <<'PY'
import sys
from sparky.config.llm_registry import llm_path, mmproj_path, resolve_llm_key

try:
    key = resolve_llm_key(sys.argv[1])
except KeyError as exc:
    print(exc.args[0], file=sys.stderr)
    raise SystemExit(2)

projector = mmproj_path(key)
print(key, llm_path(key), projector or "", sep="\t")
PY
)" || exit 1

IFS=$'\t' read -r KEY GGUF MMPROJ <<< "$RESOLVED"

if [[ ! -f "$GGUF" ]]; then
    echo "Model not found: $GGUF" >&2
    echo "Download it with: python download_models.py --llm $KEY" >&2
    exit 1
fi

if [[ -n "$MMPROJ" && ! -f "$MMPROJ" ]]; then
    MMPROJ=""
fi

ARGS=(
    -m "$GGUF"
    --alias "$KEY"
    -ngl "$NGL"
    -c "$CTX"
    --host "$HOST"
    --port "$PORT"
    -rea off
)
if [[ -n "$MMPROJ" ]]; then
    ARGS+=(--mmproj "$MMPROJ")
fi

echo "LLM server: http://$HOST:$PORT/v1 (model: $KEY)"
exec llama serve "${ARGS[@]}"
