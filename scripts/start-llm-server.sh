#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?Usage: $0 MODEL}"

# Server-side settings use llama.cpp's own env names. Defaults keep the
# standalone/loopback behavior; a LAN bind requires a key file and TLS.
HOST="${LLAMA_ARG_HOST:-127.0.0.1}"
PORT="${LLAMA_ARG_PORT:-8080}"
API_KEY_FILE="${LLAMA_ARG_API_KEY_FILE:-}"
TLS_CERT="${LLAMA_ARG_SSL_CERT_FILE:-}"
TLS_KEY="${LLAMA_ARG_SSL_KEY_FILE:-}"
NGL=99
CTX=4096

die() {
    echo "Error: $*" >&2
    exit 1
}

# Only addresses that are definitely local count as loopback. Anything else
# (0.0.0.0, a LAN IP, an unrecognized host) must be authenticated and encrypted.
is_loopback_bind() {
    local addr
    local -a addrs=()
    IFS=',' read -r -a addrs <<< "$1"
    [[ ${#addrs[@]} -gt 0 ]] || return 1
    for addr in "${addrs[@]}"; do
        case "$addr" in
            localhost|127.*|::1|\[::1\]|*.sock) ;;
            *) return 1 ;;
        esac
    done
    return 0
}

# TLS is all-or-nothing, even for a loopback bind.
if [[ -n "$TLS_CERT" || -n "$TLS_KEY" ]]; then
    [[ -n "$TLS_CERT" && -n "$TLS_KEY" ]] || die "Set both LLAMA_ARG_SSL_CERT_FILE and LLAMA_ARG_SSL_KEY_FILE, or neither."
    [[ -r "$TLS_CERT" ]] || die "TLS certificate is not readable: $TLS_CERT"
    [[ -r "$TLS_KEY" ]] || die "TLS private key is not readable: $TLS_KEY"
fi

if ! is_loopback_bind "$HOST"; then
    [[ -n "$API_KEY_FILE" ]] || die "Non-loopback bind ($HOST) requires LLAMA_ARG_API_KEY_FILE (a file with one API key per line)."
    [[ -r "$API_KEY_FILE" ]] || die "API key file is not readable: $API_KEY_FILE"
    [[ -n "$TLS_CERT" ]] || die "Non-loopback bind ($HOST) requires HTTPS: set LLAMA_ARG_SSL_CERT_FILE and LLAMA_ARG_SSL_KEY_FILE."

    # Surface loose key-file permissions without blocking the happy path.
    perms="$(stat -f '%Lp' "$API_KEY_FILE" 2>/dev/null || stat -c '%a' "$API_KEY_FILE" 2>/dev/null || true)"
    if [[ -n "$perms" ]] && (( 8#$perms & 077 )); then
        echo "Warning: API key file $API_KEY_FILE is group/world-accessible ($perms); chmod 600 it." >&2
    fi
fi

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
if [[ -n "$API_KEY_FILE" ]]; then
    # Only the path goes into argv; the key itself stays in the file.
    ARGS+=(--api-key-file "$API_KEY_FILE")
fi
if [[ -n "$TLS_CERT" && -n "$TLS_KEY" ]]; then
    ARGS+=(--ssl-cert-file "$TLS_CERT" --ssl-key-file "$TLS_KEY")
fi

SCHEME=http
if [[ -n "$TLS_CERT" ]]; then
    SCHEME=https
fi

echo "LLM server: $SCHEME://$HOST:$PORT/v1 (model: $KEY)"
exec llama serve "${ARGS[@]}"
