#!/usr/bin/env bash
# Verify that compressed WebAssembly binaries satisfy size budget constraints
set -euo pipefail

DIST_DIR="${1:-math_explorer_gui/dist}"
MAX_BYTES="${2:-3145728}" # 3 MB default threshold

if [ ! -d "$DIST_DIR" ]; then
    echo "Error: Directory '$DIST_DIR' not found." >&2
    exit 1
fi

if ! command -v brotli >/dev/null 2>&1; then
    echo "Error: 'brotli' command not found in PATH." >&2
    exit 1
fi

if ! command -v gzip >/dev/null 2>&1; then
    echo "Error: 'gzip' command not found in PATH." >&2
    exit 1
fi

echo "Checking WebAssembly size budget against maximum threshold: $MAX_BYTES bytes (3 MB)..."

shopt -s nullglob
wasm_files=("$DIST_DIR"/*.wasm)
if [ ${#wasm_files[@]} -eq 0 ]; then
    echo "Error: No .wasm files found in '$DIST_DIR'." >&2
    exit 1
fi

passed=0
for wasm in "${wasm_files[@]}"; do
    wasm_base=$(basename "$wasm")
    if [ ! -s "$wasm" ]; then
        echo "Error: WebAssembly binary '$wasm' is empty." >&2
        exit 1
    fi

    br_size=$(brotli -9 -c < "$wasm" | wc -c | tr -d '[:space:]')
    gz_size=$(gzip -9 -c < "$wasm" | wc -c | tr -d '[:space:]')

    echo "Artifact $wasm_base:"
    echo "  - Brotli compressed size: $br_size bytes"
    echo "  - Gzip compressed size:   $gz_size bytes"

    if [ "$br_size" -gt "$MAX_BYTES" ]; then
        echo "Error: Brotli compressed size ($br_size bytes) exceeds 3MB limit ($MAX_BYTES bytes)" >&2
        exit 1
    fi

    if [ "$gz_size" -gt "$MAX_BYTES" ]; then
        echo "Error: Gzip compressed size ($gz_size bytes) exceeds 3MB limit ($MAX_BYTES bytes)" >&2
        exit 1
    fi

    passed=$((passed + 1))
done

echo "Success: All $passed WebAssembly bundle(s) passed the 3MB compressed size budget verification."
