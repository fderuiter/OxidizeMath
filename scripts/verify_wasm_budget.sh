#!/usr/bin/env bash
# Verify that compressed WebAssembly binaries satisfy size budget constraints
set -euo pipefail

DIST_DIR="${1:-math_explorer_gui/dist}"
MAX_BYTES="${2:-3145728}" # 3 MB default threshold

if [ ! -d "$DIST_DIR" ]; then
    echo "Error: Directory '$DIST_DIR' not found." >&2
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
    br_file="${wasm}.br"
    gz_file="${wasm}.gz"

    if [ ! -f "$br_file" ]; then
        echo "Error: Missing Brotli compressed file: $br_file" >&2
        exit 1
    fi

    if [ ! -f "$gz_file" ]; then
        echo "Error: Missing Gzip compressed file: $gz_file" >&2
        exit 1
    fi

    if command -v stat >/dev/null 2>&1; then
        br_size=$(stat -c%s "$br_file" 2>/dev/null || stat -f%z "$br_file")
        gz_size=$(stat -c%s "$gz_file" 2>/dev/null || stat -f%z "$gz_file")
    else
        br_size=$(wc -c < "$br_file" | tr -d ' ')
        gz_size=$(wc -c < "$gz_file" | tr -d ' ')
    fi

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
