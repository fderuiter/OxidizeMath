#!/usr/bin/env bash
# Verify that compressed WebAssembly binaries satisfy size budget constraints
set -euo pipefail

if [ "${1:-}" = "--sdk" ]; then
    DIST_DIR="${2:-target/wasm-sdk-pkg}"
    MAX_BYTES="${3:-256000}" # 250 KB strict SDK budget (250 * 1024 = 256,000 bytes)
else
    DIST_DIR="${1:-apps/math_explorer_gui/dist}"
    MAX_BYTES="${2:-5242880}" # 5 MB default GUI threshold
fi

if [ "$MAX_BYTES" -eq 5242880 ]; then
    BUDGET_LABEL="5 MB"
elif [ "$MAX_BYTES" -eq 256000 ]; then
    BUDGET_LABEL="250 KB"
else
    BUDGET_LABEL="${MAX_BYTES} bytes"
fi

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

echo "Checking WebAssembly size budget in '$DIST_DIR' against maximum threshold: $MAX_BYTES bytes ($BUDGET_LABEL)..."

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
        echo "Error: Brotli compressed size ($br_size bytes) exceeds $BUDGET_LABEL limit ($MAX_BYTES bytes)" >&2
        exit 1
    fi

    if [ "$gz_size" -gt "$MAX_BYTES" ]; then
        echo "Error: Gzip compressed size ($gz_size bytes) exceeds $BUDGET_LABEL limit ($MAX_BYTES bytes)" >&2
        exit 1
    fi

    passed=$((passed + 1))
done

echo "Success: All $passed WebAssembly bundle(s) passed the $BUDGET_LABEL compressed size budget verification."
