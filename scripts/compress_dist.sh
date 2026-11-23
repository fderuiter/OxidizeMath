#!/usr/bin/env bash
# Pre-compress static distribution assets with Gzip and Brotli
set -euo pipefail

TARGET_DIR="${1:-math_explorer_gui/dist}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory '$TARGET_DIR' does not exist." >&2
    exit 1
fi

if [ "$#" -gt 1 ]; then
    EXTS=("${@:2}")
else
    EXTS=(wasm js html css svg webmanifest json)
fi

echo "Pre-compressing static distribution assets in $TARGET_DIR (${EXTS[*]})..."

shopt -s nullglob
for ext in "${EXTS[@]}"; do
    for f in "$TARGET_DIR"/*."$ext"; do
        if [ -f "$f" ]; then
            gzip -k -9 -f "$f"
            brotli -k -9 -f "$f"
            echo "Pre-compressed: $f (.gz, .br)"
        fi
    done
done

echo "Pre-compression complete."
