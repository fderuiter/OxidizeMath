#!/usr/bin/env bash
# Stage precomputed traceability manifest, local paper artifacts (when present),
# favicon, and exact WASM byte size into Trunk's web distribution directory.
set -euo pipefail

STAGING_DIR="${1:-${TRUNK_STAGING_DIR:-math_explorer_gui/dist}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BOOTSTRAP_JS="$STAGING_DIR/bootstrap.js"

MANIFEST_SRC=""
for cand in \
    "${CARGO_TARGET_DIR:-}/wasm32-unknown-unknown/release/traceability_manifest.json" \
    "${CARGO_TARGET_DIR:-}/wasm32-unknown-unknown/debug/traceability_manifest.json" \
    "${CARGO_TARGET_DIR:-}/traceability_manifest.json" \
    "$REPO_ROOT/target/wasm32-unknown-unknown/release/traceability_manifest.json" \
    "$REPO_ROOT/target/wasm32-unknown-unknown/debug/traceability_manifest.json" \
    "$REPO_ROOT/target/traceability_manifest.json"; do
    if [ -n "$cand" ] && [ -f "$cand" ]; then
        MANIFEST_SRC="$cand"
        break
    fi
done
if [ -z "$MANIFEST_SRC" ]; then
    for out_cand in \
        "${CARGO_TARGET_DIR:-}"/wasm32-unknown-unknown/release/build/math_explorer_gui-*/out/traceability_manifest.json \
        "${CARGO_TARGET_DIR:-}"/wasm32-unknown-unknown/debug/build/math_explorer_gui-*/out/traceability_manifest.json \
        "$REPO_ROOT"/target/wasm32-unknown-unknown/release/build/math_explorer_gui-*/out/traceability_manifest.json \
        "$REPO_ROOT"/target/wasm32-unknown-unknown/debug/build/math_explorer_gui-*/out/traceability_manifest.json; do
        if [ -f "$out_cand" ]; then
            MANIFEST_SRC="$out_cand"
            break
        fi
    done
fi
if [ -z "$MANIFEST_SRC" ]; then
    echo "Error: traceability_manifest.json not found in Cargo target directory" >&2
    exit 1
fi
cp -f "$MANIFEST_SRC" "$STAGING_DIR/traceability_manifest.json"

if [ -d "$REPO_ROOT/papers/output" ]; then
    shopt -s nullglob
    pdfs=("$REPO_ROOT"/papers/output/*.pdf)
    if [ ${#pdfs[@]} -gt 0 ]; then
        cp -f -- "${pdfs[@]}" "$STAGING_DIR/"
    fi
    if [ -s "$REPO_ROOT/papers/output/pages-index.html" ]; then
        mkdir -p "$STAGING_DIR/papers"
        cp -f -- "$REPO_ROOT/papers/output/pages-index.html" "$STAGING_DIR/papers/index.html"
    fi
    shopt -u nullglob
fi

if [ -f "$STAGING_DIR/icon.ico" ]; then
    mv "$STAGING_DIR/icon.ico" "$STAGING_DIR/favicon.ico"
fi

wasm_size=0
for wf in "$STAGING_DIR"/*.wasm; do
    if [ -f "$wf" ]; then
        wasm_size=$(wc -c < "$wf" | tr -d '[:space:]')
        break
    fi
done

if [ -f "$BOOTSTRAP_JS" ] && [ "$wasm_size" -gt 0 ]; then
    awk -v wasm_size="$wasm_size" '
        /^const WASM_SOURCE_SIZE = [0-9]+;$/ {
            print "const WASM_SOURCE_SIZE = " wasm_size ";"
            next
        }
        { print }
    ' "$BOOTSTRAP_JS" > "${BOOTSTRAP_JS}.tmp" && mv "${BOOTSTRAP_JS}.tmp" "$BOOTSTRAP_JS"
fi
