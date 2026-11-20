#!/usr/bin/env bash
# Externalize Trunk's inline module bootstrap for strict CSP compliance,
# harden WASM fetch stream error propagation, and pre-compress dist assets.
set -euo pipefail

STAGING_DIR="${1:-${TRUNK_STAGING_DIR:-math_explorer_gui/dist}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INDEX_HTML="$STAGING_DIR/index.html"
LOADER_JS="$STAGING_DIR/trunk-loader.js"
BOOTSTRAP_JS="$STAGING_DIR/bootstrap.js"

if [ ! -f "$REPO_ROOT/Verification_Certificate.json" ]; then
    echo "Error: $REPO_ROOT/Verification_Certificate.json not found for web staging" >&2
    exit 1
fi
cp -f "$REPO_ROOT/Verification_Certificate.json" "$STAGING_DIR/Verification_Certificate.json"

SNAPSHOT_SRC=""
for cand in \
    "${CARGO_TARGET_DIR:-}/wasm32-unknown-unknown/release/traceability_snapshot.json" \
    "${CARGO_TARGET_DIR:-}/wasm32-unknown-unknown/debug/traceability_snapshot.json" \
    "${CARGO_TARGET_DIR:-}/traceability_snapshot.json" \
    "$REPO_ROOT/target/wasm32-unknown-unknown/release/traceability_snapshot.json" \
    "$REPO_ROOT/target/wasm32-unknown-unknown/debug/traceability_snapshot.json" \
    "$REPO_ROOT/target/traceability_snapshot.json"; do
    if [ -n "$cand" ] && [ -f "$cand" ]; then
        SNAPSHOT_SRC="$cand"
        break
    fi
done
if [ -z "$SNAPSHOT_SRC" ]; then
    for out_cand in \
        "${CARGO_TARGET_DIR:-}"/wasm32-unknown-unknown/release/build/math_explorer_gui-*/out/traceability_snapshot.json \
        "${CARGO_TARGET_DIR:-}"/wasm32-unknown-unknown/debug/build/math_explorer_gui-*/out/traceability_snapshot.json \
        "$REPO_ROOT"/target/wasm32-unknown-unknown/release/build/math_explorer_gui-*/out/traceability_snapshot.json \
        "$REPO_ROOT"/target/wasm32-unknown-unknown/debug/build/math_explorer_gui-*/out/traceability_snapshot.json; do
        if [ -f "$out_cand" ]; then
            SNAPSHOT_SRC="$out_cand"
            break
        fi
    done
fi
if [ -z "$SNAPSHOT_SRC" ]; then
    echo "Error: traceability_snapshot.json not found in Cargo target directory" >&2
    exit 1
fi
cp -f "$SNAPSHOT_SRC" "$STAGING_DIR/traceability_snapshot.json"

if [ -d "$REPO_ROOT/papers/output" ]; then
    shopt -s nullglob
    pdfs=("$REPO_ROOT"/papers/output/*.pdf)
    if [ ${#pdfs[@]} -gt 0 ]; then
        mkdir -p "$STAGING_DIR/papers"
        cp -f "${pdfs[@]}" "$STAGING_DIR/"
        cp -f "${pdfs[@]}" "$STAGING_DIR/papers/"
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

if [ -f "$INDEX_HTML" ] && grep -q '<script type="module">' "$INDEX_HTML"; then
    awk -v loader="$LOADER_JS" '
        BEGIN { in_mod = 0 }
        /<script type="module">/ {
            in_mod = 1
            print "<script type=\"module\" src=\"./trunk-loader.js\"></script>"
            next
        }
        in_mod && /<\/script>/ {
            in_mod = 0
            next
        }
        in_mod {
            print > loader
            next
        }
        { print }
    ' "$INDEX_HTML" > "${INDEX_HTML}.tmp" && mv "${INDEX_HTML}.tmp" "$INDEX_HTML"

    if [ -f "$LOADER_JS" ]; then
        wasm_size=0
        for wf in "$STAGING_DIR"/*.wasm; do
            if [ -f "$wf" ]; then
                wasm_size=$(wc -c < "$wf" | tr -d '[:space:]')
                break
            fi
        done

        awk -v wasm_size="$wasm_size" '
            {
                if (wasm_size > 0 && $0 ~ /__trunkInitializer\(init,/) {
                    sub(/\.wasm'\'', [0-9]+,/, ".wasm'\'', " wasm_size ",")
                }
                print
            }
            /const statusText = response\.statusText;/ {
                print "        if (!response.ok) { throw new Error(\"HTTP \" + status + \" (\" + (statusText || \"Request Failed\") + \") while fetching \" + source); }"
            }
        ' "$LOADER_JS" | awk '
            {
                if ($0 ~ /^              \}\);$/ && prev ~ /push\(\);/) {
                    print "              }).catch((err) => { controller.error(err); });"
                } else {
                    print $0
                }
                prev = $0
            }
        ' > "${LOADER_JS}.tmp" && mv "${LOADER_JS}.tmp" "$LOADER_JS"
    fi
fi

bash "$SCRIPT_DIR/compress_dist.sh" "$STAGING_DIR"
