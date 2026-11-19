#!/usr/bin/env bash
# Externalize Trunk's inline module bootstrap for strict CSP compliance,
# harden WASM fetch stream error propagation, and pre-compress dist assets.
set -euo pipefail

STAGING_DIR="${1:-${TRUNK_STAGING_DIR:-math_explorer_gui/dist}}"
INDEX_HTML="$STAGING_DIR/index.html"
LOADER_JS="$STAGING_DIR/trunk-loader.js"

if [ -f "$STAGING_DIR/icon.ico" ]; then
    mv "$STAGING_DIR/icon.ico" "$STAGING_DIR/favicon.ico"
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

shopt -s nullglob
for ext in wasm js html css svg webmanifest; do
    for f in "$STAGING_DIR"/*."$ext"; do
        if [ -f "$f" ]; then
            gzip -k -9 -f "$f"
            brotli -k -9 -f "$f"
        fi
    done
done
