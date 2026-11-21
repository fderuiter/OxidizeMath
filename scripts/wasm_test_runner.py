#!/usr/bin/env python3
"""Headless Chromium runner for wasm32-unknown-unknown unit tests using Playwright."""

from __future__ import annotations

import functools
import http.server
import mimetypes
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from pathlib import Path
from playwright.sync_api import sync_playwright

PRODUCTION_CSP = (
    "default-src 'self'; "
    "base-uri 'self'; "
    "object-src 'none'; "
    "script-src 'self' 'wasm-unsafe-eval' blob:; "
    "style-src 'self' 'unsafe-inline'; "
    "worker-src 'self' blob:; "
    "connect-src 'self' blob:;"
)

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="color-scheme" content="dark">
    <meta name="theme-color" content="#121214">
    <title>OxidizeMath WASM Unit Test Runner</title>
</head>
<body>
    <div id="loading_indicator" role="status" aria-live="polite" aria-busy="true">Loading...</div>
    <canvas id="the_canvas_id" tabindex="0"></canvas>
    <div id="error_container" role="alert" aria-live="assertive" style="display: none;">
        <h1 id="error_title">Error</h1>
        <p id="error_message"></p>
        <div id="webgl_instructions"></div>
        <button id="webgl_reload_button" type="button">Reload</button>
        <pre id="error_details" style="display: none;"></pre>
    </div>
    <pre id="output"></pre>
    <script type="module" src="./runner.js"></script>
</body>
</html>
"""

RUNNER_JS = """import init, * as bindings from "./wasm_test_bin.js";

const outputEl = document.getElementById("output");
const rawConsoleLog = console.log.bind(console);
const rawConsoleInfo = console.info.bind(console);
const rawConsoleWarn = console.warn.bind(console);
const rawConsoleError = console.error.bind(console);
const rawConsoleDebug = console.debug.bind(console);

globalThis.__wbg_test_invoke = (fn) => fn();
globalThis.__wbgtest_og_console_log = (line) => rawConsoleLog(line);
globalThis.__wbg_test_output_writeln = (value) => {
    const text = typeof value === "string" ? value : (value && value.textContent ? value.textContent : String(value));
    if (outputEl) {
        outputEl.textContent += text + "\\n";
    }
};

async function runSuite() {
    try {
        const wasm = await init();
        if (typeof bindings.__wbgtest_console_log === "function") {
            console.log = (...args) => bindings.__wbgtest_console_log(args);
            console.info = (...args) => bindings.__wbgtest_console_info(args);
            console.warn = (...args) => bindings.__wbgtest_console_warn(args);
            console.error = (...args) => bindings.__wbgtest_console_error(args);
            console.debug = (...args) => bindings.__wbgtest_console_debug(args);
        }

        const params = new URLSearchParams(window.location.search);
        const nameFilter = params.get("filter") || "";
        const includeIgnored = params.get("include_ignored") === "1";

        const allTests = Object.entries(wasm)
            .filter(([key, val]) => key.startsWith("__wbgt_") && typeof val === "function");
        const selected = nameFilter
            ? allTests.filter(([key]) => key.includes(nameFilter))
            : allTests;
        const filteredOut = allTests.length - selected.length;

        const cx = new bindings.WasmBindgenTestContext(false);
        if (filteredOut > 0 && typeof cx.filtered_count === "function") {
            cx.filtered_count(filteredOut);
        }
        if (includeIgnored && typeof cx.include_ignored === "function") {
            cx.include_ignored(true);
        }

        const res = await cx.run(selected.map(([, fn]) => fn));
        const outText = outputEl ? outputEl.textContent || "" : "";
        const passed = res === true || (res !== false && outText.includes("test result: ok."));
        outputEl.setAttribute("data-test-status", passed ? "ok" : "failed");
    } catch (err) {
        const detail = err && err.stack ? err.stack : String(err);
        outputEl.textContent += "\\nRunner fatal error: " + detail + "\\n";
        outputEl.setAttribute("data-test-status", "failed");
    }
}

runSuite();
"""


class ReusableThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class WasmTestRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/__test_vfs_fixture.txt":
            body = b"oxidize-wasm-vfs-ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/__missing_vfs_fixture_404.txt":
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"not found")
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Content-Security-Policy", PRODUCTION_CSP)
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        return


def parse_cargo_test_args(extra_args: list[str]) -> tuple[str, bool, bool]:
    name_filter = ""
    include_ignored = False
    list_only = False
    for arg in extra_args:
        if arg == "--ignored" or arg == "--include-ignored":
            include_ignored = True
        elif arg == "--list":
            list_only = True
        elif not arg.startswith("-"):
            name_filter = arg
    return name_filter, include_ignored, list_only


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: wasm_test_runner.py <path-to-test.wasm> [test-args...]", file=sys.stderr)
        return 2

    wasm_path = Path(sys.argv[1]).resolve()
    if not wasm_path.is_file():
        print(f"Error: WASM test binary not found: {wasm_path}", file=sys.stderr)
        return 2

    name_filter, include_ignored, list_only = parse_cargo_test_args(sys.argv[2:])
    if list_only:
        return 0

    wasm_bindgen_bin = shutil.which("wasm-bindgen")
    if not wasm_bindgen_bin:
        print("Error: 'wasm-bindgen' CLI not found on PATH. Install wasm-bindgen-cli.", file=sys.stderr)
        return 2

    mimetypes.add_type("application/wasm", ".wasm")
    mimetypes.add_type("text/javascript", ".js")

    with tempfile.TemporaryDirectory(prefix="oxidize_wasm_tests_") as tmp_dir:
        out_dir = Path(tmp_dir)
        proc = subprocess.run(
            [
                wasm_bindgen_bin,
                str(wasm_path),
                "--out-dir",
                str(out_dir),
                "--out-name",
                "wasm_test_bin",
                "--target",
                "web",
                "--no-typescript",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            return proc.returncode

        js_wrapper = out_dir / "wasm_test_bin.js"
        js_text = js_wrapper.read_text(encoding="utf-8")
        if "export class WasmBindgenTestContext" not in js_text:
            print(f"\nrunning 0 wasm-bindgen tests in {wasm_path.name}\ntest result: ok. 0 passed; 0 failed\n")
            return 0

        js_text = js_text.replace(
            "wasm.__wbindgen_start();",
            'if (typeof wasm.__wbindgen_init_externref_table === "function") { wasm.__wbindgen_init_externref_table(); }',
        )
        js_wrapper.write_text(js_text, encoding="utf-8")
        (out_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
        (out_dir / "runner.js").write_text(RUNNER_JS, encoding="utf-8")

        handler = functools.partial(WasmTestRequestHandler, directory=str(out_dir))
        server = ReusableThreadingTCPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        csp_errors: list[str] = []
        page_errors: list[str] = []
        try:
            query = urllib.parse.urlencode(
                {
                    "filter": name_filter,
                    "include_ignored": "1" if include_ignored else "0",
                }
            )
            url = f"http://127.0.0.1:{port}/index.html?{query}"
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--enable-webgl",
                        "--use-gl=angle",
                        "--use-angle=swiftshader",
                        "--enable-unsafe-swiftshader",
                    ],
                )
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()
                page.on(
                    "console",
                    lambda msg: csp_errors.append(msg.text)
                    if msg.type == "error" and "Content Security Policy" in msg.text
                    else None,
                )
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))

                page.goto(url, wait_until="domcontentloaded")
                page.locator("#output[data-test-status]").wait_for(
                    state="attached",
                    timeout=90_000,
                )
                status = page.locator("#output").get_attribute("data-test-status")
                rendered_output = page.locator("#output").inner_text()
                browser.close()
        finally:
            server.shutdown()
            server.server_close()

        if rendered_output.strip():
            print(rendered_output)

        if csp_errors:
            for err in csp_errors:
                print(f"CSP violation during WASM unit tests: {err}", file=sys.stderr)
            return 1
        if page_errors:
            for err in page_errors:
                print(f"Uncaught page error during WASM unit tests: {err}", file=sys.stderr)
            return 1

        return 0 if status == "ok" else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
