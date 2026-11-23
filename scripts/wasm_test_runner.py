#!/usr/bin/env python3
"""Headless Chromium runner for wasm32-unknown-unknown unit and browser tests using Playwright."""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import http.server
import json
import mimetypes
import os
try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
import shutil
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from pathlib import Path

PRODUCTION_CSP = (
    "default-src 'self'; "
    "base-uri 'self'; "
    "object-src 'none'; "
    "script-src 'self' 'wasm-unsafe-eval' blob:; "
    "style-src 'self' 'unsafe-inline'; "
    "worker-src 'self' blob:; "
    "connect-src 'self' blob:;"
)

CHROMIUM_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--enable-webgl",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
]

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

RUNNER_JS = """const outputEl = document.getElementById("output");
const rawConsoleLog = console.log.bind(console);

function appendLine(value) {
    const text = typeof value === "string"
        ? value
        : (value && value.textContent ? value.textContent : String(value));
    if (outputEl) {
        outputEl.textContent += text + "\\n";
    }
}

globalThis.__wbg_test_invoke = (fn) => fn();
globalThis.__wbgtest_og_console_log = (line) => rawConsoleLog(line);
globalThis.__wbg_test_output_writeln = (value) => appendLine(value);

async function runWbgAndOptionalUnit(cfg) {
    const bindings = await import("./wasm_test_bin.js");
    const wasm = await bindings.default();
    if (typeof bindings.__wbgtest_console_log === "function") {
        console.log = (...args) => bindings.__wbgtest_console_log(args);
        console.info = (...args) => bindings.__wbgtest_console_info(args);
        console.warn = (...args) => bindings.__wbgtest_console_warn(args);
        console.error = (...args) => bindings.__wbgtest_console_error(args);
        console.debug = (...args) => bindings.__wbgtest_console_debug(args);
    }

    let wbgPassed = true;
    if (cfg.run_wbg) {
        const selectedFns = [];
        for (const exportName of cfg.selected_wbgt_exports) {
            const fn = wasm[exportName];
            if (typeof fn !== "function") {
                throw new Error("Missing exported wasm-bindgen test function: " + exportName);
            }
            selectedFns.push(fn);
        }
        const cx = new bindings.WasmBindgenTestContext(false);
        if (cfg.filtered_out_wbgt > 0 && typeof cx.filtered_count === "function") {
            cx.filtered_count(cfg.filtered_out_wbgt);
        }
        if (cfg.include_ignored && typeof cx.include_ignored === "function") {
            cx.include_ignored(true);
        }
        const res = await cx.run(selectedFns);
        const outText = outputEl ? outputEl.textContent || "" : "";
        wbgPassed = res === true || (res !== false && outText.includes("test result: ok."));
    }

    let unitPassed = true;
    if (cfg.run_unit) {
        if (typeof wasm.main !== "function") {
            throw new Error("Expected exported main(argc, argv) function in unit test target");
        }
        const rc = wasm.main(0, 0);
        unitPassed = rc === 0;
        appendLine(
            `running ${cfg.unit_test_count} unit tests in ${cfg.binary_label}\\n` +
            `test result: ${unitPassed ? "ok" : "FAILED"}. ` +
            `${unitPassed ? cfg.unit_test_count : 0} passed; ${unitPassed ? 0 : 1} failed`
        );
    }

    return wbgPassed && unitPassed;
}

async function runRawUnit(cfg) {
    const resp = await fetch("./raw_unit_test.wasm");
    if (!resp.ok) {
        throw new Error("Failed to fetch raw_unit_test.wasm: HTTP " + resp.status);
    }
    const bytes = await resp.arrayBuffer();
    const mod = await WebAssembly.compile(bytes);
    const importObj = {};
    for (const imp of WebAssembly.Module.imports(mod)) {
        if (!importObj[imp.module]) {
            importObj[imp.module] = {};
        }
        if (imp.kind === "function") {
            importObj[imp.module][imp.name] = () => 0;
        }
    }
    const instance = await WebAssembly.instantiate(mod, importObj);
    if (typeof instance.exports.main !== "function") {
        throw new Error("Raw WASM unit binary does not export main(argc, argv)");
    }
    const rc = instance.exports.main(0, 0);
    const passed = rc === 0;
    appendLine(
        `running ${cfg.unit_test_count} unit tests in ${cfg.binary_label}\\n` +
        `test result: ${passed ? "ok" : "FAILED"}. ` +
        `${passed ? cfg.unit_test_count : 0} passed; ${passed ? 0 : 1} failed`
    );
    return passed;
}

async function runSuite() {
    try {
        const cfgResp = await fetch("./runner_config.json");
        if (!cfgResp.ok) {
            throw new Error("Failed to load runner_config.json: HTTP " + cfgResp.status);
        }
        const cfg = await cfgResp.json();
        const passed = cfg.mode === "raw_unit"
            ? await runRawUnit(cfg)
            : await runWbgAndOptionalUnit(cfg);
        outputEl.setAttribute("data-test-status", passed ? "ok" : "failed");
    } catch (err) {
        const detail = err && err.stack ? err.stack : String(err);
        appendLine("\\nRunner fatal error: " + detail);
        outputEl.setAttribute("data-test-status", "failed");
    }
}

runSuite();
"""


@dataclass(frozen=True)
class DiscoveredTest:
    canonical_name: str
    crate_qualified_name: str
    export_name: str | None
    ignored: bool = False


@dataclass(frozen=True)
class WasmBinarySummary:
    crate_name: str
    has_wbindgen_imports: bool
    has_main_export: bool
    wbgt_tests: list[DiscoveredTest]
    unit_tests: list[DiscoveredTest]


@dataclass
class TestRunnerOptions:
    filters: list[str] = field(default_factory=list)
    skips: list[str] = field(default_factory=list)
    exact: bool = False
    list_only: bool = False
    include_ignored: bool = False
    only_ignored: bool = False
    test_threads: int | None = None
    quiet: bool = False
    nocapture: bool = False
    output_format: str = "pretty"
    show_help: bool = False


class ReusableThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128


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


def parse_cargo_test_args(extra_args: list[str]) -> TestRunnerOptions:
    opts = TestRunnerOptions()
    idx = 0
    positional_only = False
    while idx < len(extra_args):
        arg = extra_args[idx]
        if positional_only:
            opts.filters.append(arg)
            idx += 1
            continue
        if arg == "--":
            positional_only = True
        elif arg in ("-h", "--help"):
            opts.show_help = True
        elif arg == "--list":
            opts.list_only = True
        elif arg == "--exact":
            opts.exact = True
        elif arg == "--ignored":
            opts.only_ignored = True
            opts.include_ignored = True
        elif arg == "--include-ignored":
            opts.include_ignored = True
        elif arg in ("-q", "--quiet"):
            opts.quiet = True
        elif arg in ("--nocapture", "--no-capture", "--show-output"):
            opts.nocapture = True
        elif arg in ("--test", "--bench", "--report-time", "--ensure-time", "--shuffle"):
            pass
        elif arg == "--skip":
            if idx + 1 >= len(extra_args):
                raise ValueError("Argument to option '--skip' missing")
            idx += 1
            opts.skips.append(extra_args[idx])
        elif arg.startswith("--skip="):
            opts.skips.append(arg.split("=", 1)[1])
        elif arg in ("--test-threads", "-t"):
            if idx + 1 >= len(extra_args):
                raise ValueError(f"Argument to option '{arg}' missing")
            idx += 1
            opts.test_threads = _parse_positive_int(arg, extra_args[idx])
        elif arg.startswith("--test-threads="):
            opts.test_threads = _parse_positive_int("--test-threads", arg.split("=", 1)[1])
        elif arg == "--format":
            if idx + 1 >= len(extra_args):
                raise ValueError("Argument to option '--format' missing")
            idx += 1
            opts.output_format = extra_args[idx]
        elif arg.startswith("--format="):
            opts.output_format = arg.split("=", 1)[1]
        elif arg in ("--color", "--logfile", "--shuffle-seed", "-Z"):
            if idx + 1 >= len(extra_args):
                raise ValueError(f"Argument to option '{arg}' missing")
            idx += 1
        elif arg.startswith(("--color=", "--logfile=", "--shuffle-seed=")) or (
            arg.startswith("-Z") and len(arg) > 2
        ):
            pass
        elif arg.startswith("-"):
            raise ValueError(f"Unrecognized option: '{arg}'")
        else:
            opts.filters.append(arg)
        idx += 1
    return opts


def _parse_positive_int(flag: str, raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for '{flag}': {raw!r}") from exc
    if value < 1:
        raise ValueError(f"Value for '{flag}' must be >= 1, got {value}")
    return value


def _infer_crate_name(wasm_path: Path) -> str:
    stem = wasm_path.stem
    if "-" in stem:
        prefix, suffix = stem.rsplit("-", 1)
        if len(suffix) == 16 and all(c in "0123456789abcdef" for c in suffix.lower()):
            return prefix
    return stem


def _read_uleb128(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, pos
        shift += 7


def _read_sleb128(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if (byte & 0x80) == 0:
            if shift < 64 and (byte & 0x40) != 0:
                result |= -(1 << shift)
            return result, pos


def _read_wasm_string(buf: bytes, pos: int) -> tuple[str, int]:
    length, pos = _read_uleb128(buf, pos)
    text = buf[pos : pos + length].decode("utf-8", errors="replace")
    return text, pos + length


def _skip_limits(buf: bytes, pos: int) -> int:
    flags, pos = _read_uleb128(buf, pos)
    _, pos = _read_uleb128(buf, pos)
    if flags & 0x01:
        _, pos = _read_uleb128(buf, pos)
    return pos


def _parse_const_i32_expr(buf: bytes, pos: int) -> tuple[int | None, int]:
    opcode = buf[pos]
    pos += 1
    offset: int | None = None
    if opcode == 0x41:  # i32.const
        offset, pos = _read_sleb128(buf, pos)
    elif opcode == 0x42:  # i64.const
        _, pos = _read_sleb128(buf, pos)
    elif opcode == 0x23:  # global.get
        _, pos = _read_uleb128(buf, pos)
    while pos < len(buf) and buf[pos] != 0x0B:
        pos += 1
    return offset, pos + 1


def _extract_libtest_descriptors(
    active_segments: list[tuple[int, bytes]],
    crate_name: str,
) -> list[DiscoveredTest]:
    def read_mem(ptr: int, length: int) -> bytes | None:
        for base, seg in active_segments:
            if base <= ptr and ptr + length <= base + len(seg):
                rel = ptr - base
                return seg[rel : rel + length]
        return None

    seen: set[str] = set()
    discovered: list[DiscoveredTest] = []
    for _base, seg in active_segments:
        limit = len(seg) - 48
        for pos in range(0, max(0, limit + 1), 4):
            (
                name_ptr,
                name_len,
                name_tag,
                src_ptr,
                src_len,
                msg_ptr,
                msg_len,
                start_line,
                start_col,
                end_line,
                end_col,
                flags,
            ) = struct.unpack_from("<12I", seg, pos)
            if name_tag != 0 or not (1 <= name_len <= 256 and 4 <= src_len <= 256):
                continue
            if not (
                1 <= start_line <= end_line <= 200_000
                and 1 <= start_col <= 10_000
                and 1 <= end_col <= 10_000
            ):
                continue
            if msg_ptr == 0 and msg_len != 0:
                continue
            src_bytes = read_mem(src_ptr, src_len)
            if not src_bytes or not src_bytes.endswith(b".rs"):
                continue
            name_bytes = read_mem(name_ptr, name_len)
            if not name_bytes:
                continue
            try:
                src_str = src_bytes.decode("utf-8")
                name_str = name_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if not src_str or not all(c.isalnum() or c in "_:" for c in name_str):
                continue
            if name_str in seen:
                continue
            seen.add(name_str)
            ignored = msg_ptr != 0 or ((flags >> 8) & 0xFF) == 1
            crate_qual = (
                name_str
                if name_str.startswith(f"{crate_name}::")
                else f"{crate_name}::{name_str}"
            )
            discovered.append(
                DiscoveredTest(
                    canonical_name=name_str,
                    crate_qualified_name=crate_qual,
                    export_name=None,
                    ignored=ignored,
                )
            )
    return discovered


def inspect_wasm_binary(wasm_path: Path) -> WasmBinarySummary:
    data = wasm_path.read_bytes()
    if len(data) < 8 or data[:4] != b"\x00asm":
        raise RuntimeError(f"Invalid WebAssembly binary header in {wasm_path}")

    crate_name = _infer_crate_name(wasm_path)
    has_wbindgen_imports = False
    has_main_export = False
    wbgt_tests: list[DiscoveredTest] = []
    active_segments: list[tuple[int, bytes]] = []

    idx = 8
    while idx < len(data):
        section_id = data[idx]
        idx += 1
        section_len, idx = _read_uleb128(data, idx)
        section_end = idx + section_len

        if section_id == 2:  # Import section
            count, p = _read_uleb128(data, idx)
            for _ in range(count):
                mod_name, p = _read_wasm_string(data, p)
                field_name, p = _read_wasm_string(data, p)
                if mod_name.startswith("__wbindgen") or field_name.startswith("__wbindgen"):
                    has_wbindgen_imports = True
                kind = data[p]
                p += 1
                if kind == 0x00:
                    _, p = _read_uleb128(data, p)
                elif kind == 0x01:
                    p += 1
                    p = _skip_limits(data, p)
                elif kind == 0x02:
                    p = _skip_limits(data, p)
                elif kind == 0x03:
                    p += 2
                elif kind == 0x04:
                    p += 1
                    _, p = _read_uleb128(data, p)
        elif section_id == 7:  # Export section
            count, p = _read_uleb128(data, idx)
            for _ in range(count):
                exp_name, p = _read_wasm_string(data, p)
                kind = data[p]
                p += 1
                _, p = _read_uleb128(data, p)
                if kind != 0x00:
                    continue
                if exp_name == "main":
                    has_main_export = True
                elif exp_name.startswith("__wbgt_"):
                    raw_path = exp_name[len("__wbgt__") :] if exp_name.startswith("__wbgt__") else exp_name[len("__wbgt_") :]
                    canonical = raw_path
                    if canonical.startswith(f"{crate_name}::"):
                        canonical = canonical[len(crate_name) + 2 :]
                    crate_qual = (
                        raw_path
                        if raw_path.startswith(f"{crate_name}::")
                        else f"{crate_name}::{canonical}"
                    )
                    wbgt_tests.append(
                        DiscoveredTest(
                            canonical_name=canonical,
                            crate_qualified_name=crate_qual,
                            export_name=exp_name,
                            ignored=False,
                        )
                    )
        elif section_id == 11:  # Data section
            count, p = _read_uleb128(data, idx)
            for _ in range(count):
                mode, p = _read_uleb128(data, p)
                if mode == 0:
                    offset, p = _parse_const_i32_expr(data, p)
                    seg_len, p = _read_uleb128(data, p)
                    if offset is not None:
                        active_segments.append((offset, data[p : p + seg_len]))
                    p += seg_len
                elif mode == 1:
                    seg_len, p = _read_uleb128(data, p)
                    p += seg_len
                elif mode == 2:
                    _, p = _read_uleb128(data, p)
                    offset, p = _parse_const_i32_expr(data, p)
                    seg_len, p = _read_uleb128(data, p)
                    if offset is not None:
                        active_segments.append((offset, data[p : p + seg_len]))
                    p += seg_len

        idx = section_end

    unit_tests = _extract_libtest_descriptors(active_segments, crate_name) if has_main_export else []
    return WasmBinarySummary(
        crate_name=crate_name,
        has_wbindgen_imports=has_wbindgen_imports,
        has_main_export=has_main_export,
        wbgt_tests=wbgt_tests,
        unit_tests=unit_tests,
    )


def _pattern_matches(test: DiscoveredTest, pattern: str, exact: bool) -> bool:
    if exact:
        return pattern == test.canonical_name or pattern == test.crate_qualified_name
    return pattern in test.canonical_name or pattern in test.crate_qualified_name


def filter_tests(tests: list[DiscoveredTest], opts: TestRunnerOptions) -> list[DiscoveredTest]:
    selected: list[DiscoveredTest] = []
    for test in tests:
        if opts.only_ignored and test.export_name is None and not test.ignored:
            continue
        if any(_pattern_matches(test, skip, opts.exact) for skip in opts.skips):
            continue
        if opts.filters and not any(_pattern_matches(test, flt, opts.exact) for flt in opts.filters):
            continue
        selected.append(test)
    return selected


def raise_nofile_limit() -> None:
    if resource is None:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (OSError, ValueError):
        pass


def print_runner_help() -> None:
    print(
        "Usage: wasm_test_runner.py <path-to-test.wasm> [OPTIONS] [FILTER...]\n\n"
        "Options:\n"
        "  --list                 List all discovered tests in libtest format\n"
        "  --exact                Exactly match filter and skip names\n"
        "  --skip <PATTERN>       Skip tests whose names match PATTERN (repeatable)\n"
        "  --test-threads <N>     Compatibility flag for Cargo test thread count\n"
        "  --ignored              Run only ignored tests\n"
        "  --include-ignored      Run both ignored and non-ignored tests\n"
        "  --format <FMT>         Configure output format (pretty|terse|json)\n"
        "  -q, --quiet            Display minimal output\n"
        "  --nocapture            Compatibility flag for uncaptured console output"
    )


def _active_test_pools(
    summary: WasmBinarySummary,
    opts: TestRunnerOptions,
) -> tuple[list[DiscoveredTest], list[DiscoveredTest]]:
    """Selects which test pools apply for the binary (`wbgt_tests` and/or `unit_tests`)."""
    if summary.wbgt_tests:
        # math_explorer_gui's #[test] pool contains desktop host-filesystem kittest suites,
        # whereas library crates (oxidize_core, egui_plot) run both browser and #[test] suites.
        include_unit_pool = summary.crate_name != "math_explorer_gui"
        unit_pool = summary.unit_tests if include_unit_pool else []
        return summary.wbgt_tests, unit_pool
    return [], summary.unit_tests


def _launch_chromium_suite(out_dir: Path) -> tuple[str | None, str, list[str], list[str]]:
    try:
        from playwright.sync_api import Error as PlaywrightError, sync_playwright
    except ImportError:
        print(
            "Error: Python 'playwright' package is not installed. "
            "Install with: python3 -m pip install --user playwright && "
            "python3 -m playwright install --with-deps chromium",
            file=sys.stderr,
        )
        return None, "", [], []

    handler = functools.partial(WasmTestRequestHandler, directory=str(out_dir))
    server = ReusableThreadingTCPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    csp_errors: list[str] = []
    page_errors: list[str] = []
    status: str | None = None
    rendered_output = ""
    try:
        url = f"http://127.0.0.1:{port}/index.html"
        for attempt in range(3):
            csp_errors.clear()
            page_errors.clear()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        headless=True,
                        args=CHROMIUM_LAUNCH_ARGS,
                    )
                    try:
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
                    finally:
                        browser.close()
                break
            except PlaywrightError as exc:
                if attempt < 2 and any(
                    t in str(exc) for t in ("Page crashed", "Target crashed", "net::ERR_ABORTED")
                ):
                    continue
                raise
    finally:
        server.shutdown()
        server.server_close()

    return status, rendered_output, csp_errors, page_errors


def main() -> int:
    raise_nofile_limit()
    if len(sys.argv) < 2:
        print("Usage: wasm_test_runner.py <path-to-test.wasm> [test-args...]", file=sys.stderr)
        return 2

    if sys.argv[1] in ("-h", "--help"):
        print_runner_help()
        return 0

    wasm_path = Path(sys.argv[1]).resolve()
    if not wasm_path.is_file():
        print(f"Error: WASM test binary not found: {wasm_path}", file=sys.stderr)
        return 2

    try:
        opts = parse_cargo_test_args(sys.argv[2:])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if opts.show_help:
        print_runner_help()
        return 0

    summary = inspect_wasm_binary(wasm_path)
    wbgt_pool, unit_pool = _active_test_pools(summary, opts)
    selected_wbgt = filter_tests(wbgt_pool, opts)
    selected_unit = filter_tests(unit_pool, opts)
    all_selected = [*selected_wbgt, *selected_unit]
    total_pool_count = len(wbgt_pool) + len(unit_pool)

    if opts.list_only:
        for test in all_selected:
            print(f"{test.canonical_name}: test")
        if opts.output_format != "terse":
            if all_selected:
                print()
            print(f"{len(all_selected)} tests, 0 benchmarks")
        return 0

    if (opts.filters or opts.skips) and not all_selected:
        print(
            f"error: 0 tests matched filter {opts.filters!r} (skip={opts.skips!r}, exact={opts.exact}) "
            f"in {wasm_path.name} ({total_pool_count} available tests)",
            file=sys.stderr,
        )
        return 1

    if not all_selected:
        if not opts.quiet:
            print(f"\nrunning 0 tests in {wasm_path.name}\ntest result: ok. 0 passed; 0 failed\n")
        return 0

    mimetypes.add_type("application/wasm", ".wasm")
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("application/json", ".json")

    with tempfile.TemporaryDirectory(prefix="oxidize_wasm_tests_") as tmp_dir:
        out_dir = Path(tmp_dir)
        use_wbindgen = bool(selected_wbgt) or summary.has_wbindgen_imports

        if use_wbindgen:
            wasm_bindgen_bin = shutil.which("wasm-bindgen")
            if not wasm_bindgen_bin:
                print(
                    "Error: 'wasm-bindgen' CLI not found on PATH. Install wasm-bindgen-cli.",
                    file=sys.stderr,
                )
                return 2
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
            js_text = js_text.replace(
                "wasm.__wbindgen_start();",
                'if (typeof wasm.__wbindgen_init_externref_table === "function") { wasm.__wbindgen_init_externref_table(); }',
            )
            js_wrapper.write_text(js_text, encoding="utf-8")
            mode = "wbg_and_unit"
        else:
            shutil.copyfile(wasm_path, out_dir / "raw_unit_test.wasm")
            mode = "raw_unit"

        runner_cfg = {
            "mode": mode,
            "run_wbg": bool(selected_wbgt),
            "selected_wbgt_exports": [t.export_name for t in selected_wbgt if t.export_name],
            "filtered_out_wbgt": len(wbgt_pool) - len(selected_wbgt),
            "include_ignored": opts.include_ignored,
            "run_unit": bool(selected_unit),
            "unit_test_count": len(selected_unit),
            "binary_label": wasm_path.name,
        }
        (out_dir / "runner_config.json").write_text(json.dumps(runner_cfg), encoding="utf-8")
        (out_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
        (out_dir / "runner.js").write_text(RUNNER_JS, encoding="utf-8")

        status, rendered_output, csp_errors, page_errors = _launch_chromium_suite(out_dir)

        if rendered_output.strip() and not opts.quiet:
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
    try:
        sys.exit(main())
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
