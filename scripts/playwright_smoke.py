#!/usr/bin/env python3
"""Headless Chromium Playwright smoke and fault-injection gate for Math Explorer."""

from __future__ import annotations

import argparse
import functools
import gzip
import html.parser
import http.server
import mimetypes
import os
import re
try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
import socketserver
import subprocess
import sys
import threading
import time
import tomllib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Playwright,
    expect,
    sync_playwright,
)

CONTAINER_SAFE_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
]

WEBGL_SWIFTSHADER_ARGS = [
    *CONTAINER_SAFE_ARGS,
    "--enable-webgl",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
]

WEBGL_DISABLED_ARGS = [
    *CONTAINER_SAFE_ARGS,
    "--disable-webgl",
    "--disable-webgl2",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
]


def raise_nofile_limit() -> None:
    if resource is None:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (OSError, ValueError):
        pass


@dataclass(frozen=True)
class FaultExpectation:
    category: str
    title: str
    message_substr: str
    alert_substr: str


EXPECTED_FAULTS: dict[str, FaultExpectation] = {
    "network": FaultExpectation(
        category="network",
        title="Network Error Loading Math Explorer",
        message_substr="could not finish downloading",
        alert_substr="Network error loading Math Explorer",
    ),
    "compile": FaultExpectation(
        category="compile",
        title="WebAssembly Startup Error",
        message_substr="failed to compile, instantiate, or start",
        alert_substr="WebAssembly startup error",
    ),
    "timeout": FaultExpectation(
        category="timeout",
        title="Startup Timed Out",
        message_substr="before startup could complete",
        alert_substr="timed out",
    ),
    "webgl": FaultExpectation(
        category="webgl",
        title="WebGL Initialization Failed",
        message_substr="unable to initialize a WebGL rendering context",
        alert_substr="WebGL initialization failed",
    ),
}


class MetaCspParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.csp_values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if attr_map.get("http-equiv", "").lower() == "content-security-policy":
            self.csp_values.append(attr_map.get("content", ""))


class ReusableThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128


STARTUP_TELEMETRY_PROBE_JS = b"""(function () {
    const trace = {
        bytesTexts: [],
        politeAnnouncements: [],
        sawHundredPercent: false,
        sawDownloadCompleteAnnouncement: false,
    };
    window.__mathExplorerSmokeTrace = trace;

    function recordDomSnapshot() {
        const bytesEl = document.getElementById('loading_bytes');
        if (bytesEl) {
            const txt = (bytesEl.textContent || '').trim();
            if (txt && trace.bytesTexts[trace.bytesTexts.length - 1] !== txt) {
                trace.bytesTexts.push(txt);
            }
            if (txt.indexOf('(100%)') !== -1) {
                trace.sawHundredPercent = true;
            }
        }
        const progressEl = document.getElementById('loading_progress');
        if (progressEl) {
            const vText = progressEl.getAttribute('aria-valuetext') || '';
            if (/^100%\\b/.test(vText)) {
                trace.sawHundredPercent = true;
            }
        }
        const slot0 = document.getElementById('aria-live-polite-slot-0');
        const slot1 = document.getElementById('aria-live-polite-slot-1');
        const s0 = slot0 && slot0.textContent ? slot0.textContent.trim() : '';
        const s1 = slot1 && slot1.textContent ? slot1.textContent.trim() : '';
        const activePolite = s0 || s1;
        if (
            activePolite &&
            trace.politeAnnouncements[trace.politeAnnouncements.length - 1] !== activePolite
        ) {
            trace.politeAnnouncements.push(activePolite);
            if (activePolite.indexOf('Download complete.') !== -1) {
                trace.sawDownloadCompleteAnnouncement = true;
            }
        }
    }

    const attachObserver = () => {
        const root = document.documentElement || document;
        if (!root || typeof MutationObserver === 'undefined') {
            return;
        }
        recordDomSnapshot();
        const obs = new MutationObserver(() => {
            recordDomSnapshot();
        });
        obs.observe(root, {
            subtree: true,
            childList: true,
            characterData: true,
            attributes: true,
            attributeFilter: ['value', 'max', 'aria-valuenow', 'aria-valuetext'],
        });
    };

    if (document.documentElement) {
        attachObserver();
    } else {
        document.addEventListener('DOMContentLoaded', attachObserver, { once: true });
    }
})();
"""


class CspDistRequestHandler(http.server.SimpleHTTPRequestHandler):
    csp_header: str = ""
    truncated_br_chunk: bytes = b""

    def _referer_query_param(self, key: str) -> str:
        referer = self.headers.get("Referer", "")
        if not referer:
            return ""
        parsed = urllib.parse.urlsplit(referer)
        vals = urllib.parse.parse_qs(parsed.query).get(key)
        return vals[0] if vals else ""

    def _serve_custom_wasm_mode(self, mode: str) -> bool:
        dist_root = Path(self.directory)
        wasm_path = dist_root / "math_explorer_gui_bg.wasm"
        br_path = dist_root / "math_explorer_gui_bg.wasm.br"
        gz_path = dist_root / "math_explorer_gui_bg.wasm.gz"

        if mode == "drop_identity":
            raw = wasm_path.read_bytes()
            self.protocol_version = "HTTP/1.1"
            self.send_response(200)
            self.send_header("Content-Type", "application/wasm")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw[:131072])
            self.wfile.flush()
            self.close_connection = True
            return True

        if mode == "drop_brotli":
            br = br_path.read_bytes()
            self.protocol_version = "HTTP/1.1"
            self.send_response(200)
            self.send_header("Content-Type", "application/wasm")
            self.send_header("Content-Encoding", "br")
            self.send_header("Content-Length", str(len(br)))
            self.end_headers()
            self.wfile.write(br[:65536])
            self.wfile.flush()
            self.close_connection = True
            return True

        if mode == "truncated_chunked":
            raw = wasm_path.read_bytes()
            self.protocol_version = "HTTP/1.1"
            self.send_response(200)
            self.send_header("Content-Type", "application/wasm")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in (raw[:65536], raw[65536:131072]):
                self.wfile.write(f"{len(part):x}\r\n".encode("ascii") + part + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            self.close_connection = True
            return True

        if mode == "truncated_brotli_chunked":
            part = self.truncated_br_chunk
            self.protocol_version = "HTTP/1.1"
            self.send_response(200)
            self.send_header("Content-Type", "application/wasm")
            self.send_header("Content-Encoding", "br")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(f"{len(part):x}\r\n".encode("ascii") + part + b"\r\n0\r\n\r\n")
            self.wfile.flush()
            self.close_connection = True
            return True

        if mode in ("full_br", "full_gz", "chunked_br", "chunked_gz"):
            is_br = "br" in mode
            is_chunked = mode.startswith("chunked_")
            payload = (br_path if is_br else gz_path).read_bytes()
            encoding = "br" if is_br else "gzip"
            slices = 4 if is_chunked else 8
            step = max(1, (len(payload) + slices - 1) // slices)

            self.protocol_version = "HTTP/1.1"
            self.send_response(200)
            self.send_header("Content-Type", "application/wasm")
            self.send_header("Content-Encoding", encoding)
            if is_chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

            for offset in range(0, len(payload), step):
                chunk = payload[offset : offset + step]
                if is_chunked:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n")
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
                if offset + step < len(payload):
                    time.sleep(0.03)
            if is_chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                self.close_connection = True
            return True

        return False

    def do_GET(self) -> None:
        clean_path = self.path.split("?")[0]
        referer = self.headers.get("Referer", "")
        if "stall_timeout_ms=" in referer and clean_path.endswith(".wasm"):
            time.sleep(0.9)
            try:
                body = b"\x00asm\x01\x00\x00\x00"
                self.send_response(200)
                self.send_header("Content-Type", "application/wasm")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass
            return
        if clean_path.endswith(".wasm"):
            wasm_mode = self._referer_query_param("wasm_mode")
            if wasm_mode:
                try:
                    if self._serve_custom_wasm_mode(wasm_mode):
                        return
                except OSError:
                    return
        try:
            super().do_GET()
        except OSError:
            pass

    def end_headers(self) -> None:
        self.send_header("Content-Security-Policy", self.csp_header)
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        return


def verify_stamped_dist_artifacts(dist_dir: Path) -> tuple[int, int, int]:
    wasm_path = dist_dir / "math_explorer_gui_bg.wasm"
    br_path = dist_dir / "math_explorer_gui_bg.wasm.br"
    gz_path = dist_dir / "math_explorer_gui_bg.wasm.gz"
    bootstrap_path = dist_dir / "bootstrap.js"
    bootstrap_gz = dist_dir / "bootstrap.js.gz"
    bootstrap_br = dist_dir / "bootstrap.js.br"

    for artifact in (wasm_path, br_path, gz_path, bootstrap_path, bootstrap_gz, bootstrap_br):
        if not artifact.is_file():
            raise RuntimeError(f"Expected staged artifact missing at {artifact}")

    wasm_size = wasm_path.stat().st_size
    br_size = br_path.stat().st_size
    gz_size = gz_path.stat().st_size
    bootstrap_bytes = bootstrap_path.read_bytes()
    bootstrap_text = bootstrap_bytes.decode("utf-8")

    for const_name, expected_val in (
        ("WASM_SOURCE_SIZE", wasm_size),
        ("WASM_BROTLI_SIZE", br_size),
        ("WASM_GZIP_SIZE", gz_size),
    ):
        match = re.search(rf"^const {const_name} = (\d+);$", bootstrap_text, flags=re.MULTILINE)
        if not match or int(match.group(1)) != expected_val:
            raise AssertionError(
                f"Expected {const_name} = {expected_val} in {bootstrap_path}, got {match.group(0) if match else None!r}"
            )

    if gzip.decompress(bootstrap_gz.read_bytes()) != bootstrap_bytes:
        raise AssertionError("bootstrap.js.gz does not match stamped bootstrap.js")
    br_decoded = subprocess.run(
        ["brotli", "-d", "-c", str(bootstrap_br)], capture_output=True, check=True
    ).stdout
    if br_decoded != bootstrap_bytes:
        raise AssertionError("bootstrap.js.br does not match stamped bootstrap.js")

    print(
        f"[PASS] dist_stamping: source={wasm_size}B, brotli={br_size}B, gzip={gz_size}B (.gz/.br bootstrap match)"
    )
    return wasm_size, br_size, gz_size


def install_telemetry_probe(page: Page, dist_dir: Path) -> None:
    reload_bytes = STARTUP_TELEMETRY_PROBE_JS + (dist_dir / "reload.js").read_bytes()
    page.route(
        "**/reload.js*",
        lambda route: route.fulfill(
            status=200,
            content_type="text/javascript",
            body=reload_bytes,
        ),
    )


def run_truncated_stream_check(
    browser: Browser,
    base_url: str,
    dist_dir: Path,
    scenario: str,
    wasm_mode: str,
    screenshot_path: Path | None = None,
) -> None:
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        csp_violations: list[str] = []
        attach_csp_observer(page, csp_violations)
        install_telemetry_probe(page, dist_dir)

        page.goto(f"{base_url}/?wasm_mode={wasm_mode}", wait_until="domcontentloaded")
        verify_error_recovery(page, scenario, EXPECTED_FAULTS["network"])
        page.wait_for_timeout(40)

        trace = page.evaluate("() => window.__mathExplorerSmokeTrace")
        if trace["sawHundredPercent"]:
            raise AssertionError(
                f"[{scenario}] Truncated stream flashed fake 100% progress: {trace['bytesTexts']}"
            )
        if trace["sawDownloadCompleteAnnouncement"]:
            raise AssertionError(
                f"[{scenario}] Truncated stream announced download completion: {trace['politeAnnouncements']}"
            )
        assert_zero_csp_violations(scenario, csp_violations)

        if screenshot_path is not None:
            page.screenshot(path=str(screenshot_path))
        print(
            f"[PASS] {scenario}: category='network', 0 fake 100% banners, 0 completion announcements"
        )
    finally:
        context.close()


def extract_reported_totals_mb(bytes_texts: list[str]) -> list[float]:
    totals: list[float] = []
    for entry in bytes_texts:
        slash_match = re.search(r"/\s*([0-9]+\.[0-9]+)\s*MB\b", entry)
        if slash_match:
            totals.append(float(slash_match.group(1)))
            continue
        loaded_match = re.search(r"^([0-9]+\.[0-9]+)\s*MB\s+loaded\b", entry)
        if loaded_match:
            totals.append(float(loaded_match.group(1)))
    return totals


def run_compressed_progress_and_aria_check(
    browser: Browser,
    base_url: str,
    dist_dir: Path,
    scenario: str,
    wasm_mode: str,
    expected_wire_bytes: int,
    uncompressed_bytes: int,
    screenshot_path: Path | None = None,
) -> None:
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        csp_violations: list[str] = []
        console_errors: list[str] = []
        page_errors: list[str] = []
        attach_csp_observer(page, csp_violations)
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        install_telemetry_probe(page, dist_dir)

        page.goto(f"{base_url}/?wasm_mode={wasm_mode}", wait_until="domcontentloaded")
        expect(page.locator("#loading_indicator")).to_be_hidden(timeout=20_000)
        expect(page.locator("#error_container")).to_be_hidden()
        expect(page.locator("#the_canvas_id")).to_be_visible()
        expect(page.locator("#the_canvas_id")).to_be_focused()

        trace = page.evaluate("() => window.__mathExplorerSmokeTrace")
        expected_mb = round(expected_wire_bytes / 1_000_000.0, 2)
        uncompressed_mb = round(uncompressed_bytes / 1_000_000.0, 2)
        reported_totals = extract_reported_totals_mb(trace["bytesTexts"])
        if not reported_totals:
            raise AssertionError(
                f"[{scenario}] Expected MB totals in #loading_bytes, observed: {trace['bytesTexts']}"
            )
        for observed_mb in reported_totals:
            if abs(observed_mb - expected_mb) > 0.01:
                raise AssertionError(
                    f"[{scenario}] Expected wire total {expected_mb:.2f} MB (±0.01 MB), got {observed_mb:.2f} MB in {trace['bytesTexts']}"
                )
            if abs(observed_mb - uncompressed_mb) <= 0.05:
                raise AssertionError(
                    f"[{scenario}] Inflated decoded total {observed_mb:.2f} MB reported instead of wire {expected_mb:.2f} MB"
                )

        polite_log: list[str] = trace["politeAnnouncements"]
        required_sequence = [
            "WebAssembly download 25% complete.",
            "WebAssembly download 50% complete.",
            "WebAssembly download 75% complete.",
            "Download complete. Compiling WebAssembly module.",
        ]
        indices: list[int] = []
        for milestone in required_sequence:
            if milestone not in polite_log:
                raise AssertionError(
                    f"[{scenario}] Missing polite ARIA milestone {milestone!r} in {polite_log}"
                )
            indices.append(polite_log.index(milestone))
        if indices != sorted(indices):
            raise AssertionError(
                f"[{scenario}] Polite ARIA milestones announced out of order: {polite_log}"
            )

        assert_zero_csp_violations(scenario, csp_violations)
        if console_errors:
            raise AssertionError(f"[{scenario}] Console errors detected: {console_errors}")
        if page_errors:
            raise AssertionError(f"[{scenario}] Uncaught page errors detected: {page_errors}")

        if screenshot_path is not None:
            page.screenshot(path=str(screenshot_path))
        print(
            f"[PASS] {scenario}: wire_total={reported_totals[-1]:.2f} MB (expected {expected_mb:.2f} MB), ARIA 25%->50%->75%->compile verified"
        )
    finally:
        context.close()


def run_internal_error_and_fallback_compile_checks(
    browser: Browser,
    base_url: str,
    screenshot_path: Path | None = None,
) -> None:
    internal_messages = (
        "out of memory",
        "too much recursion",
        "allocation size overflow",
    )

    def _make_loader_fulfiller(payload: bytes) -> Callable[[Route], None]:
        def _fulfill(route: Route) -> None:
            route.fulfill(
                status=200,
                content_type="text/javascript",
                body=payload,
            )

        return _fulfill

    captured_shot = False
    for trap_msg in internal_messages:
        loader_stub = (
            "export default async function init() {\n"
            f"    const err = new Error({trap_msg!r});\n"
            "    err.name = 'InternalError';\n"
            "    throw err;\n"
            "}\n"
        ).encode("utf-8")
        fulfill_loader = _make_loader_fulfiller(loader_stub)

        for mode_label, use_fallback in (("reload", False), ("bootstrap_fallback", True)):
            ctx = browser.new_context(viewport={"width": 1280, "height": 800})
            try:
                pg = ctx.new_page()
                csp_violations: list[str] = []
                attach_csp_observer(pg, csp_violations)
                if use_fallback:
                    pg.route(
                        "**/reload.js*",
                        lambda route: route.fulfill(
                            status=200,
                            content_type="text/javascript",
                            body=b"// fallback handler check without reload.js\n",
                        ),
                    )
                pg.route("**/math_explorer_gui.js*", fulfill_loader)
                pg.goto(f"{base_url}/", wait_until="domcontentloaded")
                err_box = pg.locator("#error_container")
                expect(err_box).to_be_visible(timeout=12_000)
                expect(err_box).to_have_attribute("data-error-active", "true")
                expect(err_box).to_have_attribute("data-error-category", "compile")
                expect(pg.locator("#error_title")).to_have_text("WebAssembly Startup Error")
                expect(pg.locator("#webgl_reload_button")).to_be_focused()
                if not use_fallback:
                    expect(pg.locator("#aria-live-assertive")).to_contain_text(
                        "WebAssembly startup error"
                    )
                assert_zero_csp_violations(f"internal_error_{mode_label}_{trap_msg}", csp_violations)
                if screenshot_path and not captured_shot and not use_fallback:
                    pg.screenshot(path=str(screenshot_path), full_page=True)
                    captured_shot = True
            finally:
                ctx.close()

    # Also verify bootstrap.js fallback classifies complete invalid fault_corrupt_wasm as compile
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        pg = ctx.new_page()
        csp_violations = []
        attach_csp_observer(pg, csp_violations)
        pg.route(
            "**/reload.js*",
            lambda route: route.fulfill(
                status=200,
                content_type="text/javascript",
                body=b"// fallback handler corrupt wasm check\n",
            ),
        )
        pg.route(
            "**/math_explorer_gui_bg.wasm*",
            lambda route: route.fulfill(
                status=200,
                content_type="application/wasm",
                body=b"corrupted-wasm-payload-bytes",
            ),
        )
        pg.goto(f"{base_url}/", wait_until="domcontentloaded")
        err_box = pg.locator("#error_container")
        expect(err_box).to_be_visible(timeout=12_000)
        expect(err_box).to_have_attribute("data-error-category", "compile")
        expect(pg.locator("#error_title")).to_have_text("WebAssembly Startup Error")
        expect(pg.locator("#webgl_reload_button")).to_be_focused()
        assert_zero_csp_violations("fallback_corrupt_wasm", csp_violations)
    finally:
        ctx.close()

    print(
        "[PASS] internal_error_and_fallback_compile: Firefox InternalError traps and corrupt WASM route to 'compile' in reload.js and bootstrap.js"
    )


def load_and_verify_production_csp(trunk_toml: Path, dist_dir: Path) -> str:
    config = tomllib.loads(trunk_toml.read_text(encoding="utf-8"))
    csp = config.get("serve", {}).get("headers", {}).get("Content-Security-Policy", "").strip()
    if not csp:
        raise RuntimeError(f"Missing [serve.headers].Content-Security-Policy in {trunk_toml}")

    index_html = dist_dir / "index.html"
    if not index_html.is_file():
        raise RuntimeError(f"Built index.html not found at {index_html}")

    parser = MetaCspParser()
    parser.feed(index_html.read_text(encoding="utf-8"))
    if not parser.csp_values:
        raise RuntimeError(f"No <meta http-equiv='Content-Security-Policy'> found in {index_html}")
    if parser.csp_values[0].strip() != csp:
        raise RuntimeError(
            f"CSP drift between {trunk_toml} ({csp!r}) and {index_html} ({parser.csp_values[0]!r})"
        )
    return csp


def configure_mimetypes() -> None:
    mimetypes.add_type("application/wasm", ".wasm")
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("application/json", ".json")
    mimetypes.add_type("image/svg+xml", ".svg")
    mimetypes.add_type("application/manifest+json", ".webmanifest")


def attach_csp_observer(page: Page, csp_violations: list[str]) -> None:
    def on_console(msg: object) -> None:
        msg_type = getattr(msg, "type", "")
        msg_text = getattr(msg, "text", "")
        if msg_type == "error" and "Content Security Policy" in msg_text:
            csp_violations.append(msg_text)

    page.on("console", on_console)


class BrowserPool:
    def __init__(self, playwright: Playwright, args: list[str]) -> None:
        self._playwright = playwright
        self._args = args
        self.browser: Browser = playwright.chromium.launch(headless=True, args=args)

    def reset(self) -> None:
        try:
            self.browser.close()
        except Exception:
            pass
        time.sleep(0.2)
        self.browser = self._playwright.chromium.launch(headless=True, args=self._args)

    def close(self) -> None:
        try:
            self.browser.close()
        except Exception:
            pass


def run_with_crash_retry(pool: BrowserPool, label: str, fn: object) -> None:
    if not callable(fn):
        return
    for attempt in range(4):
        try:
            fn(pool.browser)
            return
        except (TransientBrowserCrashError, PlaywrightError, AssertionError) as exc:
            if attempt < 3 and (
                isinstance(exc, TransientBrowserCrashError) or is_playwright_crash(exc)
            ):
                print(
                    f"[RETRY {attempt + 1}/3] {label}: recovering from headless Chromium crash ({exc})",
                    file=sys.stderr,
                )
                pool.reset()
                continue
            raise


def is_playwright_crash(exc: BaseException) -> bool:
    msg = str(exc)
    return any(
        token in msg
        for token in (
            "Page crashed",
            "Target crashed",
            "Target page, context or browser has been closed",
        )
    ) or (isinstance(exc, PlaywrightError) and "net::ERR_ABORTED" in msg)


class TransientBrowserCrashError(RuntimeError):
    """Raised when headless Chromium's Renderer or NetworkService crashes in a container."""
def assert_zero_csp_violations(scenario: str, csp_violations: list[str]) -> None:
    if csp_violations:
        raise AssertionError(f"[{scenario}] CSP violations detected: {csp_violations}")


def run_happy_path_check(
    browser: Browser,
    base_url: str,
    expected_csp: str,
    screenshot_dir: Path | None,
) -> None:
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        console_errors: list[str] = []
        page_errors: list[str] = []
        csp_violations: list[str] = []
        failed_requests: list[str] = []
        status_by_url: dict[str, int] = {}

        attach_csp_observer(page, csp_violations)
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on(
            "requestfailed",
            lambda req: failed_requests.append(f"{req.method} {req.url}: {req.failure}"),
        )
        page.on("response", lambda resp: status_by_url.__setitem__(resp.url, resp.status))

        doc_resp = page.goto(f"{base_url}/", wait_until="domcontentloaded")
        if doc_resp is None:
            raise AssertionError("[happy_path] Navigation returned no main document response")
        if doc_resp.status != 200:
            raise AssertionError(f"[happy_path] Expected HTTP 200 on /, got {doc_resp.status}")
        actual_csp = doc_resp.headers.get("content-security-policy", "")
        if actual_csp != expected_csp:
            raise AssertionError(
                f"[happy_path] Response CSP header mismatch: {actual_csp!r} != {expected_csp!r}"
            )

        expect(page.locator("#loading_indicator")).to_be_hidden(timeout=20_000)
        if any("net::ERR_ABORTED" in req for req in failed_requests):
            raise TransientBrowserCrashError(
                f"[happy_path] Loopback request aborted by Chromium NetworkService: {failed_requests}"
            )
        expect(page.locator("#loading_indicator")).to_have_count(0, timeout=2_000)
        expect(page.locator("html")).to_have_attribute("data-boot-initialized", "true")
        if page.locator("html").get_attribute("data-boot-failed") is not None:
            raise AssertionError("[happy_path] html[data-boot-failed] must not be set on clean boot")
        expect(page.locator("#error_container")).to_be_hidden()

        canvas = page.locator("#the_canvas_id")
        expect(canvas).to_be_visible()
        expect(canvas).to_be_focused()

        box = canvas.bounding_box()
        if not box or box["width"] < 800 or box["height"] < 500:
            raise AssertionError(f"[happy_path] Unexpected WebGL canvas dimensions: {box}")

        # Exercise interactive pointer movement on the rendered WebGL canvas.
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        page.wait_for_timeout(150)
        expect(canvas).to_be_focused()

        required_assets = [
            "/reload.js",
            "/bootstrap.js",
            "/initializer.js",
            "/math_explorer_gui.js",
            "/math_explorer_gui_bg.wasm",
        ]
        for suffix in required_assets:
            matched = [code for url, code in status_by_url.items() if url.split("?")[0].endswith(suffix)]
            if not matched or any(code != 200 for code in matched):
                raise AssertionError(
                    f"[happy_path] Expected HTTP 200 for {suffix}, observed: {matched}"
                )

        non_200 = {url: code for url, code in status_by_url.items() if code != 200}
        if non_200:
            raise AssertionError(f"[happy_path] Non-200 asset responses observed: {non_200}")
        if failed_requests:
            raise AssertionError(f"[happy_path] Failed network requests: {failed_requests}")
        assert_zero_csp_violations("happy_path", csp_violations)
        if console_errors:
            raise AssertionError(f"[happy_path] Console errors detected: {console_errors}")
        if page_errors:
            raise AssertionError(f"[happy_path] Uncaught page errors detected: {page_errors}")

        if screenshot_dir is not None:
            page.screenshot(path=str(screenshot_dir / "smoke_healthy_webgl_boot.png"))
        print("[PASS] happy_path: HTTP 200 assets, strict CSP, WebGL canvas rendered and focused")
    finally:
        context.close()


def verify_error_recovery(
    page: Page,
    scenario: str,
    expectation: FaultExpectation,
    timeout_ms: int = 12_000,
) -> None:
    error_container = page.locator("#error_container")
    expect(error_container).to_be_visible(timeout=timeout_ms)
    expect(error_container).to_have_attribute("data-error-active", "true")
    expect(error_container).to_have_attribute("data-error-category", expectation.category)
    expect(error_container).to_have_attribute("role", "alert")
    expect(page.locator("html")).to_have_attribute("data-boot-initialized", "true")
    expect(page.locator("html")).to_have_attribute("data-boot-failed", "true")
    expect(page.locator("#loading_indicator")).to_be_hidden()
    expect(page.locator("#the_canvas_id")).to_be_hidden()

    expect(page.locator("#error_title")).to_have_text(expectation.title)
    expect(page.locator("#error_message")).to_contain_text(expectation.message_substr)
    expect(page.locator("#aria-live-assertive")).to_contain_text(expectation.alert_substr)

    if expectation.category == "webgl":
        expect(page.locator("#webgl_instructions")).to_be_visible()
        expect(page.locator("#error_details")).to_be_visible()

    reload_btn = page.locator("#webgl_reload_button")
    expect(reload_btn).to_be_visible()
    expect(reload_btn).to_be_focused()


def run_fault_scenario(
    browser: Browser,
    base_url: str,
    scenario: str,
    category: str,
    url_suffix: str,
    configure_routes: object,
    screenshot_path: Path | None = None,
) -> None:
    context: BrowserContext = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        csp_violations: list[str] = []
        aborted_unrouted: list[str] = []
        attach_csp_observer(page, csp_violations)
        page.on(
            "requestfailed",
            lambda req: aborted_unrouted.append(f"{req.url}: {req.failure}")
            if req.failure == "net::ERR_ABORTED"
            and any(
                req.url.split("?")[0].endswith(s)
                for s in ("/reload.js", "/initializer.js")
                + (("/bootstrap.js",) if scenario != "fault_broken_bootstrap_script" else ())
                + (("/math_explorer_gui.js",) if scenario != "fault_broken_loader_script" else ())
                + (
                    ("/math_explorer_gui_bg.wasm",)
                    if scenario not in ("fault_missing_wasm", "fault_corrupt_wasm", "fault_stall_timeout")
                    else ()
                )
            )
            else None,
        )
        if callable(configure_routes):
            configure_routes(page)

        page.goto(f"{base_url}{url_suffix}", wait_until="domcontentloaded")
        try:
            verify_error_recovery(page, scenario, EXPECTED_FAULTS[category])
        except AssertionError as exc:
            if aborted_unrouted:
                raise TransientBrowserCrashError(
                    f"[{scenario}] Unrouted script aborted by Chromium NetworkService: {aborted_unrouted}"
                ) from exc
            raise
        assert_zero_csp_violations(scenario, csp_violations)

        if screenshot_path is not None:
            page.screenshot(path=str(screenshot_path))
        print(f"[PASS] {scenario}: category={category!r}, reload button focused, zero CSP violations")
    finally:
        context.close()


def break_recovery_scripts(page: Page) -> None:
    page.route(
        "**/reload.js*",
        lambda route: route.fulfill(
            status=200,
            content_type="text/javascript",
            body=b"// simulated broken recovery handler\n",
        ),
    )
    page.route(
        "**/bootstrap.js*",
        lambda route: route.fulfill(
            status=200,
            content_type="text/javascript",
            body=b"// simulated broken bootstrap without error boundary\n",
        ),
    )


def verify_network_regression_detection(
    swiftshader_browser: Browser,
    base_url: str,
) -> None:
    """Confirms that broken recovery logic causes network smoke assertions to fail."""

    def missing_wasm_broken(page: Page) -> None:
        break_recovery_scripts(page)
        page.route("**/math_explorer_gui_bg.wasm*", lambda route: route.fulfill(status=404, body=b"missing"))

    def broken_loader_broken(page: Page) -> None:
        break_recovery_scripts(page)
        page.route("**/math_explorer_gui.js*", lambda route: route.fulfill(status=500, body=b"broken"))

    for label, route_fn, cat in [
        ("regression_missing_wasm", missing_wasm_broken, "network"),
        ("regression_broken_loader", broken_loader_broken, "network"),
    ]:
        ctx = swiftshader_browser.new_context(viewport={"width": 1280, "height": 800})
        try:
            pg = ctx.new_page()
            route_fn(pg)
            pg.goto(f"{base_url}/", wait_until="domcontentloaded")
            failed_as_expected = False
            try:
                verify_error_recovery(pg, label, EXPECTED_FAULTS[cat], timeout_ms=800)
            except AssertionError:
                failed_as_expected = True
            if not failed_as_expected:
                raise RuntimeError(f"Smoke suite failed to catch broken recovery in {label}")
        finally:
            ctx.close()


def verify_disabled_webgl_regression_detection(
    disabled_webgl_browser: Browser,
    base_url: str,
) -> None:
    """Confirms that broken WebGL recovery causes disabled-WebGL smoke assertions to fail."""
    ctx = disabled_webgl_browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        pg = ctx.new_page()
        break_recovery_scripts(pg)
        pg.goto(f"{base_url}/", wait_until="domcontentloaded")
        failed_as_expected = False
        try:
            verify_error_recovery(
                pg, "regression_disabled_webgl", EXPECTED_FAULTS["webgl"], timeout_ms=800
            )
        except AssertionError:
            failed_as_expected = True
        if not failed_as_expected:
            raise RuntimeError("Smoke suite failed to catch broken WebGL recovery")
    finally:
        ctx.close()

    print("[PASS] regression_detection: missing .wasm, broken loader, and disabled WebGL fail when recovery breaks")


def run_parallel_preload_and_sw_http_cache_check(
    browser: Browser,
    dist_dir: Path,
    csp: str,
) -> None:
    origin_hits: list[tuple[str, int]] = []
    origin_lock = threading.Lock()
    revision_state = {"rev": 1}

    class CacheableOriginHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            clean_path = self.path.split("?")[0]
            rel_name = "index.html" if clean_path in ("", "/") else clean_path.lstrip("/")
            target = (dist_dir / rel_name).resolve()
            if not str(target).startswith(str(dist_dir.resolve())) or not target.is_file():
                with origin_lock:
                    origin_hits.append((clean_path, 404))
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Security-Policy", csp)
                self.end_headers()
                self.wfile.write(b"Not Found")
                return

            payload = target.read_bytes()
            rev = revision_state["rev"]
            if clean_path == "/bootstrap.js" and rev > 1:
                payload = payload + f"\n// origin-rev-{rev}\n".encode("utf-8")
            etag = f'"{rel_name}-v{rev}-{len(payload)}"'
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"

            if self.headers.get("If-None-Match") == etag:
                with origin_lock:
                    origin_hits.append((clean_path, 304))
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "public, max-age=600")
                self.send_header("Content-Security-Policy", csp)
                self.end_headers()
                return

            with origin_lock:
                origin_hits.append((clean_path, 200))
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "public, max-age=600")
            self.send_header("Content-Security-Policy", csp)
            self.end_headers()
            try:
                self.wfile.write(payload)
            except OSError:
                pass

        def log_message(self, fmt: str, *args: object) -> None:
            return

    cache_server = ReusableThreadingTCPServer(("127.0.0.1", 0), CacheableOriginHandler)
    cache_url = f"http://127.0.0.1:{cache_server.server_address[1]}"
    threading.Thread(target=cache_server.serve_forever, daemon=True).start()

    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page1 = context.new_page()
        csp_violations: list[str] = []
        preload_warnings: list[str] = []
        pre_sw_requests: list[str] = []
        saw_sw_request = False
        attach_csp_observer(page1, csp_violations)

        def on_console(msg: object) -> None:
            text = getattr(msg, "text", "")
            if "preload" in text.lower() and (
                "not used" in text.lower() or "credentials mode" in text.lower()
            ):
                preload_warnings.append(text)

        def on_request(req: object) -> None:
            nonlocal saw_sw_request
            url_path = urllib.parse.urlsplit(getattr(req, "url", "")).path
            if url_path.endswith("/sw.js"):
                saw_sw_request = True
                return
            if not saw_sw_request:
                pre_sw_requests.append(url_path)

        page1.on("console", on_console)
        page1.on("request", on_request)

        page1.goto(f"{cache_url}/", wait_until="domcontentloaded")
        expect(page1.locator("#loading_indicator")).to_be_hidden(timeout=20_000)
        expect(page1.locator("#the_canvas_id")).to_be_focused()
        assert_zero_csp_violations("cold_parallel_wave", csp_violations)
        if preload_warnings:
            raise AssertionError(
                f"[cold_parallel_wave] Preload credential/reuse warnings detected: {preload_warnings}"
            )

        expected_preboot = {
            "/",
            "/reload.js",
            "/bootstrap.js",
            "/initializer.js",
            "/math_explorer_gui.js",
            "/math_explorer_gui_bg.wasm",
        }
        if len(pre_sw_requests) != 6 or set(pre_sw_requests) != expected_preboot:
            raise AssertionError(
                f"[cold_parallel_wave] Expected 6 pre-boot requests {sorted(expected_preboot)}, got {pre_sw_requests}"
            )

        # Wait for post-boot sw.js registration, precacheShell completion, and client claim.
        page1.wait_for_function(
            """async () => {
                if (!navigator.serviceWorker) return false;
                await navigator.serviceWorker.ready;
                if (!navigator.serviceWorker.controller) return false;
                const cache = await caches.open('math-explorer-shell-v1');
                const keys = (await cache.keys()).map((r) => new URL(r.url).pathname);
                return keys.includes('/index.html') && keys.includes('/bootstrap.js');
            }""",
            timeout=10_000,
        )

        with origin_lock:
            post_install_hits = list(origin_hits)
        for boot_asset in (
            "/",
            "/reload.js",
            "/bootstrap.js",
            "/initializer.js",
            "/math_explorer_gui.js",
            "/math_explorer_gui_bg.wasm",
        ):
            asset_hits = [code for path, code in post_install_hits if path == boot_asset]
            if asset_hits != [200]:
                raise AssertionError(
                    f"[sw_precache_http_reuse] Expected single origin 200 for {boot_asset} across boot+precacheShell, got {asset_hits}"
                )

        # Repeat navigation within max-age must trigger zero 304s and zero origin fetches for cached boot assets.
        with origin_lock:
            origin_hits.clear()

        page2 = context.new_page()
        repeat_statuses: list[tuple[str, int]] = []
        attach_csp_observer(page2, csp_violations)
        page2.on(
            "response",
            lambda resp: repeat_statuses.append(
                (urllib.parse.urlsplit(resp.url).path, resp.status)
            ),
        )
        page2.goto(f"{cache_url}/", wait_until="domcontentloaded")
        expect(page2.locator("#loading_indicator")).to_be_hidden(timeout=20_000)
        expect(page2.locator("#the_canvas_id")).to_be_focused()
        assert_zero_csp_violations("warm_repeat_navigation", csp_violations)

        revalidations_304 = [(p, c) for p, c in repeat_statuses if c == 304]
        if revalidations_304:
            raise AssertionError(
                f"[warm_repeat_navigation] Unexpected 304 responses on warm navigation: {revalidations_304}"
            )
        with origin_lock:
            warm_origin_hits = list(origin_hits)
        cached_boot_hits = [
            (p, c)
            for p, c in warm_origin_hits
            if p in expected_preboot
        ]
        if cached_boot_hits:
            raise AssertionError(
                f"[warm_repeat_navigation] Expected zero origin requests within max-age, got {cached_boot_hits}"
            )

        # Explicit reload and no-cache modes must forward through sw.js to origin.
        revision_state["rev"] = 2
        with origin_lock:
            origin_hits.clear()

        reload_probe = page2.evaluate(
            """async () => {
                const reloadResp = await fetch('./bootstrap.js', { cache: 'reload' });
                const reloadText = await reloadResp.text();
                const revalidateResp = await fetch('./math_explorer_gui_bg.wasm', { cache: 'no-cache' });
                return {
                    reloadStatus: reloadResp.status,
                    hasUpdatedMarker: reloadText.includes('// origin-rev-2'),
                    revalidateStatus: revalidateResp.status,
                };
            }"""
        )
        with origin_lock:
            bypass_hits = list(origin_hits)
        if not reload_probe["hasUpdatedMarker"] or reload_probe["reloadStatus"] != 200:
            raise AssertionError(
                f"[sw_hard_refresh_passthrough] cache:'reload' did not fetch updated origin asset: {reload_probe}"
            )
        if not any(p == "/bootstrap.js" and c == 200 for p, c in bypass_hits) or not any(
            p == "/math_explorer_gui_bg.wasm" for p, _ in bypass_hits
        ):
            raise AssertionError(
                f"[sw_hard_refresh_passthrough] Expected origin hits for reload/no-cache modes, got {bypass_hits}"
            )

        print(
            "[PASS] parallel_preload_and_sw_http_cache: 6 pre-boot requests in 1 wave, HTTP-cached precacheShell, 0 warm 304s, reload/no-cache forwarded"
        )
    finally:
        context.close()
        cache_server.shutdown()
        cache_server.server_close()


def run_offline_sw_shell_recovery_check(
    browser: Browser,
    base_url: str,
    screenshot_path: Path | None = None,
) -> None:
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        csp_violations: list[str] = []
        attach_csp_observer(page, csp_violations)

        page.goto(f"{base_url}/", wait_until="domcontentloaded")
        expect(page.locator("#loading_indicator")).to_be_hidden(timeout=20_000)
        expect(page.locator("#the_canvas_id")).to_be_focused()

        page.wait_for_function(
            """async () => {
                if (!navigator.serviceWorker) return false;
                await navigator.serviceWorker.ready;
                if (!navigator.serviceWorker.controller) return false;
                const cache = await caches.open('math-explorer-shell-v1');
                const keys = (await cache.keys()).map((r) => new URL(r.url).pathname);
                return (
                    keys.includes('/index.html') &&
                    keys.includes('/reload.js') &&
                    keys.includes('/bootstrap.js') &&
                    keys.includes('/initializer.js')
                );
            }""",
            timeout=10_000,
        )

        offline_statuses: dict[str, list[int]] = {}
        page.on(
            "response",
            lambda resp: offline_statuses.setdefault(
                urllib.parse.urlsplit(resp.url).path, []
            ).append(resp.status),
        )

        context.set_offline(True)
        doc_resp = page.reload(wait_until="domcontentloaded")
        if doc_resp is None or doc_resp.status != 200:
            raise AssertionError(
                f"[offline_sw_shell_recovery] Expected HTTP 200 shell navigation from SHELL_CACHE_NAME, got {doc_resp.status if doc_resp else None}"
            )

        verify_error_recovery(page, "offline_sw_shell_recovery", EXPECTED_FAULTS["network"])
        wasm_codes = offline_statuses.get("/math_explorer_gui_bg.wasm", [])
        if 503 not in wasm_codes:
            raise AssertionError(
                f"[offline_sw_shell_recovery] Expected HTTP 503 for uncached .wasm offline, observed: {offline_statuses}"
            )
        assert_zero_csp_violations("offline_sw_shell_recovery", csp_violations)

        if screenshot_path is not None:
            page.screenshot(path=str(screenshot_path))
        print(
            "[PASS] offline_sw_shell_recovery: cached shell 200, uncached .wasm 503, #error_container active"
        )
    finally:
        context.close()


def run_smoke_suite(
    dist_dir: Path,
    trunk_toml: Path,
    screenshot_dir: Path | None,
    check_regressions: bool,
) -> None:
    configure_mimetypes()
    csp = load_and_verify_production_csp(trunk_toml, dist_dir)
    wasm_size, br_size, gz_size = verify_stamped_dist_artifacts(dist_dir)
    if screenshot_dir is not None:
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    raw_wasm_prefix = (dist_dir / "math_explorer_gui_bg.wasm").read_bytes()[:131072]
    truncated_br_prefix = subprocess.run(
        ["brotli", "-c", "-9"], input=raw_wasm_prefix, capture_output=True, check=True
    ).stdout

    class BoundHandler(CspDistRequestHandler):
        csp_header = csp
        truncated_br_chunk = truncated_br_prefix

    handler = functools.partial(BoundHandler, directory=str(dist_dir))
    server = ReusableThreadingTCPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        with sync_playwright() as playwright:
            webgl_pool = BrowserPool(playwright, WEBGL_SWIFTSHADER_ARGS)
            try:
                run_with_crash_retry(
                    webgl_pool,
                    "happy_path",
                    lambda b: run_happy_path_check(b, base_url, csp, screenshot_dir),
                )

                run_with_crash_retry(
                    webgl_pool,
                    "parallel_preload_and_sw_http_cache",
                    lambda b: run_parallel_preload_and_sw_http_cache_check(b, dist_dir, csp),
                )

                run_with_crash_retry(
                    webgl_pool,
                    "offline_sw_shell_recovery",
                    lambda b: run_offline_sw_shell_recovery_check(
                        b,
                        base_url,
                        screenshot_dir / "smoke_offline_sw_shell_recovery.png"
                        if screenshot_dir
                        else None,
                    ),
                )

                run_with_crash_retry(
                    webgl_pool,
                    "fault_missing_wasm",
                    lambda b: run_fault_scenario(
                        b,
                        base_url,
                        "fault_missing_wasm",
                        "network",
                        "/",
                        lambda page: page.route(
                            "**/math_explorer_gui_bg.wasm*",
                            lambda route: route.fulfill(
                                status=404, content_type="text/plain", body=b"Not Found"
                            ),
                        ),
                        screenshot_dir / "smoke_fault_network_wasm_404.png"
                        if screenshot_dir
                        else None,
                    ),
                )

                run_with_crash_retry(
                    webgl_pool,
                    "fault_broken_loader_script",
                    lambda b: run_fault_scenario(
                        b,
                        base_url,
                        "fault_broken_loader_script",
                        "network",
                        "/",
                        lambda page: page.route(
                            "**/math_explorer_gui.js*",
                            lambda route: route.fulfill(
                                status=404, content_type="text/plain", body=b"Not Found"
                            ),
                        ),
                    ),
                )

                run_with_crash_retry(
                    webgl_pool,
                    "fault_broken_bootstrap_script",
                    lambda b: run_fault_scenario(
                        b,
                        base_url,
                        "fault_broken_bootstrap_script",
                        "network",
                        "/",
                        lambda page: page.route(
                            "**/bootstrap.js*",
                            lambda route: route.fulfill(
                                status=500, content_type="text/plain", body=b"Server Error"
                            ),
                        ),
                    ),
                )

                run_with_crash_retry(
                    webgl_pool,
                    "fault_corrupt_wasm",
                    lambda b: run_fault_scenario(
                        b,
                        base_url,
                        "fault_corrupt_wasm",
                        "compile",
                        "/",
                        lambda page: page.route(
                            "**/math_explorer_gui_bg.wasm*",
                            lambda route: route.fulfill(
                                status=200,
                                content_type="application/wasm",
                                body=b"corrupted-wasm-payload-bytes",
                            ),
                        ),
                        screenshot_dir / "smoke_fault_compile_corrupt_wasm.png"
                        if screenshot_dir
                        else None,
                    ),
                )

                for trunc_scenario, trunc_mode, trunc_shot in (
                    (
                        "fault_wasm_socket_drop_identity",
                        "drop_identity",
                        screenshot_dir / "smoke_fault_network_truncated_stream.png"
                        if screenshot_dir
                        else None,
                    ),
                    ("fault_wasm_socket_drop_brotli", "drop_brotli", None),
                    ("fault_wasm_truncated_chunked", "truncated_chunked", None),
                    ("fault_wasm_truncated_brotli_chunked", "truncated_brotli_chunked", None),
                ):
                    run_with_crash_retry(
                        webgl_pool,
                        trunc_scenario,
                        lambda b, sc=trunc_scenario, md=trunc_mode, sh=trunc_shot: run_truncated_stream_check(
                            b, base_url, dist_dir, sc, md, sh
                        ),
                    )

                for enc_scenario, enc_mode, expected_wire in (
                    ("compressed_brotli_wire_progress", "full_br", br_size),
                    ("compressed_brotli_chunked_progress", "chunked_br", br_size),
                    ("compressed_gzip_wire_progress", "full_gz", gz_size),
                    ("compressed_gzip_chunked_progress", "chunked_gz", gz_size),
                ):
                    run_with_crash_retry(
                        webgl_pool,
                        enc_scenario,
                        lambda b, sc=enc_scenario, md=enc_mode, ew=expected_wire: run_compressed_progress_and_aria_check(
                            b, base_url, dist_dir, sc, md, ew, wasm_size
                        ),
                    )

                run_with_crash_retry(
                    webgl_pool,
                    "internal_error_and_fallback_compile",
                    lambda b: run_internal_error_and_fallback_compile_checks(
                        b,
                        base_url,
                        screenshot_dir / "smoke_fault_compile_internal_error.png"
                        if screenshot_dir
                        else None,
                    ),
                )

                run_with_crash_retry(
                    webgl_pool,
                    "fault_stall_timeout",
                    lambda b: run_fault_scenario(
                        b,
                        base_url,
                        "fault_stall_timeout",
                        "timeout",
                        "/?stall_timeout_ms=250",
                        None,
                        screenshot_dir / "smoke_fault_timeout_stall.png"
                        if screenshot_dir
                        else None,
                    ),
                )

                if check_regressions:
                    run_with_crash_retry(
                        webgl_pool,
                        "regression_network",
                        lambda b: verify_network_regression_detection(b, base_url),
                    )
            finally:
                webgl_pool.close()

            disabled_pool = BrowserPool(playwright, WEBGL_DISABLED_ARGS)
            try:
                if check_regressions:
                    run_with_crash_retry(
                        disabled_pool,
                        "regression_disabled_webgl",
                        lambda b: verify_disabled_webgl_regression_detection(b, base_url),
                    )

                run_with_crash_retry(
                    disabled_pool,
                    "fault_disabled_webgl",
                    lambda b: run_fault_scenario(
                        b,
                        base_url,
                        "fault_disabled_webgl",
                        "webgl",
                        "/",
                        None,
                        screenshot_dir / "smoke_fault_webgl_disabled.png"
                        if screenshot_dir
                        else None,
                    ),
                )
            finally:
                disabled_pool.close()
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    raise_nofile_limit()
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run Playwright Chromium smoke checks on web dist.")
    parser.add_argument(
        "--dist",
        type=Path,
        default=repo_root / "math_explorer_gui" / "dist",
        help="Path to built Trunk web distribution directory",
    )
    parser.add_argument(
        "--trunk-toml",
        type=Path,
        default=repo_root / "math_explorer_gui" / "Trunk.toml",
        help="Path to Trunk.toml defining production CSP headers",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=None,
        help="Optional directory to write verification PNG screenshots",
    )
    parser.add_argument(
        "--verify-regression-detection",
        action="store_true",
        help="Also verify that simulated broken recovery causes smoke assertions to fail",
    )
    args = parser.parse_args()
    run_smoke_suite(
        dist_dir=args.dist.resolve(),
        trunk_toml=args.trunk_toml.resolve(),
        screenshot_dir=args.screenshot_dir.resolve() if args.screenshot_dir else None,
        check_regressions=args.verify_regression_detection,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
