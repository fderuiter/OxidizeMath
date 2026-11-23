#!/usr/bin/env python3
"""Headless Chromium Playwright smoke and fault-injection gate for Math Explorer."""

from __future__ import annotations

import argparse
import functools
import html.parser
import http.server
import mimetypes
import os
try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
import socketserver
import sys
import threading
import time
import tomllib
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


class CspDistRequestHandler(http.server.SimpleHTTPRequestHandler):
    csp_header: str = ""

    def do_GET(self) -> None:
        referer = self.headers.get("Referer", "")
        if "stall_timeout_ms=" in referer and self.path.split("?")[0].endswith(".wasm"):
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


def run_smoke_suite(
    dist_dir: Path,
    trunk_toml: Path,
    screenshot_dir: Path | None,
    check_regressions: bool,
) -> None:
    configure_mimetypes()
    csp = load_and_verify_production_csp(trunk_toml, dist_dir)
    if screenshot_dir is not None:
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    class BoundHandler(CspDistRequestHandler):
        csp_header = csp

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
