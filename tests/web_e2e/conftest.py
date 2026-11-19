"""Pytest fixtures and runtime diagnostics for Math Explorer browser E2E tests."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from playwright.sync_api import Browser, ConsoleMessage, Error as PlaywrightError, Page, Response, sync_playwright

# Ensure local sibling imports resolve regardless of pytest rootdir invocation mode.
SUITE_DIR = Path(__file__).resolve().parent
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from server import LocalDistServer, start_dist_server  # noqa: E402

CHROMIUM_LAUNCH_ARGS = [
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--no-sandbox",
]

CSP_MONITOR_INIT_SCRIPT = """
(() => {
    window.__cspViolations = [];
    window.addEventListener('securitypolicyviolation', (event) => {
        window.__cspViolations.push({
            blockedURI: event.blockedURI || '',
            violatedDirective: event.violatedDirective || '',
            effectiveDirective: event.effectiveDirective || '',
            originalPolicy: event.originalPolicy || '',
            disposition: event.disposition || '',
        });
    });
})();
"""


@dataclass
class PageDiagnostics:
    """Captures console messages, uncaught page errors, CSP violations, and HTTP responses."""

    console_messages: List[str] = field(default_factory=list)
    console_errors: List[str] = field(default_factory=list)
    csp_console_errors: List[str] = field(default_factory=list)
    page_errors: List[str] = field(default_factory=list)
    responses: List[Dict[str, object]] = field(default_factory=list)

    def attach(self, page: Page) -> "PageDiagnostics":
        """Register event listeners and CSP violation hooks prior to navigation."""
        page.add_init_script(CSP_MONITOR_INIT_SCRIPT)

        def on_console(msg: ConsoleMessage) -> None:
            entry = f"[{msg.type}] {msg.text}"
            self.console_messages.append(entry)
            if "Content Security Policy" in msg.text or "securitypolicyviolation" in msg.text:
                self.csp_console_errors.append(msg.text)
            if msg.type == "error":
                self.console_errors.append(msg.text)

        def on_page_error(err: Exception) -> None:
            self.page_errors.append(str(err))

        def on_response(resp: Response) -> None:
            self.responses.append(
                {
                    "url": resp.url,
                    "status": resp.status,
                    "headers": resp.headers,
                }
            )

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("response", on_response)
        return self

    def read_csp_violations(self, page: Page) -> List[Dict[str, str]]:
        """Read DOM SecurityPolicyViolationEvent records using a CSP-safe function expression."""
        return page.evaluate("() => window.__cspViolations || []")

    def format_summary(self, page: Page) -> str:
        """Render console and DOM diagnostics for assertion failure messages."""
        dom_snapshot = page.evaluate(
            """() => ({
                activeId: document.activeElement ? document.activeElement.id : null,
                hasLoading: Boolean(document.getElementById('loading_indicator')),
                errorDisplay: document.getElementById('error_container')
                    ? window.getComputedStyle(document.getElementById('error_container')).display
                    : null,
                errorDetails: document.getElementById('error_details')
                    ? document.getElementById('error_details').textContent
                    : null,
                crossOriginIsolated: window.crossOriginIsolated,
            })"""
        )
        return (
            f"DOM={dom_snapshot}\n"
            f"page_errors={self.page_errors}\n"
            f"csp_console_errors={self.csp_console_errors}\n"
            f"console_errors={self.console_errors}\n"
            f"console_tail={self.console_messages[-15:]}"
        )

    def assert_clean_runtime(self, page: Page) -> None:
        """Assert zero uncaught page errors, zero console errors, and zero CSP violations."""
        csp_events = self.read_csp_violations(page)
        summary = self.format_summary(page)
        assert not csp_events, f"CSP SecurityPolicyViolationEvent(s) detected: {csp_events}\n{summary}"
        assert not self.csp_console_errors, f"CSP console error(s) detected:\n{summary}"
        assert not self.page_errors, f"Uncaught pageerror(s) detected:\n{summary}"
        assert not self.console_errors, f"Browser console.error message(s) detected:\n{summary}"


def poll_until(
    page: Page,
    predicate_fn: str,
    *,
    timeout_ms: int = 15000,
    interval_ms: int = 50,
) -> Any:
    """Poll a JavaScript arrow-function predicate via CDP page.evaluate without unsafe-eval."""
    assert predicate_fn.lstrip().startswith("("), (
        f"predicate_fn must be a JS function expression for strict CSP compatibility: {predicate_fn}"
    )
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = page.evaluate(predicate_fn)
            if result:
                return result
        except PlaywrightError as exc:
            # Transient navigation (e.g. coi-serviceworker reload) resets execution context.
            last_error = exc
        time.sleep(interval_ms / 1000.0)

    detail = f" (last transient error: {last_error})" if last_error else ""
    raise TimeoutError(
        f"Timed out after {timeout_ms}ms waiting for predicate: {predicate_fn}{detail}"
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def dist_dir(repo_root: Path) -> Path:
    dist = repo_root / "math_explorer_gui" / "dist"
    index_html = dist / "index.html"
    assert index_html.is_file(), (
        f"Expected built web artifact at {index_html}. "
        "Run 'cargo run -p xtask -- web-e2e' to build math_explorer_gui/dist before testing."
    )
    return dist


@pytest.fixture(scope="session")
def isolated_server(dist_dir: Path) -> Iterator[LocalDistServer]:
    server = start_dist_server(dist_dir, isolated=True)
    try:
        yield server
    finally:
        server.close()


@pytest.fixture(scope="session")
def unisolated_server(dist_dir: Path) -> Iterator[LocalDistServer]:
    server = start_dist_server(dist_dir, isolated=False)
    try:
        yield server
    finally:
        server.close()


@pytest.fixture(scope="session")
def chromium_browser() -> Iterator[Browser]:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=CHROMIUM_LAUNCH_ARGS)
        try:
            yield browser
        finally:
            browser.close()
