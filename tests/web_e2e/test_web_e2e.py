"""Headless Chromium E2E tests for Math Explorer WebAssembly startup and fault recovery."""

from __future__ import annotations

import time
import urllib.request
from typing import Dict, List

import pytest
from playwright.sync_api import Browser, Page, Route

from conftest import PageDiagnostics, poll_until
from server import LocalDistServer


def verify_favicons_delivered(page: Page, diagnostics: PageDiagnostics) -> List[Dict[str, object]]:
    """Verify declared <link rel="icon"> assets exist and return non-empty HTTP 200 images."""
    icon_hrefs: List[str] = page.evaluate(
        """() => Array.from(document.querySelectorAll('link[rel~="icon"]'))
            .map((el) => el.getAttribute('href') || '')
            .filter((href) => href.length > 0)"""
    )
    assert len(icon_hrefs) >= 2, f"Expected at least 2 favicon <link> tags, found: {icon_hrefs}"

    # Verify any passive favicon responses recorded by Chromium did not fail.
    for resp in diagnostics.responses:
        url = str(resp["url"])
        if any(url.endswith(href.lstrip("./")) for href in icon_hrefs) or url.endswith("/favicon.ico"):
            assert resp["status"] == 200, f"Passive favicon request failed: {resp}"

    results: List[Dict[str, object]] = []
    for href in icon_hrefs:
        fetched = page.evaluate(
            """async (targetHref) => {
                const response = await fetch(targetHref, { cache: 'no-store' });
                const buffer = await response.arrayBuffer();
                return {
                    href: targetHref,
                    status: response.status,
                    contentType: response.headers.get('content-type') || '',
                    byteLength: buffer.byteLength,
                };
            }""",
            href,
        )
        assert fetched["status"] == 200, f"Favicon '{href}' returned HTTP {fetched['status']}: {fetched}"
        assert int(fetched["byteLength"]) > 0, f"Favicon '{href}' body was empty: {fetched}"
        content_type = str(fetched["contentType"]).lower()
        assert "image/" in content_type, f"Favicon '{href}' had unexpected Content-Type: {fetched}"
        results.append(fetched)

    return results


def assert_accessible_error_alert_and_reload_focus(page: Page) -> str:
    """Wait for #error_container alert state, #error_details visibility, and #webgl_reload_button focus."""
    poll_until(
        page,
        """() => {
            const container = document.getElementById('error_container');
            const loading = document.getElementById('loading_indicator');
            return Boolean(
                container &&
                window.getComputedStyle(container).display === 'flex' &&
                !loading &&
                document.activeElement &&
                document.activeElement.id === 'webgl_reload_button'
            );
        }""",
        timeout_ms=15000,
    )

    alert_state = page.evaluate(
        """() => {
            const container = document.getElementById('error_container');
            const details = document.getElementById('error_details');
            const canvas = document.getElementById('the_canvas_id');
            return {
                role: container ? container.getAttribute('role') : null,
                ariaLive: container ? container.getAttribute('aria-live') : null,
                containerDisplay: container ? window.getComputedStyle(container).display : null,
                detailsDisplay: details ? window.getComputedStyle(details).display : null,
                detailsText: details ? (details.textContent || '').trim() : '',
                canvasDisplay: canvas ? window.getComputedStyle(canvas).display : null,
                activeId: document.activeElement ? document.activeElement.id : null,
            };
        }"""
    )

    assert alert_state["role"] == "alert", f"Expected role='alert' on #error_container: {alert_state}"
    assert (
        alert_state["ariaLive"] == "assertive"
    ), f"Expected aria-live='assertive' on #error_container: {alert_state}"
    assert alert_state["containerDisplay"] == "flex", f"#error_container not flex: {alert_state}"
    assert alert_state["detailsDisplay"] == "block", f"#error_details not visible: {alert_state}"
    assert len(alert_state["detailsText"]) > 0, f"#error_details was empty: {alert_state}"
    assert alert_state["canvasDisplay"] == "none", f"#the_canvas_id should be hidden: {alert_state}"
    assert (
        alert_state["activeId"] == "webgl_reload_button"
    ), f"Expected keyboard focus on #webgl_reload_button, got: {alert_state}"

    return str(alert_state["detailsText"])


def assert_canvas_ready_and_focused(page: Page, timeout_ms: int = 25000) -> None:
    """Wait for #loading_indicator removal and keyboard focus on #the_canvas_id."""
    poll_until(
        page,
        """() => (
            !document.getElementById('loading_indicator') &&
            Boolean(document.activeElement && document.activeElement.id === 'the_canvas_id')
        )""",
        timeout_ms=timeout_ms,
    )
    error_display = page.evaluate(
        """() => {
            const container = document.getElementById('error_container');
            return container ? window.getComputedStyle(container).display : 'none';
        }"""
    )
    assert error_display == "none", (
        f"#error_container should remain hidden on healthy startup, got display={error_display}"
    )


def test_cold_load_startup_csp_isolation_favicons_and_focus(
    chromium_browser: Browser,
    isolated_server: LocalDistServer,
) -> None:
    """Verify cold startup, strict CSP, crossOriginIsolated, favicons, and canvas focus."""
    context = chromium_browser.new_context()
    try:
        page = context.new_page()
        diagnostics = PageDiagnostics().attach(page)

        response = page.goto(isolated_server.base_url, wait_until="domcontentloaded")
        assert response is not None and response.status == 200
        assert not diagnostics.csp_console_errors, (
            f"CSP violation during initial document parse: {diagnostics.csp_console_errors}"
        )

        assert_canvas_ready_and_focused(page)

        csp_meta = page.evaluate(
            """() => {
                const el = document.querySelector('meta[http-equiv="Content-Security-Policy"]');
                return el ? el.getAttribute('content') || '' : '';
            }"""
        )
        assert "script-src 'self' 'wasm-unsafe-eval' blob:" in csp_meta
        assert "'unsafe-eval'" not in csp_meta
        assert "'unsafe-inline'" not in csp_meta.split("style-src")[0]

        is_isolated = page.evaluate("() => window.crossOriginIsolated === true")
        assert is_isolated, "Expected window.crossOriginIsolated === true under COOP/COEP headers"

        favicons = verify_favicons_delivered(page, diagnostics)
        hrefs = {str(item["href"]) for item in favicons}
        assert any(h.endswith("icon.ico") for h in hrefs), f"Missing icon.ico in {hrefs}"
        assert any(h.endswith("32x32.png") for h in hrefs), f"Missing 32x32.png in {hrefs}"

        diagnostics.assert_clean_runtime(page)
    finally:
        context.close()


def test_favicon_404_fails_check_immediately(
    chromium_browser: Browser,
    isolated_server: LocalDistServer,
) -> None:
    """Verify returning HTTP 404 on favicon requests fails favicon verification immediately."""
    context = chromium_browser.new_context()
    try:
        page = context.new_page()
        diagnostics = PageDiagnostics().attach(page)

        def fail_favicon(route: Route) -> None:
            route.fulfill(status=404, content_type="text/plain", body="Not Found")

        page.route("**/icon.ico", fail_favicon)

        page.goto(isolated_server.base_url, wait_until="domcontentloaded")
        assert_canvas_ready_and_focused(page)

        with pytest.raises(AssertionError, match="404"):
            verify_favicons_delivered(page, diagnostics)
    finally:
        context.close()


def test_csp_blocks_inline_script_and_fails_diagnostics(
    chromium_browser: Browser,
    isolated_server: LocalDistServer,
) -> None:
    """Verify strict CSP blocks inline script injection and triggers immediate diagnostic failure."""
    context = chromium_browser.new_context()
    try:
        page = context.new_page()
        diagnostics = PageDiagnostics().attach(page)

        page.goto(isolated_server.base_url, wait_until="domcontentloaded")
        assert_canvas_ready_and_focused(page)

        page.evaluate(
            """() => {
                const s = document.createElement('script');
                s.textContent = 'window.__unauthorizedInlineScriptRan = true;';
                document.head.appendChild(s);
            }"""
        )
        poll_until(page, "() => (window.__cspViolations || []).length > 0", timeout_ms=5000)

        inline_ran = page.evaluate("() => Boolean(window.__unauthorizedInlineScriptRan)")
        assert not inline_ran, "Inline script must be blocked by strict script-src CSP"

        with pytest.raises(AssertionError, match="CSP"):
            diagnostics.assert_clean_runtime(page)
    finally:
        context.close()


def test_coi_serviceworker_activates_isolation_on_static_host(
    chromium_browser: Browser,
    unisolated_server: LocalDistServer,
) -> None:
    """Verify coi-serviceworker.min.js activates crossOriginIsolated without server COOP/COEP headers."""
    with urllib.request.urlopen(unisolated_server.base_url, timeout=5.0) as raw_resp:
        raw_headers = {k.lower(): v for k, v in raw_resp.headers.items()}
        assert "cross-origin-opener-policy" not in raw_headers
        assert "cross-origin-embedder-policy" not in raw_headers

    context = chromium_browser.new_context()
    try:
        page = context.new_page()
        diagnostics = PageDiagnostics().attach(page)

        page.goto(unisolated_server.base_url, wait_until="domcontentloaded")

        # coi-serviceworker registers and triggers an automatic reload into isolated context.
        poll_until(
            page,
            """() => (
                window.crossOriginIsolated === true &&
                Boolean(navigator.serviceWorker && navigator.serviceWorker.controller)
            )""",
            timeout_ms=25000,
        )

        assert_canvas_ready_and_focused(page)
        diagnostics.assert_clean_runtime(page)
    finally:
        context.close()


def test_slow_3g_progressive_loading_before_canvas_focus(
    chromium_browser: Browser,
    isolated_server: LocalDistServer,
) -> None:
    """Verify Slow-3G throttling via Playwright routes + CDP shows progressive #loading_indicator updates."""
    context = chromium_browser.new_context()
    try:
        page = context.new_page()
        diagnostics = PageDiagnostics().attach(page)

        cdp = context.new_cdp_session(page)
        cdp.send("Network.enable")
        cdp.send(
            "Network.emulateNetworkConditions",
            {
                "offline": False,
                "latency": 120,
                "downloadThroughput": 1_400_000,
                "uploadThroughput": 400_000,
                "connectionType": "cellular3g",
            },
        )

        throttled_assets: List[str] = []

        def throttle_bundle(route: Route) -> None:
            url = route.request.url
            if url.endswith("math_explorer_gui.js") or url.endswith("math_explorer_gui_bg.wasm"):
                throttled_assets.append(url)
                time.sleep(0.25)
            route.continue_()

        page.route("**/math_explorer_gui*", throttle_bundle)

        page.goto(isolated_server.base_url, wait_until="domcontentloaded")

        # Immediately after DOMContentLoaded under Slow-3G, #loading_indicator must be visible
        # and #the_canvas_id must not yet hold keyboard focus.
        initial_state = page.evaluate(
            """() => {
                const indicator = document.getElementById('loading_indicator');
                return {
                    present: Boolean(indicator),
                    role: indicator ? indicator.getAttribute('role') : null,
                    ariaBusy: indicator ? indicator.getAttribute('aria-busy') : null,
                    activeId: document.activeElement ? document.activeElement.id : null,
                };
            }"""
        )
        assert initial_state["present"] is True, f"Expected #loading_indicator during Slow-3G load: {initial_state}"
        assert initial_state["role"] == "status"
        assert initial_state["ariaBusy"] == "true"
        assert initial_state["activeId"] != "the_canvas_id"

        # Observe live progressive WASM streaming updates while #loading_indicator is still mounted.
        poll_until(
            page,
            """() => {
                const stages = window.__mathExplorerLoadingStages || [];
                return stages.some((entry) => entry.stage === 'downloading-wasm' && (entry.loadedBytes || 0) > 0);
            }""",
            timeout_ms=25000,
        )

        assert_canvas_ready_and_focused(page, timeout_ms=35000)

        recorded_stages = page.evaluate("() => window.__mathExplorerLoadingStages || []")
        stage_names = [str(item["stage"]) for item in recorded_stages]
        assert "loading-js" in stage_names, f"Missing loading-js stage in {stage_names}"
        assert "fetching-wasm" in stage_names, f"Missing fetching-wasm stage in {stage_names}"
        assert "downloading-wasm" in stage_names, f"Missing downloading-wasm stage in {stage_names}"
        assert "compiling-wasm" in stage_names, f"Missing compiling-wasm stage in {stage_names}"

        download_bytes = [
            int(item["loadedBytes"])
            for item in recorded_stages
            if item["stage"] == "downloading-wasm" and item["loadedBytes"] is not None
        ]
        assert len(download_bytes) >= 2, f"Expected multiple progressive WASM chunks, got: {download_bytes}"
        assert download_bytes == sorted(download_bytes), f"Byte progress must be monotonic: {download_bytes}"
        assert download_bytes[-1] > download_bytes[0], f"Byte progress did not advance: {download_bytes}"
        assert len(throttled_assets) >= 2, f"Expected both .js and .wasm routes throttled, got: {throttled_assets}"

        diagnostics.assert_clean_runtime(page)
    finally:
        context.close()


def test_fault_offline_alerts_and_recovers_via_reload_button(
    chromium_browser: Browser,
    isolated_server: LocalDistServer,
) -> None:
    """Verify offline state during WASM fetch triggers accessible alert and #webgl_reload_button recovery."""
    context = chromium_browser.new_context()
    try:
        page = context.new_page()
        page_errors: List[str] = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        offline_injected = {"active": True}

        def drop_connection_on_wasm(route: Route) -> None:
            if offline_injected["active"]:
                context.set_offline(True)
                route.abort("internetdisconnected")
            else:
                route.continue_()

        page.route("**/math_explorer_gui_bg.wasm", drop_connection_on_wasm)

        page.goto(isolated_server.base_url, wait_until="domcontentloaded")
        details = assert_accessible_error_alert_and_reload_focus(page)
        assert len(details) > 0
        assert not page_errors, f"Unexpected uncaught pageerror(s): {page_errors}"

        # Restore connectivity and recover via #webgl_reload_button (handled by reload.js).
        offline_injected["active"] = False
        context.set_offline(False)
        page.locator("#webgl_reload_button").click()

        assert_canvas_ready_and_focused(page)
        assert not page_errors, f"Unexpected uncaught pageerror(s) after recovery: {page_errors}"
    finally:
        context.close()


def test_fault_http_500_alerts_and_recovers_via_reload_button(
    chromium_browser: Browser,
    isolated_server: LocalDistServer,
) -> None:
    """Verify HTTP 500 on WASM fetch surfaces accessible #error_container alert and recovers on reload."""
    context = chromium_browser.new_context()
    try:
        page = context.new_page()
        page_errors: List[str] = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        fail_wasm = {"active": True}

        def inject_http_500(route: Route) -> None:
            if fail_wasm["active"]:
                route.fulfill(
                    status=500,
                    content_type="text/plain; charset=utf-8",
                    headers={"Cache-Control": "no-store"},
                    body="Internal Server Error",
                )
            else:
                route.continue_()

        page.route("**/math_explorer_gui_bg.wasm", inject_http_500)

        page.goto(isolated_server.base_url, wait_until="domcontentloaded")
        details = assert_accessible_error_alert_and_reload_focus(page)
        assert "500" in details, f"Expected HTTP 500 diagnostic in #error_details, got: {details}"
        assert not page_errors, f"Unexpected uncaught pageerror(s): {page_errors}"

        fail_wasm["active"] = False
        page.locator("#webgl_reload_button").click()

        assert_canvas_ready_and_focused(page)
        assert not page_errors, f"Unexpected uncaught pageerror(s) after recovery: {page_errors}"
    finally:
        context.close()


def test_fault_corrupt_wasm_alerts_and_recovers_via_reload_button(
    chromium_browser: Browser,
    isolated_server: LocalDistServer,
) -> None:
    """Verify corrupt .wasm bytes trigger accessible #error_container alert and recover on reload."""
    context = chromium_browser.new_context()
    try:
        page = context.new_page()
        page_errors: List[str] = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        corrupt_wasm = {"active": True}

        def inject_corrupt_wasm(route: Route) -> None:
            if corrupt_wasm["active"]:
                route.fulfill(
                    status=200,
                    content_type="application/wasm",
                    headers={"Cache-Control": "no-store"},
                    body=b"\x00\x61\x00\x00corrupt_wasm_binary_payload",
                )
            else:
                route.continue_()

        page.route("**/math_explorer_gui_bg.wasm", inject_corrupt_wasm)

        page.goto(isolated_server.base_url, wait_until="domcontentloaded")
        details = assert_accessible_error_alert_and_reload_focus(page)
        assert (
            "wasm" in details.lower() or "compile" in details.lower() or "magic" in details.lower()
        ), f"Expected WebAssembly compilation error details, got: {details}"
        assert not page_errors, f"Unexpected uncaught pageerror(s): {page_errors}"

        corrupt_wasm["active"] = False
        page.locator("#webgl_reload_button").click()

        assert_canvas_ready_and_focused(page)
        assert not page_errors, f"Unexpected uncaught pageerror(s) after recovery: {page_errors}"
    finally:
        context.close()


def test_webgl_disabled_fallback_shows_details_and_focuses_reload_button(
    chromium_browser: Browser,
    isolated_server: LocalDistServer,
) -> None:
    """Verify blocking webgl and webgl2 contexts triggers Rust handle_webgl_failure fallback UI."""
    context = chromium_browser.new_context()
    try:
        page = context.new_page()
        diagnostics = PageDiagnostics().attach(page)

        page.add_init_script(
            """
            (() => {
                const origGetContext = HTMLCanvasElement.prototype.getContext;
                HTMLCanvasElement.prototype.getContext = function (contextType, ...rest) {
                    if (
                        contextType === 'webgl' ||
                        contextType === 'webgl2' ||
                        contextType === 'experimental-webgl'
                    ) {
                        return null;
                    }
                    return origGetContext.call(this, contextType, ...rest);
                };
            })();
            """
        )

        page.goto(isolated_server.base_url, wait_until="domcontentloaded")
        details = assert_accessible_error_alert_and_reload_focus(page)

        title_text = page.locator("#error_container .error-title").inner_text()
        assert "WebGL Initialization Failed" in title_text
        assert len(details) > 0

        assert not diagnostics.read_csp_violations(page)
        assert not diagnostics.page_errors
    finally:
        context.close()
