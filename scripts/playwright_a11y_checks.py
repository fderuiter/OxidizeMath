"""Playwright end-to-end accessibility, speech opt-in, and WebGL recovery checks."""

from __future__ import annotations

from pathlib import Path
from playwright.sync_api import Browser, Page, expect

READY_ANNOUNCEMENT = "Math Explorer ready. Interactive simulation workspace loaded."

SPEECH_SPY_INIT_JS = """
(() => {
    const state = {
        calls: [],
        cancels: 0,
        speakingNow: false,
    };
    window.__mathExplorerSpeechSpy = state;

    if (typeof window.SpeechSynthesisUtterance === 'undefined') {
        window.SpeechSynthesisUtterance = class SpeechSynthesisUtterance {
            constructor(text) {
                this.text = String(text ?? '');
                this.volume = 1.0;
                this.rate = 1.0;
                this.pitch = 1.0;
            }
        };
    }

    const proto = window.SpeechSynthesisUtterance.prototype;
    const synthTarget = window.speechSynthesis || {};
    const speakFn = function (utterance) {
        const text = utterance && typeof utterance.text === 'string' ? utterance.text : String(utterance ?? '');
        const volume = utterance && typeof utterance.volume === 'number' ? utterance.volume : 1.0;
        state.calls.push({ text, volume });
        state.speakingNow = true;
    };
    const cancelFn = function () {
        state.cancels += 1;
        state.speakingNow = false;
    };

    try {
        Object.defineProperty(synthTarget, 'speaking', {
            configurable: true,
            get: () => state.speakingNow,
        });
        synthTarget.speak = speakFn;
        synthTarget.cancel = cancelFn;
        if (!window.speechSynthesis) {
            Object.defineProperty(window, 'speechSynthesis', {
                configurable: true,
                value: synthTarget,
            });
        }
    } catch (_err) {
        Object.defineProperty(window, 'speechSynthesis', {
            configurable: true,
            value: {
                get speaking() {
                    return state.speakingNow;
                },
                speak: speakFn,
                cancel: cancelFn,
            },
        });
    }
    void proto;
})();
"""


def assert_ready_announced_once_in_dom(page: Page, scenario: str) -> None:
    slots = page.evaluate(
        """() => {
            const s0 = (document.getElementById('aria-live-polite-slot-0')?.textContent || '').trim();
            const s1 = (document.getElementById('aria-live-polite-slot-1')?.textContent || '').trim();
            return [s0, s1];
        }"""
    )
    non_empty = [s for s in slots if s]
    if READY_ANNOUNCEMENT not in non_empty or len(non_empty) != 1:
        raise AssertionError(
            f"[{scenario}] Expected twin-slot #aria-live-polite to hold single ready announcement "
            f"{READY_ANNOUNCEMENT!r}, got slots={slots!r}"
        )


def assert_accesskit_dom_semantics(page: Page, scenario: str) -> dict[str, int]:
    page.wait_for_function(
        """() => {
            const root = document.getElementById('accesskit_dom_root');
            if (!root) return false;
            return (
                root.querySelectorAll('[role="region"], [role="navigation"]').length >= 2 &&
                root.querySelectorAll('[role="heading"]').length >= 1 &&
                root.querySelectorAll('button, input, textarea, [role="slider"], [role="checkbox"]').length >= 3
            );
        }""",
        timeout=10_000,
    )
    counts = page.evaluate(
        """() => {
            const root = document.getElementById('accesskit_dom_root');
            if (!root) return null;
            return {
                total: root.querySelectorAll('[data-ak-id]').length,
                landmarks: root.querySelectorAll('[role="region"], [role="navigation"], [role="main"]').length,
                headings: root.querySelectorAll('[role="heading"]').length,
                controls: root.querySelectorAll(
                    'button, input, textarea, [role="slider"], [role="checkbox"], [role="radio"], [role="separator"], [role="spinbutton"]'
                ).length,
                visualizations: root.querySelectorAll('[role="graphics-document"], [role="img"], figure').length,
            };
        }"""
    )
    if not counts or counts["landmarks"] < 2 or counts["headings"] < 1 or counts["controls"] < 3:
        raise AssertionError(f"[{scenario}] Incomplete #accesskit_dom_root projection: {counts}")
    return counts


def run_accesskit_dom_speech_and_webgl_recovery_check(
    browser: Browser,
    base_url: str,
    attach_csp_observer: object,
    assert_zero_csp_violations: object,
    screenshot_dir: Path | None,
) -> None:
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        context.add_init_script(SPEECH_SPY_INIT_JS)
        page = context.new_page()
        csp_violations: list[str] = []
        console_errors: list[str] = []
        page_errors: list[str] = []
        if callable(attach_csp_observer):
            attach_csp_observer(page, csp_violations)
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        page.goto(f"{base_url}/", wait_until="domcontentloaded")
        expect(page.locator("#loading_indicator")).to_be_hidden(timeout=20_000)

        canvas = page.locator("#the_canvas_id")
        if canvas.get_attribute("role") == "application":
            raise AssertionError("[a11y_bridge] #the_canvas_id must not have role='application'")

        assert_ready_announced_once_in_dom(page, "a11y_bridge_boot")
        counts = assert_accesskit_dom_semantics(page, "a11y_bridge_boot")

        speech_boot = page.evaluate(
            "() => ({ speaking: Boolean(window.speechSynthesis?.speaking), calls: window.__mathExplorerSpeechSpy.calls.length })"
        )
        if speech_boot["speaking"] or speech_boot["calls"] != 0:
            raise AssertionError(
                f"[a11y_bridge] SpeechSynthesis must remain silent on boot: {speech_boot}"
            )

        # Dismiss cold-boot Starter & Challenge Gallery onboarding modal (#1728) via #accesskit_dom_root if open.
        close_gallery_btn = page.locator("#accesskit_dom_root button", has_text="Close Gallery")
        if close_gallery_btn.count() > 0:
            close_gallery_btn.first.evaluate("(el) => { el.focus(); el.click(); }")
            expect(close_gallery_btn).to_have_count(0, timeout=5_000)

        # 1. DOM focus + click routing on #accesskit_dom_root: focus & open Appearance menu.
        appearance_btn = page.locator("#accesskit_dom_root button", has_text="Appearance").first
        appearance_btn.evaluate("(el) => { el.focus(); }")
        page.wait_for_timeout(80)
        focused_in_ak = page.evaluate(
            "() => Boolean(document.activeElement && document.activeElement.closest('#accesskit_dom_root'))"
        )
        if not focused_in_ak:
            raise AssertionError("[a11y_bridge] DOM focus inside #accesskit_dom_root was stolen by canvas")
        speech_after_focus = page.evaluate("() => window.__mathExplorerSpeechSpy.calls.length")
        if speech_after_focus != 0:
            raise AssertionError(
                f"[a11y_bridge] Widget focus spoke {speech_after_focus} utterances while Speech Narration was disabled"
            )

        appearance_btn.evaluate("(el) => { el.click(); }")
        speech_checkbox = page.locator(
            '#accesskit_dom_root [role="checkbox"][aria-label*="Speech Narration"]'
        ).first
        expect(speech_checkbox).to_be_attached(timeout=5_000)
        expect(speech_checkbox).to_have_attribute("aria-checked", "false")

        if screenshot_dir is not None:
            page.screenshot(path=str(screenshot_dir / "smoke_a11y_appearance_speech_controls.png"))

        # 2. Opt-in Speech Narration toggle, independent volume input, and immediate Stop Speech cancel.
        volume_slider = page.locator(
            '#accesskit_dom_root input[type="range"][aria-label*="Speech Volume"]'
        ).first
        expect(volume_slider).to_be_attached(timeout=5_000)
        volume_slider.evaluate(
            """(el) => {
                el.value = '55';
                el.dispatchEvent(new Event('input', { bubbles: true }));
            }"""
        )
        page.wait_for_timeout(80)
        expect(volume_slider).to_have_attribute("aria-valuenow", "55")

        speech_checkbox.evaluate("(el) => { el.click(); }")
        expect(speech_checkbox).to_have_attribute("aria-checked", "true", timeout=5_000)

        # Trigger focus change while Speech Narration is opted-in so egui emits spoken description at 55% volume.
        stop_btn = page.locator("#accesskit_dom_root button", has_text="Stop Speech").first
        stop_btn.evaluate("(el) => { el.focus(); }")
        page.wait_for_function(
            "() => window.__mathExplorerSpeechSpy.calls.length >= 1",
            timeout=5_000,
        )
        latest_speech = page.evaluate(
            "() => window.__mathExplorerSpeechSpy.calls[window.__mathExplorerSpeechSpy.calls.length - 1]"
        )
        if abs(float(latest_speech["volume"]) - 0.55) > 0.05:
            raise AssertionError(
                f"[a11y_bridge] Expected SpeechSynthesisUtterance volume ~0.55, got {latest_speech}"
            )

        cancels_before = page.evaluate("() => window.__mathExplorerSpeechSpy.cancels")
        stop_btn.evaluate("(el) => { el.click(); }")
        page.wait_for_function(
            f"() => window.__mathExplorerSpeechSpy.cancels > {cancels_before} && !window.speechSynthesis.speaking",
            timeout=5_000,
        )

        # Disable Speech Narration again and close Appearance menu.
        speech_checkbox.evaluate("(el) => { el.click(); }")
        expect(speech_checkbox).to_have_attribute("aria-checked", "false", timeout=5_000)
        appearance_btn.evaluate("(el) => { el.click(); }")
        page.wait_for_timeout(80)

        # 3. Activate Split Horizontal via View -> Viewport Layout -> Split Horizontal and drive sash separator via DOM keydown!
        view_btn = page.locator("#accesskit_dom_root button", has_text="View").first
        view_btn.evaluate("(el) => { el.click(); }")
        dup_btn = page.locator("#accesskit_dom_root button", has_text="Viewport Layout").first
        expect(dup_btn).to_be_attached(timeout=5_000)
        dup_btn.evaluate("(el) => { el.focus(); el.click(); }")
        page.wait_for_timeout(80)
        split_radio = page.locator(
            '#accesskit_dom_root [role="radio"][aria-label*="Split Horizontal"], #accesskit_dom_root button:has-text("Duplicate Focused Pane")'
        ).first
        if split_radio.count() > 0:
            split_radio.evaluate("(el) => { el.click(); }")
        else:
            canvas.focus()
            page.keyboard.press("Control+Shift+Digit2")

        separator = page.locator('#accesskit_dom_root [role="separator"]').first
        expect(separator).to_be_attached(timeout=5_000)
        before_sash = float(separator.get_attribute("aria-valuenow") or "50")
        separator.evaluate(
            """(el) => {
                el.focus();
                el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
            }"""
        )
        page.wait_for_function(
            f"""() => {{
                const sep = document.querySelector('#accesskit_dom_root [role="separator"]');
                return sep && Math.abs(Number(sep.getAttribute('aria-valuenow') || '0') - {before_sash}) >= 0.5;
            }}""",
            timeout=5_000,
        )
        after_sash = float(separator.get_attribute("aria-valuenow") or "50")
        if screenshot_dir is not None:
            page.screenshot(path=str(screenshot_dir / "smoke_a11y_split_pane_sash_keyboard.png"))

        # 4. WebGL context loss & restoration lifecycle: #accesskit_dom_root clears on destroy and rebuilds on restart.
        calls_before_loss = page.evaluate("() => window.__mathExplorerSpeechSpy.calls.length")
        has_lose_ext = page.evaluate(
            """() => {
                const canvas = document.getElementById('the_canvas_id');
                const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                const ext = gl && gl.getExtension('WEBGL_lose_context');
                if (!ext) return false;
                window.__smokeLoseExt = ext;
                ext.loseContext();
                return true;
            }"""
        )
        if not has_lose_ext:
            raise AssertionError("[a11y_bridge] WEBGL_lose_context unavailable on #the_canvas_id")

        expect(page.locator("#webgl_recovery_overlay")).to_be_visible(timeout=5_000)
        page.wait_for_function(
            "() => (document.getElementById('accesskit_dom_root')?.childElementCount ?? -1) === 0",
            timeout=5_000,
        )

        page.evaluate("() => window.__smokeLoseExt.restoreContext()")
        expect(page.locator("#webgl_recovery_overlay")).to_be_hidden(timeout=15_000)
        restored_counts = assert_accesskit_dom_semantics(page, "a11y_bridge_webgl_restored")

        speech_after_restore = page.evaluate(
            "() => ({ speaking: Boolean(window.speechSynthesis?.speaking), calls: window.__mathExplorerSpeechSpy.calls.length })"
        )
        if speech_after_restore["speaking"] or speech_after_restore["calls"] != calls_before_loss:
            raise AssertionError(
                f"[a11y_bridge] SpeechSynthesis must remain silent across WebGL recovery: {speech_after_restore}"
            )

        if screenshot_dir is not None:
            page.screenshot(path=str(screenshot_dir / "smoke_a11y_webgl_context_restored_dom.png"))

        if callable(assert_zero_csp_violations):
            assert_zero_csp_violations("a11y_dom_speech_and_webgl_recovery", csp_violations)
        if console_errors:
            raise AssertionError(f"[a11y_bridge] Console errors detected: {console_errors}")
        if page_errors:
            raise AssertionError(f"[a11y_bridge] Uncaught page errors detected: {page_errors}")

        print(
            f"[PASS] a11y_dom_speech_and_webgl_recovery: boot nodes={counts['total']} "
            f"(landmarks={counts['landmarks']}, headings={counts['headings']}, controls={counts['controls']}, "
            f"vis={counts['visualizations']}), sash {before_sash:.0f}%->{after_sash:.0f}%, "
            f"opt-in speech+stop verified, WebGL restored nodes={restored_counts['total']}"
        )
    finally:
        context.close()
