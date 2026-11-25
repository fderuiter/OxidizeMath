//! Fatal panic bridge forwarding Rust diagnostics to the browser shell error boundary.
//!
//! Registers a `Severity::Fatal` listener on [`diagnostics::global_bus`] before WebAssembly
//! executes its `unreachable` abort instruction so active audio streams are silenced immediately,
//! panic messages and `file:line` source locations reach `#error_details`, and trapped
//! `eframe::WebRunner` instances are never re-entered.

use diagnostics::{DiagnosticEvent, Severity};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};

static FATAL_PANIC_OCCURRED: AtomicBool = AtomicBool::new(false);
static INSTALL_PANIC_BRIDGE_ONCE: Once = Once::new();
static LAST_FATAL_PANIC_SUMMARY: Mutex<Option<String>> = Mutex::new(None);

/// Returns `true` once a `Severity::Fatal` diagnostic event has been observed in this session.
#[must_use]
pub fn has_fatal_panic_occurred() -> bool {
    FATAL_PANIC_OCCURRED.load(Ordering::SeqCst)
}

/// Returns the most recently formatted fatal panic summary recorded by the crash bridge.
#[cfg(test)]
#[must_use]
pub fn last_fatal_panic_summary() -> Option<String> {
    LAST_FATAL_PANIC_SUMMARY
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
}

/// Resets recorded fatal panic flags between isolated unit tests.
#[cfg(test)]
pub fn reset_fatal_panic_state_for_tests() {
    FATAL_PANIC_OCCURRED.store(false, Ordering::SeqCst);
    if let Ok(mut guard) = LAST_FATAL_PANIC_SUMMARY.lock() {
        *guard = None;
    }
}

/// Formats a `Severity::Fatal` [`DiagnosticEvent`] into `(summary_with_location, location)`.
#[must_use]
pub fn format_fatal_diagnostic(event: &DiagnosticEvent) -> (String, String) {
    let location = event
        .metadata
        .get("location")
        .map(|loc| loc.trim().to_string())
        .unwrap_or_default();
    let base_msg = event.message.trim();
    let message = if base_msg.is_empty() {
        "Thread panicked in WebAssembly runtime"
    } else {
        base_msg
    };
    let summary = if !location.is_empty() && !message.contains(&location) {
        format!("{message} (at {location})")
    } else {
        message.to_string()
    };
    (summary, location)
}

/// Processes a diagnostic event, forwarding `Severity::Fatal` panics to audio and shell handlers.
pub fn handle_diagnostic_event(event: &DiagnosticEvent) {
    if event.severity != Severity::Fatal {
        return;
    }
    FATAL_PANIC_OCCURRED.store(true, Ordering::SeqCst);
    crate::audio::silence_all_audio_on_fatal_panic();

    let (summary, location) = format_fatal_diagnostic(event);
    if let Ok(mut guard) = LAST_FATAL_PANIC_SUMMARY.lock() {
        *guard = Some(summary.clone());
    }

    #[cfg(target_arch = "wasm32")]
    forward_fatal_panic_to_shell(&summary, &location);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = &location;
}

/// Registers an idempotent `Severity::Fatal` listener on [`diagnostics::global_bus`].
pub fn install_fatal_panic_bridge() {
    INSTALL_PANIC_BRIDGE_ONCE.call_once(|| {
        diagnostics::global_bus().register_listener(|event| {
            handle_diagnostic_event(event);
        });
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn register_active_audio_graph(
    ctx: &web_sys::AudioContext,
    master_gain: &web_sys::GainNode,
) {
    use wasm_bindgen::{JsCast, JsValue};
    let Some(win) = web_sys::window() else {
        return;
    };
    let Ok(boot) = js_sys::Reflect::get(win.as_ref(), &JsValue::from_str("__mathExplorerBoot"))
    else {
        return;
    };
    if boot.is_null() || boot.is_undefined() {
        return;
    }
    if let Ok(reg_val) = js_sys::Reflect::get(&boot, &JsValue::from_str("registerAudioStream")) {
        if let Ok(func) = reg_val.dyn_into::<js_sys::Function>() {
            let _ = func.call2(&boot, ctx.as_ref(), master_gain.as_ref());
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn try_invoke_js_panic_bridge(win: &web_sys::Window, summary: &str, location: &str) -> bool {
    use wasm_bindgen::{JsCast, JsValue};
    let Ok(boot) = js_sys::Reflect::get(win.as_ref(), &JsValue::from_str("__mathExplorerBoot"))
    else {
        return false;
    };
    if boot.is_null() || boot.is_undefined() {
        return false;
    }
    let Ok(record_val) = js_sys::Reflect::get(&boot, &JsValue::from_str("recordFatalPanic")) else {
        return false;
    };
    let Ok(func) = record_val.dyn_into::<js_sys::Function>() else {
        return false;
    };
    func.call2(
        &boot,
        &JsValue::from_str(summary),
        &JsValue::from_str(location),
    )
    .is_ok()
}

#[cfg(target_arch = "wasm32")]
fn apply_fallback_runtime_dom(doc: &web_sys::Document, summary: &str) {
    if let Some(root) = doc.document_element() {
        let _ = root.set_attribute("data-boot-initialized", "true");
        let _ = root.set_attribute("data-boot-failed", "true");
    }
    if let Some(loading) = doc.get_element_by_id("loading_indicator") {
        let _ = loading.set_attribute("aria-busy", "false");
        loading.remove();
    }
    if let Some(canvas) = doc.get_element_by_id("the_canvas_id") {
        let _ = canvas.set_attribute("style", "display: none;");
    }
    if let Some(err_el) = doc.get_element_by_id("error_container") {
        let _ = err_el.set_attribute("data-error-category", "runtime");
        let _ = err_el.set_attribute("data-error-active", "true");
        let _ = err_el.set_attribute("style", "display: flex;");
    }
    if let Some(title_el) = doc.get_element_by_id("error_title") {
        title_el.set_text_content(Some("Application Runtime Error"));
    }
    if let Some(details_el) = doc.get_element_by_id("error_details") {
        details_el.set_text_content(Some(summary));
        let _ = details_el.set_attribute("style", "display: block;");
    }
    for btn_id in ["reset_workspace_button", "download_workspace_button"] {
        if let Some(btn) = doc.get_element_by_id(btn_id) {
            let _ = btn.remove_attribute("hidden");
        }
    }
    crate::accessibility::announce_status_with_priority(
        "Application runtime error in Math Explorer. Download your saved workspace backup, select Reload Page, or select Reset Saved Workspace and Reload.",
        "assertive",
    );
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn forward_fatal_panic_to_shell(summary: &str, location: &str) {
    let Some(win) = web_sys::window() else {
        return;
    };
    if try_invoke_js_panic_bridge(&win, summary, location) {
        return;
    }
    if let Some(doc) = win.document() {
        apply_fallback_runtime_dom(&doc, summary);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn fatal_listener_formats_location_silences_audio_and_sets_flag() {
        let _guard = crate::reflective_ui::tests::TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_fatal_panic_state_for_tests();
        install_fatal_panic_bridge();

        crate::audio::set_muted_from_user_gesture_target(false);
        assert!(!crate::audio::snapshot_ui_state().muted);

        let mut meta = HashMap::new();
        meta.insert(
            "location".to_string(),
            "math_explorer_gui/src/tabs/solver.rs:142".to_string(),
        );
        diagnostics::global_bus().emit(DiagnosticEvent {
            severity: Severity::Fatal,
            message: "Thread panicked: attempt to divide by zero".to_string(),
            metadata: meta,
            thread_name: Some("main".to_string()),
        });

        assert!(has_fatal_panic_occurred());
        assert!(crate::audio::snapshot_ui_state().muted);
        let recorded = last_fatal_panic_summary().expect("fatal summary recorded");
        assert!(recorded.contains("attempt to divide by zero"));
        assert!(recorded.contains("math_explorer_gui/src/tabs/solver.rs:142"));

        reset_fatal_panic_state_for_tests();
    }

    #[test]
    fn non_fatal_diagnostics_do_not_trip_crash_boundary() {
        reset_fatal_panic_state_for_tests();
        handle_diagnostic_event(&DiagnosticEvent {
            severity: Severity::Error,
            message: "Recoverable solver warning".to_string(),
            metadata: HashMap::new(),
            thread_name: Some("main".to_string()),
        });
        assert!(!has_fatal_panic_occurred());
        assert!(last_fatal_panic_summary().is_none());
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "crash_boundary_wasm_tests.rs"]
mod wasm_tests;
