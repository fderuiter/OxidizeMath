//! Session-scoped WebAssembly runner retention, synchronous fatal panic bridge,
//! and first-frame boot sentinel handoff.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once, OnceLock};

static BRIDGE_INIT: Once = Once::new();
static FIRST_FRAME_REPORTED: AtomicBool = AtomicBool::new(false);

const MAX_RECORDED_FATAL_SUMMARIES: usize = 32;

fn recorded_fatal_panics() -> &'static Mutex<Vec<String>> {
    static SLOT: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(Vec::new()))
}

/// Formats a [`diagnostics::DiagnosticEvent`] into a crash summary when its severity is `Fatal`.
pub(crate) fn format_fatal_diagnostic(event: &diagnostics::DiagnosticEvent) -> Option<String> {
    if event.severity != diagnostics::Severity::Fatal {
        return None;
    }
    let mut summary = event.message.clone();
    if let Some(loc) = event
        .metadata
        .get("location")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        summary.push_str("\nLocation: ");
        summary.push_str(loc);
    }
    if let Some(thread_name) = event
        .thread_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        summary.push_str("\nThread: ");
        summary.push_str(thread_name);
    }
    Some(summary)
}

/// Registers a synchronous listener on [`diagnostics::global_bus`] that forwards `Severity::Fatal`
/// panic summaries to the browser recovery shell before WebAssembly `panic = "abort"` traps.
pub(crate) fn init_fatal_panic_bridge() {
    BRIDGE_INIT.call_once(|| {
        diagnostics::global_bus().register_listener(|event| {
            let Some(summary) = format_fatal_diagnostic(event) else {
                return;
            };
            let mut guard = recorded_fatal_panics()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if guard.len() >= MAX_RECORDED_FATAL_SUMMARIES {
                guard.remove(0);
            }
            guard.push(summary.clone());
            drop(guard);
            #[cfg(target_arch = "wasm32")]
            wasm_support::forward_fatal_panic_to_js(&summary);
        });
    });
}

/// Clears the browser boot-in-progress sentinel once the first interactive frame finishes rendering.
pub(crate) fn notify_first_interactive_frame_rendered() {
    if !FIRST_FRAME_REPORTED.swap(true, Ordering::Relaxed) {
        #[cfg(target_arch = "wasm32")]
        wasm_support::mark_first_frame_in_js();
    }
}

/// Reads a validated shell `html[data-theme]` override when `localStorage["app"]` was quarantined.
pub(crate) fn read_document_shell_theme() -> Option<crate::app::ThemeMode> {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_support::read_document_data_theme()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        None
    }
}

/// Returns `true` when the browser error boundary or workspace reset flow is active so
/// `eframe` canvas blur / `pagehide` callbacks do not overwrite quarantined `localStorage["app"]`.
pub(crate) fn should_suppress_app_save() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_support::is_recovery_or_reset_active()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use wasm_support::keep_web_runner_alive;

#[cfg(target_arch = "wasm32")]
mod wasm_support {
    use std::cell::RefCell;
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::{JsCast, JsValue};

    const BOOT_SENTINEL_KEY: &str = "math_explorer_boot_sentinel";

    thread_local! {
        static SESSION_WEB_RUNNER: RefCell<Option<eframe::WebRunner>> = const { RefCell::new(None) };
        static PANIC_READER_CLOSURE: RefCell<Option<Closure<dyn Fn() -> JsValue>>> = const { RefCell::new(None) };
        static RUNNER_DESTROY_CLOSURE: RefCell<Option<Closure<dyn Fn()>>> = const { RefCell::new(None) };
    }

    // theory_verification!
    pub(crate) fn keep_web_runner_alive(
        runner: eframe::WebRunner,
        window: Option<&web_sys::Window>,
    ) {
        SESSION_WEB_RUNNER.with(|slot| {
            *slot.borrow_mut() = Some(runner);
        });
        if let Some(win) = window {
            install_panic_summary_reader(win);
            install_runner_destroy_hook(win);
        }
    }

    // theory_verification!
    fn install_runner_destroy_hook(win: &web_sys::Window) {
        RUNNER_DESTROY_CLOSURE.with(|slot| {
            if slot.borrow().is_none() {
                let closer = Closure::wrap(Box::new(|| {
                    SESSION_WEB_RUNNER.with(|runner_slot| {
                        if let Ok(mut guard) = runner_slot.try_borrow_mut() {
                            if let Some(runner) = guard.take() {
                                runner.destroy();
                            }
                        }
                    });
                }) as Box<dyn Fn()>);
                let _ = js_sys::Reflect::set(
                    win.as_ref(),
                    &JsValue::from_str("__mathExplorerDestroyRunner"),
                    closer.as_ref(),
                );
                *slot.borrow_mut() = Some(closer);
            }
        });
    }

    // theory_verification!
    fn install_panic_summary_reader(win: &web_sys::Window) {
        PANIC_READER_CLOSURE.with(|slot| {
            if slot.borrow().is_none() {
                let reader = Closure::wrap(Box::new(|| -> JsValue {
                    read_runner_panic_summary()
                        .map_or(JsValue::NULL, |text| JsValue::from_str(&text))
                }) as Box<dyn Fn() -> JsValue>);
                let _ = js_sys::Reflect::set(
                    win.as_ref(),
                    &JsValue::from_str("__mathExplorerReadPanicSummary"),
                    reader.as_ref(),
                );
                *slot.borrow_mut() = Some(reader);
            }
        });
    }

    // theory_verification!
    fn read_runner_panic_summary() -> Option<String> {
        SESSION_WEB_RUNNER.with(|slot| {
            let guard = slot.try_borrow().ok()?;
            let runner = guard.as_ref()?;
            let summary = runner.panic_summary()?;
            let message = summary.message();
            let callstack = summary.callstack();
            if callstack.trim().is_empty() {
                Some(message)
            } else {
                Some(format!("{message}\n{callstack}"))
            }
        })
    }

    // theory_verification!
    pub(super) fn forward_fatal_panic_to_js(summary: &str) {
        let Some(win) = web_sys::window() else {
            return;
        };
        let payload = JsValue::from_str(summary);
        let _ = js_sys::Reflect::set(
            win.as_ref(),
            &JsValue::from_str("__mathExplorerLastFatalPanic"),
            &payload,
        );
        invoke_boot_method(&win, "reportFatalPanic", Some(&payload));
    }

    // theory_verification!
    pub(super) fn mark_first_frame_in_js() {
        let Some(win) = web_sys::window() else {
            return;
        };
        invoke_boot_method(&win, "markFirstFrameRendered", None);
        if let Ok(Some(storage)) = win.local_storage() {
            let _ = storage.remove_item(BOOT_SENTINEL_KEY);
        }
    }

    // theory_verification!
    fn invoke_boot_method(win: &web_sys::Window, method: &str, arg: Option<&JsValue>) {
        let Ok(boot) = js_sys::Reflect::get(win.as_ref(), &JsValue::from_str("__mathExplorerBoot"))
        else {
            return;
        };
        if boot.is_undefined() || boot.is_null() {
            return;
        }
        let Ok(prop) = js_sys::Reflect::get(&boot, &JsValue::from_str(method)) else {
            return;
        };
        let Ok(func) = prop.dyn_into::<js_sys::Function>() else {
            return;
        };
        if let Some(value) = arg {
            let _ = func.call1(&boot, value);
        } else {
            let _ = func.call0(&boot);
        }
    }

    // theory_verification!
    pub(super) fn read_document_data_theme() -> Option<crate::app::ThemeMode> {
        let win = web_sys::window()?;
        let doc = win.document()?;
        let root = doc.document_element()?;
        let token = root.get_attribute("data-theme")?;
        crate::app::ThemeMode::from_token(&token)
    }

    // theory_verification!
    pub(super) fn is_recovery_or_reset_active() -> bool {
        let Some(root) = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.document_element())
        else {
            return false;
        };
        root.get_attribute("data-boot-failed").as_deref() == Some("true")
            || root.get_attribute("data-workspace-resetting").as_deref() == Some("true")
    }

    /// Diagnostic probe used by end-to-end browser verification to exercise real WASM `panic!` traps.
    #[wasm_bindgen::prelude::wasm_bindgen]
    pub fn trigger_wasm_panic_probe(reason: &str) {
        panic!("{reason}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_format_fatal_diagnostic_ignores_non_fatal_and_formats_location() {
        let mut metadata = HashMap::new();
        metadata.insert("location".to_string(), "src/sim.rs:42:9".to_string());
        let non_fatal = diagnostics::DiagnosticEvent {
            severity: diagnostics::Severity::Error,
            message: "recoverable error".to_string(),
            metadata: metadata.clone(),
            thread_name: Some("main".to_string()),
        };
        assert!(format_fatal_diagnostic(&non_fatal).is_none());

        let fatal = diagnostics::DiagnosticEvent {
            severity: diagnostics::Severity::Fatal,
            message: "Thread panicked: unreachable state".to_string(),
            metadata,
            thread_name: Some("wasm-main".to_string()),
        };
        let formatted = format_fatal_diagnostic(&fatal).expect("fatal event should format");
        assert!(formatted.contains("Thread panicked: unreachable state"));
        assert!(formatted.contains("Location: src/sim.rs:42:9"));
        assert!(formatted.contains("Thread: wasm-main"));
    }

    #[test]
    fn test_fatal_panic_bridge_captures_panic_hook_events() {
        diagnostics::init_panic_hook();
        init_fatal_panic_bridge();

        let _ = std::panic::catch_unwind(|| {
            panic!("wasm_bridge_unit_probe");
        });

        let guard = recorded_fatal_panics()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let recorded = guard
            .iter()
            .find(|entry| entry.contains("Thread panicked: wasm_bridge_unit_probe"))
            .cloned()
            .expect("fatal panic should be recorded by bridge");
        drop(guard);
        assert!(recorded.contains("Location: "));
        assert!(!recorded.contains('\\'));
    }
}
