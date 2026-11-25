//! DOM, `localStorage`, and `window.__mathExplorerBoot` bridge helpers for [`super`].

use crate::accessibility;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};

pub(super) const LOSS_STATUS_MSG: &str =
    "WebGL graphics context lost. Attempting automatic restoration; your workspace is preserved.";
pub(super) const RESTORED_STATUS_MSG: &str =
    "WebGL graphics context restored. Simulation workspace resumed.";
const CONTEXT_LOST_TITLE: &str = "WebGL Graphics Context Lost";
const CONTEXT_LOST_MESSAGE: &str =
    "Your browser lost its WebGL graphics context and could not restore hardware rendering automatically. Your workspace state is preserved\u{2014}select Reload Page to resume or download a workspace backup (.mws) below.";
const CONTEXT_LOST_BLOCKED_STORAGE_MESSAGE: &str =
    "Your browser lost its WebGL graphics context and blocked local storage writes. Select Download Saved Workspace (.mws) below before reloading so you do not lose your session.";
const CONTEXT_LOST_ALERT: &str =
    "WebGL graphics context lost and could not be restored automatically. Select Reload Page to recover.";
const CONTEXT_LOST_BLOCKED_STORAGE_ALERT: &str =
    "WebGL graphics context lost and browser storage blocked saving. Download your saved workspace envelope before selecting Reload Page.";

fn boot_api(window: Option<&web_sys::Window>) -> Option<JsValue> {
    let win = window?;
    let boot = js_sys::Reflect::get(win.as_ref(), &JsValue::from_str("__mathExplorerBoot")).ok()?;
    if boot.is_undefined() || boot.is_null() {
        None
    } else {
        Some(boot)
    }
}

pub(super) fn persist_envelope_to_local_storage(
    window: Option<&web_sys::Window>,
    envelope_json: &str,
) -> bool {
    let Some(win) = window else {
        return false;
    };
    let Ok(Some(storage)) = win.local_storage() else {
        return false;
    };
    storage.set_item(eframe::APP_KEY, envelope_json).is_ok()
}

pub(super) fn set_overlay_visible(doc: &web_sys::Document, visible: bool) {
    let Some(overlay) = doc.get_element_by_id("webgl_recovery_overlay") else {
        return;
    };
    if visible {
        let _ = overlay.remove_attribute("hidden");
        let _ = overlay.set_attribute("aria-busy", "true");
        let _ = overlay.set_attribute("style", "display: flex;");
    } else {
        let _ = overlay.set_attribute("aria-busy", "false");
        let _ = overlay.set_attribute("hidden", "");
        let _ = overlay.set_attribute("style", "display: none;");
    }
}

pub(super) fn bind_fallback_download_listener<F>(doc: &web_sys::Document, read_snapshot: F)
where
    F: Fn() -> (String, String) + 'static,
{
    let Some(btn) = doc.get_element_by_id("download_workspace_button") else {
        return;
    };
    if btn.get_attribute("data-download-bound").as_deref() == Some("true") {
        return;
    }
    let _ = btn.set_attribute("data-download-bound", "true");
    let listener = Closure::wrap(Box::new(move |event: web_sys::Event| {
        event.prevent_default();
        let (mut envelope_json, mut filename) = read_snapshot();
        if envelope_json.is_empty() {
            if let Some(stored) =
                crate::read_saved_workspace_from_storage(web_sys::window().as_ref())
            {
                envelope_json = stored;
            }
        }
        if filename.is_empty() {
            filename = crate::DEFAULT_RECOVERY_FILENAME.to_string();
        }
        if !envelope_json.is_empty()
            && crate::web_io::download_browser_file(&filename, envelope_json.as_bytes()).is_ok()
        {
            accessibility::announce_status_with_priority(
                &format!("Downloaded saved workspace envelope {filename}."),
                "polite",
            );
        }
    }) as Box<dyn FnMut(web_sys::Event)>);
    let _ = btn.add_event_listener_with_callback("click", listener.as_ref().unchecked_ref());
    listener.forget();
}

fn reveal_fallback_download_button(
    window: Option<&web_sys::Window>,
    doc: &web_sys::Document,
    envelope_json: &str,
    storage_blocked: bool,
) {
    let Some(btn) = doc.get_element_by_id("download_workspace_button") else {
        return;
    };
    let blocked_str = if storage_blocked { "true" } else { "false" };
    let _ = btn.set_attribute("data-storage-blocked", blocked_str);
    let has_workspace =
        !envelope_json.is_empty() || crate::read_saved_workspace_from_storage(window).is_some();
    if has_workspace {
        let _ = btn.remove_attribute("hidden");
    } else {
        let _ = btn.set_attribute("hidden", "");
    }
}

pub(super) fn notify_js_begin_recovery(
    window: Option<&web_sys::Window>,
    envelope_json: &str,
    filename: &str,
    storage_blocked: bool,
) -> bool {
    let Some(boot) = boot_api(window) else {
        return false;
    };
    let Ok(func_val) = js_sys::Reflect::get(&boot, &JsValue::from_str("beginWebglContextRecovery"))
    else {
        return false;
    };
    let Ok(func) = func_val.dyn_into::<js_sys::Function>() else {
        return false;
    };
    let opts = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &opts,
        &JsValue::from_str("envelopeJson"),
        &JsValue::from_str(envelope_json),
    );
    let _ = js_sys::Reflect::set(
        &opts,
        &JsValue::from_str("filename"),
        &JsValue::from_str(filename),
    );
    let _ = js_sys::Reflect::set(
        &opts,
        &JsValue::from_str("storageBlocked"),
        &JsValue::from_bool(storage_blocked),
    );
    let _ = js_sys::Reflect::set(
        &opts,
        &JsValue::from_str("statusMessage"),
        &JsValue::from_str(LOSS_STATUS_MSG),
    );
    func.call1(&boot, &opts).is_ok()
}

pub(super) fn notify_recovery_completed(window: Option<&web_sys::Window>, doc: &web_sys::Document) {
    if let Some(boot) = boot_api(window) {
        if let Ok(func_val) =
            js_sys::Reflect::get(&boot, &JsValue::from_str("completeWebglContextRecovery"))
        {
            if let Ok(func) = func_val.dyn_into::<js_sys::Function>() {
                if func
                    .call1(&boot, &JsValue::from_str(RESTORED_STATUS_MSG))
                    .is_ok()
                {
                    return;
                }
            }
        }
    }
    set_overlay_visible(doc, false);
    if let Some(canvas) = doc
        .get_element_by_id("the_canvas_id")
        .and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok())
    {
        let _ = canvas.style().remove_property("display");
        let _ = canvas.focus();
    }
    accessibility::announce_status_with_priority(RESTORED_STATUS_MSG, "polite");
}

fn apply_context_lost_dom_copy(doc: &web_sys::Document, detail: &str, storage_blocked: bool) {
    let blocked_str = if storage_blocked { "true" } else { "false" };
    if let Some(error_container) = doc.get_element_by_id("error_container") {
        let _ = error_container.set_attribute("data-error-category", "webgl");
        let _ = error_container.set_attribute("data-error-active", "true");
        let _ = error_container.set_attribute("data-storage-blocked", blocked_str);
        let _ = error_container.set_attribute("style", "display: flex;");
    }
    if let Some(title_el) = doc.get_element_by_id("error_title") {
        title_el.set_text_content(Some(CONTEXT_LOST_TITLE));
    }
    if let Some(msg_el) = doc.get_element_by_id("error_message") {
        let message = if storage_blocked {
            CONTEXT_LOST_BLOCKED_STORAGE_MESSAGE
        } else {
            CONTEXT_LOST_MESSAGE
        };
        msg_el.set_text_content(Some(message));
    }
    if let Some(details_el) = doc.get_element_by_id("error_details") {
        if detail.is_empty() {
            details_el.set_text_content(Some(""));
            let _ = details_el.set_attribute("style", "display: none;");
        } else {
            details_el.set_text_content(Some(detail));
            let _ = details_el.set_attribute("style", "display: block;");
        }
    }
}

fn render_fallback_context_lost_failure(
    window: Option<&web_sys::Window>,
    doc: &web_sys::Document,
    detail: &str,
    envelope_json: &str,
    storage_blocked: bool,
) {
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
    apply_context_lost_dom_copy(doc, detail, storage_blocked);
    reveal_fallback_download_button(window, doc, envelope_json, storage_blocked);
    crate::bind_fallback_reload_listener(doc);
    let alert = if storage_blocked {
        CONTEXT_LOST_BLOCKED_STORAGE_ALERT
    } else {
        CONTEXT_LOST_ALERT
    };
    accessibility::announce_status_with_priority(alert, "assertive");
    if let Some(reload_btn) = doc
        .get_element_by_id("webgl_reload_button")
        .and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok())
    {
        let _ = reload_btn.focus();
    }
}

pub(crate) fn notify_recovery_failed(
    window: Option<&web_sys::Window>,
    doc: &web_sys::Document,
    detail: &str,
    envelope_json: &str,
    filename: &str,
    storage_blocked: bool,
) {
    set_overlay_visible(doc, false);
    if let Some(boot) = boot_api(window) {
        if let Ok(set_env_val) =
            js_sys::Reflect::get(&boot, &JsValue::from_str("setSavedWorkspaceEnvelope"))
        {
            if let Ok(set_env_fn) = set_env_val.dyn_into::<js_sys::Function>() {
                let _ = set_env_fn.call3(
                    &boot,
                    &JsValue::from_str(envelope_json),
                    &JsValue::from_str(filename),
                    &JsValue::from_bool(storage_blocked),
                );
            }
        }
        if let Ok(fail_val) =
            js_sys::Reflect::get(&boot, &JsValue::from_str("failWebglContextRecovery"))
        {
            if let Ok(fail_fn) = fail_val.dyn_into::<js_sys::Function>() {
                if fail_fn.call1(&boot, &JsValue::from_str(detail)).is_ok() {
                    return;
                }
            }
        }
    }
    render_fallback_context_lost_failure(window, doc, detail, envelope_json, storage_blocked);
}

pub(super) fn register_js_timeout_bridge<F>(window: Option<&web_sys::Window>, mut on_timeout: F)
where
    F: FnMut(String) + 'static,
{
    let Some(boot) = boot_api(window) else {
        return;
    };
    let Ok(reg_val) =
        js_sys::Reflect::get(&boot, &JsValue::from_str("registerWebglRecoveryHandlers"))
    else {
        return;
    };
    let Ok(reg_fn) = reg_val.dyn_into::<js_sys::Function>() else {
        return;
    };

    let timeout_cb = Closure::wrap(Box::new(move |detail: JsValue| {
        let msg = detail
            .as_string()
            .unwrap_or_else(|| "WebGL context restoration timed out.".to_string());
        on_timeout(msg);
    }) as Box<dyn FnMut(JsValue)>);

    let handlers = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &handlers,
        &JsValue::from_str("onTimeout"),
        timeout_cb.as_ref(),
    );
    let _ = reg_fn.call1(&boot, &handlers);
    timeout_cb.forget();
}
