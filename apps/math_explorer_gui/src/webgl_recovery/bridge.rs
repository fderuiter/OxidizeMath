//! DOM, `localStorage`, and `window.__mathExplorerBoot` bridge helpers for [`super`].

use crate::accessibility;
use std::rc::Rc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};

pub(super) const LOSS_STATUS_MSG: &str =
    "WebGL graphics context lost. Attempting automatic restoration; your workspace is preserved.";
pub(super) const RESTORED_STATUS_MSG: &str =
    "WebGL graphics context restored. Simulation workspace resumed.";

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

pub(super) fn set_canvas_inert(doc: &web_sys::Document, inert: bool) {
    let Some(canvas) = doc.get_element_by_id("the_canvas_id") else {
        return;
    };
    if inert {
        let _ = canvas.set_attribute("inert", "");
        let _ = canvas.set_attribute("aria-hidden", "true");
        if let Ok(html_canvas) = canvas.dyn_into::<web_sys::HtmlElement>() {
            let _ = html_canvas.blur();
        }
    } else {
        let _ = canvas.remove_attribute("inert");
        let _ = canvas.remove_attribute("aria-hidden");
    }
}

fn focus_active_overlay_control(doc: &web_sys::Document) {
    let download_btn = doc.get_element_by_id("webgl_recovery_download_button");
    let reload_btn = doc.get_element_by_id("webgl_recovery_reload_button");
    let storage_blocked = download_btn
        .as_ref()
        .and_then(|b| b.get_attribute("data-storage-blocked"))
        .as_deref()
        == Some("true");
    let preferred = if storage_blocked
        && download_btn
            .as_ref()
            .is_some_and(|b| !b.has_attribute("hidden"))
    {
        download_btn
    } else if reload_btn
        .as_ref()
        .is_some_and(|b| !b.has_attribute("hidden"))
    {
        reload_btn
    } else if download_btn
        .as_ref()
        .is_some_and(|b| !b.has_attribute("hidden"))
    {
        download_btn
    } else {
        None
    };
    if let Some(el) = preferred.and_then(|b| b.dyn_into::<web_sys::HtmlElement>().ok()) {
        let _ = el.focus();
    }
}

pub(crate) fn set_overlay_visible(doc: &web_sys::Document, visible: bool) {
    if visible {
        set_canvas_inert(doc, true);
        crate::bind_fallback_reload_listener(doc);
    }
    if let Some(overlay) = doc.get_element_by_id("webgl_recovery_overlay") {
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
    if visible {
        focus_active_overlay_control(doc);
    }
}

fn bind_download_button_by_id<F>(doc: &web_sys::Document, id: &str, read_snapshot: Rc<F>)
where
    F: Fn() -> (String, String) + 'static,
{
    let Some(btn) = doc.get_element_by_id(id) else {
        return;
    };
    if btn.get_attribute("data-download-bound").as_deref() == Some("true") {
        return;
    }
    let _ = btn.set_attribute("data-download-bound", "true");
    let listener = Closure::wrap(Box::new(move |event: web_sys::Event| {
        event.prevent_default();
        let (envelope_json, filename) = read_snapshot();
        if !envelope_json.is_empty() {
            let _ = crate::web_io::download_browser_file(&filename, envelope_json.as_bytes());
        }
    }) as Box<dyn FnMut(web_sys::Event)>);
    let _ = btn.add_event_listener_with_callback("click", listener.as_ref().unchecked_ref());
    listener.forget();
}

pub(super) fn bind_fallback_download_listener<F>(doc: &web_sys::Document, read_snapshot: F)
where
    F: Fn() -> (String, String) + 'static,
{
    let shared = Rc::new(read_snapshot);
    bind_download_button_by_id(doc, "download_workspace_button", Rc::clone(&shared));
    bind_download_button_by_id(doc, "webgl_recovery_download_button", shared);
}

pub(crate) fn sync_fallback_download_buttons(
    doc: &web_sys::Document,
    envelope_json: &str,
    storage_blocked: bool,
) {
    let blocked_str = if storage_blocked { "true" } else { "false" };
    for id in [
        "download_workspace_button",
        "webgl_recovery_download_button",
    ] {
        if let Some(btn) = doc.get_element_by_id(id) {
            let _ = btn.set_attribute("data-storage-blocked", blocked_str);
            if envelope_json.is_empty() {
                let _ = btn.set_attribute("hidden", "");
            } else {
                let _ = btn.remove_attribute("hidden");
            }
        }
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

pub(crate) fn notify_recovery_completed(window: Option<&web_sys::Window>, doc: &web_sys::Document) {
    if let Some(boot) = boot_api(window) {
        if let Ok(func_val) =
            js_sys::Reflect::get(&boot, &JsValue::from_str("completeWebglContextRecovery"))
        {
            if let Ok(func) = func_val.dyn_into::<js_sys::Function>() {
                if func
                    .call1(&boot, &JsValue::from_str(RESTORED_STATUS_MSG))
                    .is_ok()
                {
                    set_canvas_inert(doc, false);
                    return;
                }
            }
        }
    }
    set_overlay_visible(doc, false);
    set_canvas_inert(doc, false);
    if let Some(canvas) = doc
        .get_element_by_id("the_canvas_id")
        .and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok())
    {
        let _ = canvas.style().remove_property("display");
        let _ = canvas.focus();
    }
    accessibility::announce_status_with_priority(RESTORED_STATUS_MSG, "polite");
}

pub(super) fn notify_recovery_failed(
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
    crate::handle_webgl_failure(Some(doc), detail);
    sync_fallback_download_buttons(doc, envelope_json, storage_blocked);
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
