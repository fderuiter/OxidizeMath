use super::{
    bind_fallback_reload_listener, clear_global_wasm_bindings, finish_wasm_startup,
    handle_webgl_failure, is_error_boundary_active, mark_boot_initialized, notify_js_boot_complete,
    select_webgl_error_category,
};
use std::cell::Cell;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

fn reset_startup_dom() -> (web_sys::Window, web_sys::Document) {
    let window = web_sys::window().expect("browser window must exist");
    let doc = window.document().expect("browser document must exist");
    if let Some(root) = doc.document_element() {
        let _ = root.remove_attribute("data-boot-initialized");
        let _ = root.remove_attribute("data-boot-failed");
    }
    let body = doc.body().expect("document body must exist");
    for id in ["loading_indicator", "the_canvas_id", "error_container"] {
        if let Some(el) = doc.get_element_by_id(id) {
            el.remove();
        }
    }

    let loading = doc.create_element("div").unwrap();
    loading.set_id("loading_indicator");
    let _ = loading.set_attribute("aria-busy", "true");
    let _ = body.append_child(&loading);

    let canvas = doc.create_element("canvas").unwrap();
    canvas.set_id("the_canvas_id");
    let _ = canvas.set_attribute("tabindex", "0");
    let _ = body.append_child(&canvas);

    let err_box = doc.create_element("div").unwrap();
    err_box.set_id("error_container");
    let _ = err_box.set_attribute("role", "alert");
    let _ = err_box.set_attribute("style", "display: none;");
    err_box.set_inner_html(
        "<h1 id=\"error_title\"></h1><p id=\"error_message\"></p>\
         <button id=\"webgl_reload_button\" type=\"button\">Reload</button>\
         <pre id=\"error_details\" style=\"display: none;\"></pre>",
    );
    let _ = body.append_child(&err_box);

    (window, doc)
}

#[wasm_bindgen_test]
fn wasm_preboot_dom_flags_and_js_boot_notifications_work() {
    let (window, doc) = reset_startup_dom();
    assert!(!is_error_boundary_active(&doc));

    mark_boot_initialized(&doc);
    assert_eq!(
        doc.document_element()
            .and_then(|r| r.get_attribute("data-boot-initialized"))
            .as_deref(),
        Some("true")
    );

    let _ = js_sys::Reflect::set(
        window.as_ref(),
        &wasm_bindgen::JsValue::from_str("wasmBindings"),
        &wasm_bindgen::JsValue::from_str("stub"),
    );
    clear_global_wasm_bindings(Some(&window));
    assert!(js_sys::Reflect::get(
        window.as_ref(),
        &wasm_bindgen::JsValue::from_str("wasmBindings")
    )
    .expect("Reflect::get should succeed")
    .is_undefined());

    let called = Rc::new(Cell::new(false));
    let called_clone = Rc::clone(&called);
    let cb = wasm_bindgen::closure::Closure::wrap(Box::new(move || {
        called_clone.set(true);
    }) as Box<dyn FnMut()>);
    let boot_obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &boot_obj,
        &wasm_bindgen::JsValue::from_str("markBootComplete"),
        cb.as_ref(),
    );
    let _ = js_sys::Reflect::set(
        window.as_ref(),
        &wasm_bindgen::JsValue::from_str("__mathExplorerBoot"),
        &boot_obj,
    );
    notify_js_boot_complete(Some(&window));
    assert!(called.get(), "markBootComplete hook should be invoked");
    let _ = js_sys::Reflect::delete_property(
        window.as_ref(),
        &wasm_bindgen::JsValue::from_str("__mathExplorerBoot"),
    );
    drop(cb);

    bind_fallback_reload_listener(&doc);
    bind_fallback_reload_listener(&doc);
    assert_eq!(
        doc.get_element_by_id("webgl_reload_button")
            .and_then(|b| b.get_attribute("data-reload-bound"))
            .as_deref(),
        Some("true")
    );
}

#[wasm_bindgen_test]
fn wasm_handle_webgl_failure_renders_accessible_alert_and_focuses_reload() {
    let (_window, doc) = reset_startup_dom();
    select_webgl_error_category(&doc);
    assert!(is_error_boundary_active(&doc));

    let (_window, doc) = reset_startup_dom();
    handle_webgl_failure(Some(&doc), "WebGL2 context unavailable");

    assert!(is_error_boundary_active(&doc));
    assert!(doc.get_element_by_id("loading_indicator").is_none());
    assert_eq!(
        doc.document_element()
            .and_then(|r| r.get_attribute("data-boot-failed"))
            .as_deref(),
        Some("true")
    );
    let err_box = doc.get_element_by_id("error_container").unwrap();
    assert_eq!(
        err_box.get_attribute("data-error-category").as_deref(),
        Some("webgl")
    );
    assert_eq!(
        err_box.get_attribute("data-error-active").as_deref(),
        Some("true")
    );
    assert_eq!(
        doc.get_element_by_id("error_title")
            .and_then(|el| el.text_content())
            .as_deref(),
        Some("WebGL Initialization Failed")
    );
    assert_eq!(
        doc.get_element_by_id("error_details")
            .and_then(|el| el.text_content())
            .as_deref(),
        Some("WebGL2 context unavailable")
    );
    assert_eq!(
        doc.active_element().map(|el| el.id()).as_deref(),
        Some("webgl_reload_button")
    );
}

#[wasm_bindgen_test]
fn wasm_finish_startup_focuses_canvas_on_ok_and_respects_active_error_boundary() {
    let (window, doc) = reset_startup_dom();
    let canvas = doc
        .get_element_by_id("the_canvas_id")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let runner = eframe::WebRunner::new();
    finish_wasm_startup(&doc, Some(&window), &canvas, runner, Ok(()));
    assert!(doc.get_element_by_id("loading_indicator").is_none());
    assert_eq!(
        doc.active_element().map(|el| el.id()).as_deref(),
        Some("the_canvas_id")
    );

    let (window, doc) = reset_startup_dom();
    let canvas = doc
        .get_element_by_id("the_canvas_id")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let runner = eframe::WebRunner::new();
    let err_box = doc.get_element_by_id("error_container").unwrap();
    let _ = err_box.set_attribute("style", "display: flex;");
    let _ = js_sys::Reflect::set(
        window.as_ref(),
        &wasm_bindgen::JsValue::from_str("wasmBindings"),
        &wasm_bindgen::JsValue::TRUE,
    );
    finish_wasm_startup(&doc, Some(&window), &canvas, runner, Ok(()));
    assert!(
        doc.get_element_by_id("loading_indicator").is_some(),
        "active error boundary must abort finish_wasm_startup before mutating loading state"
    );
    assert!(js_sys::Reflect::get(
        window.as_ref(),
        &wasm_bindgen::JsValue::from_str("wasmBindings")
    )
    .unwrap()
    .is_undefined());
    let _ = doc
        .get_element_by_id("the_canvas_id")
        .and_then(|c| c.dyn_into::<web_sys::HtmlElement>().ok())
        .map(|c| c.blur());
}

#[wasm_bindgen_test]
fn wasm_webgl_context_lost_prevents_default_and_reveals_recovery_overlay() {
    let (window, doc) = reset_startup_dom();
    let body = doc.body().unwrap();
    if let Some(prev) = doc.get_element_by_id("webgl_recovery_overlay") {
        prev.remove();
    }
    let overlay = doc.create_element("div").unwrap();
    overlay.set_id("webgl_recovery_overlay");
    let _ = overlay.set_attribute("hidden", "");
    let _ = overlay.set_attribute("aria-busy", "false");
    let _ = body.append_child(&overlay);

    let canvas = doc
        .get_element_by_id("the_canvas_id")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let runner = eframe::WebRunner::new();
    finish_wasm_startup(&doc, Some(&window), &canvas, runner, Ok(()));

    let init = web_sys::EventInit::new();
    init.set_cancelable(true);
    let lost_event = web_sys::Event::new_with_event_init_dict("webglcontextlost", &init).unwrap();
    let _ = canvas.dispatch_event(&lost_event).unwrap();

    assert!(
        lost_event.default_prevented(),
        "webglcontextlost handler must call prevent_default()"
    );
    assert!(
        !overlay.has_attribute("hidden"),
        "#webgl_recovery_overlay must unhide while awaiting WebGL context restoration"
    );
    assert_eq!(overlay.get_attribute("aria-busy").as_deref(), Some("true"));
    canvas.remove();
    overlay.remove();
}
