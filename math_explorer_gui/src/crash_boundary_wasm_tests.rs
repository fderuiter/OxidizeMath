//! `wasm32` browser tests verifying fatal panic forwarding to `#error_details` and
//! fallback DOM crash boundary activation.

use super::{handle_diagnostic_event, has_fatal_panic_occurred, reset_fatal_panic_state_for_tests};
use diagnostics::{DiagnosticEvent, Severity};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

fn setup_runtime_crash_dom() -> (web_sys::Window, web_sys::Document) {
    let win = web_sys::window().expect("window");
    let doc = win.document().expect("document");
    let body = doc.body().expect("body");
    if let Some(root) = doc.document_element() {
        let _ = root.remove_attribute("data-boot-initialized");
        let _ = root.remove_attribute("data-boot-failed");
    }
    for id in [
        "loading_indicator",
        "the_canvas_id",
        "error_container",
        "aria-live-assertive",
    ] {
        if let Some(el) = doc.get_element_by_id(id) {
            el.remove();
        }
    }
    let _ =
        js_sys::Reflect::delete_property(win.as_ref(), &JsValue::from_str("__mathExplorerBoot"));

    let assertive = doc.create_element("div").unwrap();
    assertive.set_id("aria-live-assertive");
    assertive.set_inner_html(
        "<span id=\"aria-live-assertive-slot-0\"></span>\
         <span id=\"aria-live-assertive-slot-1\"></span>",
    );
    let _ = body.append_child(&assertive);

    let canvas = doc.create_element("canvas").unwrap();
    canvas.set_id("the_canvas_id");
    let _ = body.append_child(&canvas);

    let err_box = doc.create_element("div").unwrap();
    err_box.set_id("error_container");
    let _ = err_box.set_attribute("style", "display: none;");
    err_box.set_inner_html(
        "<h1 id=\"error_title\"></h1><p id=\"error_message\"></p>\
         <button id=\"webgl_reload_button\" type=\"submit\">Reload</button>\
         <button id=\"download_workspace_button\" type=\"button\" hidden>Download</button>\
         <button id=\"reset_workspace_button\" type=\"button\" hidden>Reset</button>\
         <pre id=\"error_details\" style=\"display: none;\"></pre>",
    );
    let _ = body.append_child(&err_box);
    (win, doc)
}

#[wasm_bindgen_test]
fn wasm_fatal_panic_forwards_to_js_bridge_and_fallback_dom() {
    reset_fatal_panic_state_for_tests();
    let (win, doc) = setup_runtime_crash_dom();
    if let Ok(Some(ls)) = win.local_storage() {
        let _ = ls.set_item("app", "{\"theme\":\"Dark\"}");
    }

    let captured: Rc<RefCell<Option<(String, String)>>> = Rc::new(RefCell::new(None));
    let cap_cb = Rc::clone(&captured);
    let cb = wasm_bindgen::closure::Closure::wrap(Box::new(move |msg: String, loc: String| {
        *cap_cb.borrow_mut() = Some((msg, loc));
    }) as Box<dyn FnMut(String, String)>);
    let boot = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&boot, &JsValue::from_str("recordFatalPanic"), cb.as_ref());
    let _ = js_sys::Reflect::set(
        win.as_ref(),
        &JsValue::from_str("__mathExplorerBoot"),
        &boot,
    );

    let mut meta = HashMap::new();
    meta.insert("location".to_string(), "src/tabs/demo.rs:88".to_string());
    handle_diagnostic_event(&DiagnosticEvent {
        severity: Severity::Fatal,
        message: "Thread panicked: unreachable solver branch".to_string(),
        metadata: meta.clone(),
        thread_name: Some("main".to_string()),
    });
    assert!(has_fatal_panic_occurred());
    let (msg, loc) = captured.borrow().clone().expect("JS bridge invoked");
    assert!(msg.contains("unreachable solver branch") && msg.contains("src/tabs/demo.rs:88"));
    assert_eq!(loc, "src/tabs/demo.rs:88");

    let _ =
        js_sys::Reflect::delete_property(win.as_ref(), &JsValue::from_str("__mathExplorerBoot"));
    drop(cb);

    handle_diagnostic_event(&DiagnosticEvent {
        severity: Severity::Fatal,
        message: "Thread panicked: fallback DOM check".to_string(),
        metadata: meta,
        thread_name: Some("main".to_string()),
    });
    let err_el = doc.get_element_by_id("error_container").unwrap();
    assert_eq!(
        err_el.get_attribute("data-error-category").as_deref(),
        Some("runtime")
    );
    let details = doc
        .get_element_by_id("error_details")
        .and_then(|el| el.text_content())
        .unwrap_or_default();
    assert!(details.contains("fallback DOM check") && details.contains("src/tabs/demo.rs:88"));
    assert!(!doc
        .get_element_by_id("reset_workspace_button")
        .unwrap()
        .has_attribute("hidden"));
    assert!(!doc
        .get_element_by_id("download_workspace_button")
        .unwrap()
        .has_attribute("hidden"));
    reset_fatal_panic_state_for_tests();
}
