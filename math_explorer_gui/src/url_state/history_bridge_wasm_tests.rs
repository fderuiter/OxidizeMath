use super::{
    drain_pending_browser_navigations, install_browser_history_listeners,
    reset_history_bridge_for_test,
};
use crate::url_state::replace_browser_url_fragment;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_window_hashchange_and_popstate_queue_external_navigations_and_skip_self_sync_echoes() {
    let window = web_sys::window().expect("browser window must exist");
    let history = window.history().expect("browser history must exist");
    let initial_href = window.location().href().unwrap_or_default();

    reset_history_bridge_for_test();
    let ctx = eframe::egui::Context::default();
    install_browser_history_listeners(&ctx);

    // 1. Self-synced replaceState echo must be ignored even if hashchange/popstate fire
    replace_browser_url_fragment("#tab=chaos&tool=AttractorPlotter")
        .expect("self-synced replaceState should succeed");
    let hash_ev = web_sys::Event::new("hashchange").expect("hashchange event");
    window
        .dispatch_event(&hash_ev)
        .expect("dispatch hashchange");
    let pop_ev = web_sys::Event::new("popstate").expect("popstate event");
    window.dispatch_event(&pop_ev).expect("dispatch popstate");
    assert!(
        drain_pending_browser_navigations().is_empty(),
        "Self-synced replaceState echoes must not enqueue external navigations"
    );

    // 2. External navigation (paired popstate + hashchange per WHATWG) enqueues once
    history
        .replace_state_with_url(
            &wasm_bindgen::JsValue::NULL,
            "",
            Some("/index.html#tab=mri"),
        )
        .expect("simulate external URL navigation");
    window
        .dispatch_event(&web_sys::Event::new("popstate").unwrap())
        .unwrap();
    window
        .dispatch_event(&web_sys::Event::new("hashchange").unwrap())
        .unwrap();
    let drained = drain_pending_browser_navigations();
    assert_eq!(drained, vec![("#tab=mri".to_string(), String::new())]);

    // 3. Document anchor navigation (`#overview`) is queued cleanly without panicking
    history
        .replace_state_with_url(
            &wasm_bindgen::JsValue::NULL,
            "",
            Some("/index.html#overview"),
        )
        .expect("simulate document anchor navigation");
    window
        .dispatch_event(&web_sys::Event::new("hashchange").unwrap())
        .unwrap();
    assert_eq!(
        drain_pending_browser_navigations(),
        vec![("#overview".to_string(), String::new())]
    );

    reset_history_bridge_for_test();
    if !initial_href.is_empty() {
        let _ =
            history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&initial_href));
    }
}
