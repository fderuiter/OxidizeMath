use super::{MathExplorerApp, ThemeMode, ViewportLayout};
use crate::url_state::{
    install_browser_history_listeners, replace_browser_url_fragment, reset_history_bridge_for_test,
};
use eframe::App as _;
use std::collections::HashMap;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[derive(Default)]
struct WasmMemoryStorage {
    entries: HashMap<String, String>,
}

impl eframe::Storage for WasmMemoryStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.entries.get(key).cloned()
    }

    fn set_string(&mut self, key: &str, value: String) {
        self.entries.insert(key.to_string(), value);
    }

    fn flush(&mut self) {}
}

#[wasm_bindgen_test]
fn wasm_app_as_any_mut_downcasts_and_roundtrips_recovery_snapshot() {
    let mut boxed: Box<dyn eframe::App> = Box::new(MathExplorerApp::default());
    let app = boxed
        .as_any_mut()
        .expect("MathExplorerApp::as_any_mut must return Some on wasm32")
        .downcast_mut::<MathExplorerApp>()
        .expect("dyn Any must downcast to MathExplorerApp");

    app.theme = ThemeMode::Light;
    let snapshot = app.capture_recovery_snapshot();
    assert!(snapshot
        .envelope_json
        .contains("\"schema_version\": \"1.0\""));

    let mut restored = MathExplorerApp::default();
    let ctx = eframe::egui::Context::default();
    restored.restore_recovery_snapshot(&snapshot, &ctx);
    assert_eq!(restored.theme, ThemeMode::Light);
}

#[wasm_bindgen_test]
fn wasm_self_synced_reload_preserves_multi_pane_and_undo_while_external_url_pushes_undo() {
    let window = web_sys::window().expect("window");
    let history = window.history().expect("history");
    let initial_href = window.location().href().unwrap_or_default();
    reset_history_bridge_for_test();

    let mut source = MathExplorerApp::default();
    let prior_single = source.get_app_state();
    source.session_manager.push_undo(prior_single);
    source.set_layout(ViewportLayout::QuadGrid);
    source.set_pane_tab(0, 1);
    source.set_pane_tab(1, 2);
    source.set_pane_tab(2, 3);
    source.set_pane_tab(3, 4);
    source.focused_pane = 3;
    source.quad_split_ratios = (0.36, 0.68);

    let ws_frag = source.encode_current_share_fragment().unwrap();
    replace_browser_url_fragment(&ws_frag).unwrap();
    // Simulate a fast pre-debounce focus change persisted to localStorage before reload
    source.focused_pane = 2;

    let mut storage = WasmMemoryStorage::default();
    source.save(&mut storage);

    // 1. Self-synced reload (`History.state` matches URL) keeps localStorage (focused_pane == 2) + undo
    let ctx = eframe::egui::Context::default();
    install_browser_history_listeners(&ctx);
    let mut cc = eframe::CreationContext::_new_kittest(ctx.clone());
    cc.storage = Some(&storage);
    let reloaded = MathExplorerApp::new(&cc);
    assert_eq!(reloaded.viewport_layout, ViewportLayout::QuadGrid);
    assert_eq!(reloaded.focused_pane, 2);
    assert_eq!(reloaded.session_manager.undo_depth(), 1);

    // 2. External navigation (`History.state == null`) overrides and records undo
    history
        .replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some("#tab=chaos"))
        .unwrap();
    let mut cc_ext = eframe::CreationContext::_new_kittest(ctx.clone());
    cc_ext.storage = Some(&storage);
    let mut overridden = MathExplorerApp::new(&cc_ext);
    assert_eq!(overridden.viewport_layout, ViewportLayout::Single);
    assert_eq!(overridden.tabs[overridden.selected_tab].url_slug(), "chaos");
    assert_eq!(overridden.session_manager.undo_depth(), 2);

    overridden.undo();
    assert_eq!(overridden.viewport_layout, ViewportLayout::QuadGrid);
    assert_eq!(overridden.focused_pane, 2);

    // 3. Live DOM `hashchange` + `popstate` events update workspace and ignore `#overview`
    history
        .replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some("#overview"))
        .unwrap();
    let _ = window.dispatch_event(&web_sys::Event::new("hashchange").unwrap());
    overridden.poll_share_shortcut_and_clipboard(&ctx);
    assert_eq!(overridden.viewport_layout, ViewportLayout::QuadGrid);
    assert_eq!(overridden.focused_pane, 2);

    history
        .replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some("#tab=mri"))
        .unwrap();
    let _ = window.dispatch_event(&web_sys::Event::new("popstate").unwrap());
    overridden.poll_share_shortcut_and_clipboard(&ctx);
    assert_eq!(overridden.viewport_layout, ViewportLayout::Single);
    assert_eq!(overridden.tabs[overridden.selected_tab].url_slug(), "mri");

    if !initial_href.is_empty() {
        let _ =
            history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&initial_href));
    }
}
