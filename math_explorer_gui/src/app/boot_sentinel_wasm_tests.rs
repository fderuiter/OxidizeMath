//! `wasm32` browser tests verifying `sessionStorage` boot-sentinel quarantine,
//! `localStorage` backup archival, theme preservation, and `History.replaceState` URL stripping.

use super::{
    complete_first_frame, BOOT_SENTINEL_SESSION_KEY, PRESERVED_THEME_STORAGE_KEY,
    QUARANTINED_WORKSPACE_STORAGE_KEY,
};
use crate::app::{MathExplorerApp, ThemeMode};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_boot_sentinel_quarantines_local_storage_and_clears_share_hash() {
    let win = web_sys::window().expect("window");
    let ss = win.session_storage().unwrap().unwrap();
    let ls = win.local_storage().unwrap().unwrap();
    let _ = ss.set_item(BOOT_SENTINEL_SESSION_KEY, "hydrating");
    let raw_saved = r#"{"schema_version":"1.0","app_version":"0.1.0","timestamp":"t","app_state":{"theme":"Light"}}"#;
    let _ = ls.set_item("app", raw_saved);
    if let Ok(hist) = win.history() {
        let _ = hist.replace_state_with_url(&JsValue::NULL, "", Some("/?keep=1&tab=bad#ws=crash"));
    }

    let cc = eframe::CreationContext::_new_kittest(eframe::egui::Context::default());
    let mut app = MathExplorerApp::new(&cc);

    assert_eq!(app.theme, ThemeMode::Light);
    assert!(app.boot_sentinel.safe_mode_banner_open);
    assert!(ls.get_item("app").unwrap().is_none());
    assert_eq!(
        ls.get_item(QUARANTINED_WORKSPACE_STORAGE_KEY)
            .unwrap()
            .as_deref(),
        Some(raw_saved)
    );
    assert_eq!(
        ls.get_item(PRESERVED_THEME_STORAGE_KEY).unwrap().as_deref(),
        Some("light")
    );
    assert_eq!(win.location().hash().unwrap_or_default(), "");
    assert_eq!(win.location().search().unwrap_or_default(), "?keep=1");

    complete_first_frame(&mut app);
    assert!(ss.get_item(BOOT_SENTINEL_SESSION_KEY).unwrap().is_none());

    let _ = ls.remove_item(QUARANTINED_WORKSPACE_STORAGE_KEY);
    let _ = ls.remove_item(PRESERVED_THEME_STORAGE_KEY);
    if let Some(root) = win.document().and_then(|d| d.document_element()) {
        let _ = root.remove_attribute("data-theme");
    }
    if let Ok(hist) = win.history() {
        let _ = hist.replace_state_with_url(&JsValue::NULL, "", Some("/"));
    }
}
