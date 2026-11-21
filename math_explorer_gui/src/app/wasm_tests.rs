use super::{MathExplorerApp, ThemeMode};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

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
