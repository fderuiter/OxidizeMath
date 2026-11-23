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

#[cfg(feature = "applied")]
#[wasm_bindgen_test]
fn wasm_cannibalism_tab_url_hydration_stepping_and_lexis_image() {
    let mut app = MathExplorerApp::default();
    let hydrated = app
        .apply_share_location_strings("#tab=cannibalism", "", None)
        .expect("valid #tab=cannibalism share location");
    assert!(hydrated);
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "cannibalism");
    assert_eq!(app.tabs[app.selected_tab].academic_domain(), "Applied Math");

    let mut tool = crate::tabs::cannibalism::PopulationStructureTool::default();
    let t_before = tool.mvf_solver.time();
    crate::framework::InteractiveTool::step_simulation(&mut tool);
    assert!(tool.mvf_solver.time() > t_before);

    let img = tool.compute_lexis_image(true);
    assert!(img.width() > 1 && img.height() >= 16);
}
