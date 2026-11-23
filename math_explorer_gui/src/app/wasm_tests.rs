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

#[cfg(all(feature = "pure_math", feature = "physics"))]
#[wasm_bindgen_test]
fn wasm_workspace_bus_links_roundtrip_across_mws_and_share_url() {
    let mut app = MathExplorerApp::default();
    app.set_layout(super::ViewportLayout::SplitHorizontal);
    if let Some(gt_idx) = app.tabs.iter().position(|t| t.url_slug() == "graph_theory") {
        app.set_pane_tab(0, gt_idx);
        let _ = app
            .pane_tab_mut(0)
            .unwrap()
            .select_tool_by_name("GraphEditorTool");
    }
    if let Some(ch_idx) = app.tabs.iter().position(|t| t.url_slug() == "chaos") {
        app.set_pane_tab(1, ch_idx);
        let _ = app
            .pane_tab_mut(1)
            .unwrap()
            .select_tool_by_name("KuramotoOscillatorTool");
    }
    app.set_pane_bus_linked(0, true);
    app.set_pane_bus_linked(1, true);

    let envelope = app.session_manager.create_envelope(app.get_app_state());
    let mut from_mws = MathExplorerApp::default();
    from_mws.load_state_from_json(&envelope);
    assert_eq!(
        from_mws.viewport_layout,
        super::ViewportLayout::SplitHorizontal
    );
    assert_eq!(from_mws.pane_bus_links, [true, true, false, false]);

    let frag = app.encode_current_share_fragment().unwrap();
    let mut from_url = MathExplorerApp::default();
    assert!(from_url
        .apply_share_location_strings(&frag, "", None)
        .unwrap());
    assert_eq!(
        from_url.viewport_layout,
        super::ViewportLayout::SplitHorizontal
    );
    assert_eq!(from_url.pane_bus_links, [true, true, false, false]);
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
