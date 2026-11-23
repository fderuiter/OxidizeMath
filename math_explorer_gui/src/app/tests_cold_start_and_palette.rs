#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{AcademicDomain, MathExplorerApp, ThemeMode, ViewportLayout};
use crate::command_palette::{PaletteAction, PaletteSelection};
use eframe::egui;

#[test]
fn test_curated_cold_start_tab_and_flagship_tool_ordering() {
    let app = MathExplorerApp::default();
    assert!(app.tabs.len() >= 21);
    assert_eq!(app.tabs[0].name(), "Quantum Mechanics");
    assert_eq!(app.tabs[0].active_tool_id(), Some("WaveSimulator"));
    assert_eq!(app.tabs[1].name(), "MRI Physics");
    assert_eq!(app.tabs[2].name(), "Fluid Dynamics");
    assert_eq!(app.tabs[3].name(), "Chaos Theory");
    assert_eq!(app.tabs[4].name(), "Solid State Physics");
    assert_eq!(app.tabs[5].name(), "Medical Physics");
    assert_eq!(
        app.tabs.last().map(|t| t.name()),
        Some("Artificial Intelligence")
    );

    let domain_ranks: Vec<usize> = app
        .tabs
        .iter()
        .map(|t| {
            let dom = AcademicDomain::from_str(t.academic_domain()).unwrap();
            AcademicDomain::ALL.iter().position(|&d| d == dom).unwrap()
        })
        .collect();
    let mut sorted = domain_ranks.clone();
    sorted.sort_unstable();
    assert_eq!(
        domain_ranks, sorted,
        "Cold-start tabs must follow Physics -> Pure Math -> Applied Math -> Biology -> Climate -> AI"
    );
}

#[test]
fn test_legacy_and_named_session_restoration_across_single_split_and_quad() {
    let mut legacy_single = MathExplorerApp::default();
    let legacy_env = r#"{
        "selected_tab": 0,
        "selected_tab_name": null,
        "show_info": true,
        "show_warnings": true,
        "show_errors": true,
        "tab_states": { "Artificial Intelligence": "{}" }
    }"#;
    legacy_single.load_state_from_json(legacy_env);
    assert_eq!(
        legacy_single.tabs[legacy_single.selected_tab].name(),
        "Artificial Intelligence"
    );

    let mut split_src = MathExplorerApp::default();
    let chaos_idx = split_src
        .tabs
        .iter()
        .position(|t| t.name() == "Chaos Theory")
        .unwrap();
    let num_idx = split_src
        .tabs
        .iter()
        .position(|t| t.name() == "Number Theory")
        .unwrap();
    split_src.set_layout(ViewportLayout::SplitHorizontal);
    split_src.set_pane_tab(0, chaos_idx);
    split_src.set_pane_tab(1, num_idx);
    let mut saved_split = split_src.get_app_state();
    saved_split.selected_tab = 0;
    saved_split.secondary_tab = 1;
    saved_split.pane_tabs = vec![0, 1, 2, 3];
    let split_json = split_src.session_manager.create_envelope(saved_split);

    let mut split_dst = MathExplorerApp::default();
    split_dst.load_state_from_json(&split_json);
    assert_eq!(split_dst.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(
        split_dst.tabs[split_dst.selected_tab].name(),
        "Chaos Theory"
    );
    assert_eq!(
        split_dst.tabs[split_dst.secondary_tab].name(),
        "Number Theory"
    );
}

#[test]
fn test_quad_grid_session_restoration_by_tab_name() {
    let mut quad_src = MathExplorerApp::default();
    quad_src.set_layout(ViewportLayout::QuadGrid);
    let target_names = [
        "MRI Physics",
        "Graph Theory",
        "Climate",
        "Artificial Intelligence",
    ];
    for (p, name) in target_names.iter().enumerate() {
        let idx = quad_src
            .tabs
            .iter()
            .position(|t| t.name() == *name)
            .unwrap();
        quad_src.set_pane_tab(p, idx);
    }
    let mut saved_quad = quad_src.get_app_state();
    saved_quad.pane_tabs = vec![0, 1, 2, 3];
    let quad_json = quad_src.session_manager.create_envelope(saved_quad);

    let mut quad_dst = MathExplorerApp::default();
    quad_dst.load_state_from_json(&quad_json);
    assert_eq!(quad_dst.viewport_layout, ViewportLayout::QuadGrid);
    for (p, expected) in target_names.iter().enumerate() {
        assert_eq!(quad_dst.tabs[quad_dst.pane_tabs[p]].name(), *expected);
    }
}

#[test]
fn test_palette_executes_actions_and_presets_directly() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    app.execute_palette_selection(
        &ctx,
        PaletteSelection::command(PaletteAction::SetThemeHighContrast),
    );
    assert_eq!(app.theme, ThemeMode::HighContrast);

    app.execute_palette_selection(
        &ctx,
        PaletteSelection::command(PaletteAction::SetLayoutSplitHorizontal),
    );
    assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);

    app.execute_palette_selection(
        &ctx,
        PaletteSelection::command(PaletteAction::OpenStarterGallery),
    );
    assert!(app.show_starter_gallery);
    app.show_starter_gallery = false;

    app.execute_palette_selection(
        &ctx,
        PaletteSelection::command(PaletteAction::ToggleLinkPlotAxes),
    );
    assert!(app.compare_sync.link_axes);

    let preset = inventory::iter::<crate::starter_gallery::StarterPresetMetadata>
        .into_iter()
        .find(|p| p.title == "Logistic Growth")
        .expect("Logistic Growth preset registered");
    let analysis_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Analysis & Calculus")
        .unwrap();
    app.focus_pane(1);
    app.execute_palette_selection(&ctx, PaletteSelection::preset(analysis_idx, preset));
    assert_eq!(app.tabs[app.pane_tabs[1]].name(), "Analysis & Calculus");
    let pane1_state = app.pane_tab(1).and_then(|t| t.save_state()).unwrap();
    assert!(pane1_state.contains("LogisticGrowth"));
}
