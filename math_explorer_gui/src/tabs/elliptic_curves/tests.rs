#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::EllipticCurvesTab;
use super::chord_tangent_canvas::toroidal_secant_segments;
use super::chord_tangent_tool::{CurveFieldMode, WeierstrassPlaygroundTool};
use super::modular_invariants_tool::{
    ModularInvariantsTool, ModularPlotMode, ModularQSeriesKind,
};
use crate::framework::InteractiveTool;
use crate::tabs::ExplorerTab;
use eframe::egui;

#[test]
fn test_elliptic_curves_tab_metadata_and_tool_discovery() {
    let mut tab = EllipticCurvesTab::default();
    assert_eq!(tab.name(), "Elliptic Curves");
    assert_eq!(tab.academic_domain(), "Pure Math");
    assert_eq!(tab.url_slug(), "elliptic_curves");

    let fw = tab.framework().expect("backed by SimulationFramework");
    let ids: Vec<&str> = fw.available_tools.iter().map(|t| t.id).collect();
    assert!(ids.contains(&"WeierstrassPlaygroundTool"));
    assert!(ids.contains(&"ModularInvariantsTool"));

    assert!(tab.select_tool_by_name("WeierstrassPlaygroundTool"));
    let preset = tab
        .export_url_preset_json()
        .unwrap()
        .expect("preset json exported");
    let mut restored = EllipticCurvesTab::default();
    restored
        .import_url_preset_json(Some("WeierstrassPlaygroundTool"), Some(&preset))
        .unwrap();
    assert_eq!(restored.active_tool_id(), Some("WeierstrassPlaygroundTool"));
}

#[test]
fn test_playground_state_validation_and_toroidal_secants() {
    let mut tool = WeierstrassPlaygroundTool::default();
    assert!(tool.try_load_state(r#"{"unknown": 1}"#).is_err());

    let base_json = tool.save_state().unwrap();
    for (field, bad_val) in [
        ("prime_p", serde_json::json!(3)),
        ("prime_p", serde_json::json!(15)),
        ("prime_p", serde_json::json!(1009)),
        ("scalar_k", serde_json::json!(0)),
        ("real_a", serde_json::json!(100.0)),
    ] {
        let mut bad: serde_json::Value = serde_json::from_str(&base_json).unwrap();
        bad[field] = bad_val;
        assert!(
            tool.try_load_state(&bad.to_string()).is_err(),
            "Expected validation error for {field}"
        );
    }

    let mut singular: serde_json::Value = serde_json::from_str(&base_json).unwrap();
    singular["mod_a"] = serde_json::json!(0);
    singular["mod_b"] = serde_json::json!(0);
    assert!(tool.try_load_state(&singular.to_string()).is_err());

    let oversized = " ".repeat(20 * 1024);
    assert!(tool.try_load_state(&oversized).is_err());

    let segs = toroidal_secant_segments(23, 0, 1, 7);
    assert!(segs.len() >= 2, "Expected wrapped toroidal secant segments");
    for (a, b) in &segs {
        assert!((0.0..=23.0).contains(&a[0]) && (0.0..=23.0).contains(&a[1]));
        assert!((0.0..=23.0).contains(&b[0]) && (0.0..=23.0).contains(&b[1]));
    }
}

fn focus_weierstrass_canvas(
    ctx: &egui::Context,
    tool: &mut WeierstrassPlaygroundTool,
    screen: egui::Rect,
) {
    let out1 = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let ak = out1
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("accesskit enabled");
    let canvas_node_id = ak
        .nodes
        .iter()
        .find(|(_, n)| {
            n.role() == egui::accesskit::Role::GraphicsDocument
                && n.label() == Some("Weierstrass Elliptic Curve Canvas")
        })
        .map(|(id, _)| *id)
        .expect("GraphicsDocument canvas node registered");

    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            events: vec![egui::Event::AccessKitActionRequest(
                egui::accesskit::ActionRequest {
                    action: egui::accesskit::Action::Focus,
                    target: canvas_node_id,
                    data: None,
                },
            )],
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
}

#[test]
fn test_playground_canvas_accesskit_keyboard_and_step_simulation() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    crate::accessibility::clear_test_announcements();

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = WeierstrassPlaygroundTool::default();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1100.0, 720.0));
    focus_weierstrass_canvas(&ctx, &mut tool, screen);

    let before_idx = tool.state.selected_p_idx;
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            events: vec![egui::Event::Key {
                key: egui::Key::ArrowRight,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    assert_ne!(tool.state.selected_p_idx, before_idx);
    assert!(tool.focused_canvas_id().is_some());
    assert!(
        crate::accessibility::all_announcements()
            .iter()
            .any(|(msg, _)| msg.contains("Torus F_"))
    );

    let k_before = tool.state.scalar_k;
    tool.step_simulation();
    assert_ne!(tool.state.scalar_k, k_before);
}

fn export_verification_pngs(mod_tool: &mut ModularInvariantsTool) {
    let opts = egui_plot::export::ExportOptions::default();
    mod_tool.state.view_mode = ModularPlotMode::HasseWeilTraces;
    mod_tool.state.max_prime = 251;
    mod_tool.recompute();
    if let Ok(png) = egui_plot::export::export_custom_png(&mod_tool.export_datasets(), &opts) {
        let _ = std::fs::write("/tmp/elliptic_hasse_weil_preview.png", png);
    }

    mod_tool.state.view_mode = ModularPlotMode::ModularQSeriesBars;
    mod_tool.state.q_series_kind = ModularQSeriesKind::RamanujanDelta;
    mod_tool.state.q_precision = 14;
    mod_tool.recompute();
    if let Ok(png) = egui_plot::export::export_custom_png(&mod_tool.export_datasets(), &opts) {
        let _ = std::fs::write("/tmp/elliptic_modular_qseries_preview.png", png);
    }

    let mut play_tool = WeierstrassPlaygroundTool::default();
    play_tool.state.prime_p = 23;
    play_tool.state.mod_a = 1;
    play_tool.state.mod_b = 1;
    play_tool.state.scalar_k = 8;
    play_tool.rebuild_lattice();
    if let Ok(png) = egui_plot::export::export_custom_png(&play_tool.export_datasets(), &opts) {
        let _ = std::fs::write("/tmp/elliptic_torus_orbit_preview.png", png);
    }
    play_tool.state.field_mode = CurveFieldMode::Real;
    if let Ok(png) = egui_plot::export::export_custom_png(&play_tool.export_datasets(), &opts) {
        let _ = std::fs::write("/tmp/elliptic_real_chord_preview.png", png);
    }
}

#[test]
fn test_modular_invariants_modes_validation_and_png_exports() {
    let ctx = egui::Context::default();
    let mut mod_tool = ModularInvariantsTool::default();
    let base_json = mod_tool.save_state().unwrap();

    let mut bad_deg: serde_json::Value = serde_json::from_str(&base_json).unwrap();
    bad_deg["level_n"] = serde_json::json!(5);
    bad_deg["deg_i"] = serde_json::json!(6);
    assert!(mod_tool.try_load_state(&bad_deg.to_string()).is_err());

    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 760.0));
    for mode in [
        ModularPlotMode::HasseWeilTraces,
        ModularPlotMode::HurwitzKroneckerSums,
        ModularPlotMode::BreuerValuationBounds,
        ModularPlotMode::ModularQSeriesBars,
    ] {
        mod_tool.state.view_mode = mode;
        mod_tool.recompute();
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ctx| mod_tool.show(ctx),
        );
        let ds = mod_tool.export_datasets();
        assert!(!ds.is_empty() && ds.iter().any(|(_, pts)| !pts.is_empty()));
    }
    export_verification_pngs(&mut mod_tool);
}

#[cfg(target_arch = "wasm32")]
mod wasm_suite {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_elliptic_curves_tab_url_preset_and_orbit_roundtrip() {
        let mut tab = EllipticCurvesTab::default();
        assert_eq!(tab.url_slug(), "elliptic_curves");
        assert!(tab.select_tool_by_name("WeierstrassPlaygroundTool"));
        let exported = tab.export_url_preset_json().unwrap().unwrap();

        let mut dest = EllipticCurvesTab::default();
        dest.import_url_preset_json(Some("WeierstrassPlaygroundTool"), Some(&exported))
            .unwrap();
        assert_eq!(dest.export_url_preset_json().unwrap(), Some(exported));

        let mut play = WeierstrassPlaygroundTool::default();
        play.step_simulation();
        assert!(!play.export_datasets().is_empty());

        let mod_tool = ModularInvariantsTool::default();
        assert!(!mod_tool.export_datasets().is_empty());
    }
}
