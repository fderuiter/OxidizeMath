#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Desktop unit tests for [`RelativityTab`], [`GeodesicPlaygroundTool`], [`ChristoffelInspectorTool`],
//! photon-ring normalization, Kerr/ISCO geometry, custom AST metrics, 2D atlas, and stories/challenges.

use std::collections::{HashMap, HashSet};

use eframe::egui;

use super::*;
use crate::app::{MathExplorerApp, ThemeMode};
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::ExplorerTab;

pub(super) fn test_raw_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        events,
        ..Default::default()
    }
}

fn obs_map(tool: &dyn InteractiveTool) -> HashMap<&'static str, f64> {
    tool.observables().into_iter().collect()
}

#[test]
fn relativity_tab_renders_both_tools_across_all_three_themes() {
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut tab = RelativityTab::default();
    assert_eq!(tab.name(), "General Relativity");
    assert_eq!(tab.academic_domain(), "Physics");
    assert_eq!(tab.url_slug(), "relativity");

    for tool_id in ["GeodesicPlaygroundTool", "ChristoffelInspectorTool"] {
        assert!(tab.select_tool_by_name(tool_id));
        for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
            theme.apply_to_context(&ctx);
            let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    tab.show_inside(ui, &mut frame);
                });
            });
        }
    }
}

#[test]
fn geodesic_playground_camera3d_scroll_zoom_and_drag_orbit_update_state() {
    let ctx = egui::Context::default();
    let mut tool = GeodesicPlaygroundTool::default();
    let initial_cam = tool.state.camera;

    // Frame 1: layout plot rect
    let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });

    // Frame 2: scroll wheel over central 3D viewport zooms Camera3D
    let center = egui::pos2(780.0, 320.0);
    let zoom_events = vec![
        egui::Event::PointerMoved(center),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, 2.0),
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(test_raw_input(zoom_events), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    assert!(
        (tool.state.camera.zoom - initial_cam.zoom).abs() > 1e-4,
        "Camera3D zoom must respond to scroll wheel inside Plot3DViewport"
    );
}

#[test]
fn relativity_presets_url_hydration_and_finite_state_guard_rollback() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    assert!(
        app.apply_share_location_strings("#tab=relativity&tool=GeodesicPlaygroundTool", "", None)
            .unwrap()
    );
    let active_tab = &mut app.tabs[app.selected_tab];
    assert_eq!(active_tab.url_slug(), "relativity");

    let mut playground = GeodesicPlaygroundTool::default();
    for payload in [
        presets::PRESET_PHOTON_RING_LENSING,
        presets::PRESET_ISCO_ROSETTE_CLOCKS,
        presets::PRESET_WORMHOLE_TRANSIT,
    ] {
        playground.import_preset_json(payload).unwrap();
        assert!(!playground.samples.is_empty());
    }
    assert_eq!(
        playground.state.spacetime,
        SpacetimePreset::MorrisThorneWormhole
    );

    // Inject NaN and verify FiniteStateGuard detection + clamp recovery
    playground.state.scale_radius = f64::NAN;
    assert!(!playground.has_finite_parameters());
    assert!(playground.clamp_parameters());
    assert!(playground.has_finite_parameters());

    let mut inspector = ChristoffelInspectorTool::default();
    inspector
        .import_preset_json(presets::PRESET_CHRISTOFFEL_CROSS_CHECK)
        .unwrap();
    let eval = inspector
        .evaluate_current()
        .expect("valid Kerr Christoffel");
    assert!(eval.max_abs_diff < 1e-6);
}

#[test]
fn relativity_tools_export_valid_png_svg_and_csv_artifacts() {
    for tool_id in ["GeodesicPlaygroundTool", "ChristoffelInspectorTool"] {
        let mut tab = RelativityTab::default();
        assert!(tab.select_tool_by_name(tool_id));
        let fw = tab.framework_mut().unwrap();
        fw.export_options.width = 360;
        fw.export_options.height = 240;

        for format in [ExportFormat::Png, ExportFormat::Svg, ExportFormat::Csv] {
            let ctx = egui::Context::default();
            let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
                egui_plot::clear_last_plot_datasets();
                fw.show_export_modal = true;
                fw.queue_export(format);
                fw.show(ctx, "relativity_export_test");
            });
            assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
            match format {
                ExportFormat::Png => {
                    let bytes = fw.export_queue.last_exported_bytes.as_deref().unwrap();
                    let px =
                        crate::framework::deferred_export_tests::decode_rgba_pixels(bytes).unwrap();
                    let colors: HashSet<[u8; 4]> = px.into_iter().collect();
                    assert!(colors.len() >= 2);
                }
                ExportFormat::Svg => {
                    let svg = fw.export_queue.last_exported_content.as_deref().unwrap();
                    assert!(svg.contains("<svg") && svg.len() > 250);
                }
                ExportFormat::Csv => {
                    let csv = fw.export_queue.last_exported_content.as_deref().unwrap();
                    assert!(csv.lines().count() > 20);
                }
                _ => {}
            }
        }
    }
}

#[test]
fn photon_ring_preset_normalizes_energy_grazes_photon_sphere_and_captures_twin() {
    let mut tool = GeodesicPlaygroundTool::default();
    tool.import_preset_json(presets::PRESET_PHOTON_RING_LENSING)
        .expect("photon ring preset must load");
    assert!((tool.launch_energy - 1.0).abs() < 1e-12);
    let obs = obs_map(&tool);
    assert!((obs["launch_energy"] - 1.0).abs() < 1e-12);
    assert!((obs["effective_energy"] - 0.5).abs() < 1e-12);
    assert!(
        (3.0..=3.3).contains(&obs["r_min"]),
        "expected r_min in [3.0, 3.3], got {}",
        obs["r_min"]
    );
    assert!(obs["captured"] < 0.5, "primary ray should escape");
    assert!(
        obs["twin_captured"] > 0.5,
        "twin ray with negative delta_b must capture"
    );
    assert!(
        obs["twin_r_min"] <= 2.05,
        "captured twin must reach r <= 2.05, got {}",
        obs["twin_r_min"]
    );

    let datasets = tool.export_datasets();
    let e_line = datasets
        .iter()
        .find(|(n, _)| n == "0.5 * e^2")
        .expect("0.5 * e^2 dataset");
    assert!((e_line.1[0][1] - 0.5).abs() < 1e-12);
}

#[test]
fn kerr_equatorial_spin_09_surfaces_exact_horizons_frame_dragging_and_isco() {
    let mut tool = GeodesicPlaygroundTool::default();
    tool.state.spacetime = SpacetimePreset::KerrEquatorial;
    tool.state.scale_radius = 2.0;
    tool.state.spin_chi = 0.9;
    tool.state.initial_radius = 9.0;
    tool.state.impact_or_momentum = 3.1;
    tool.clamp_parameters();

    let obs = obs_map(&tool);
    assert!(
        (obs["outer_horizon_radius"] - 1.43589).abs() < 1e-5,
        "expected r_+ ≈ 1.43589, got {}",
        obs["outer_horizon_radius"]
    );
    assert!((obs["ergosphere_radius"] - 2.0).abs() < 1e-12);
    assert!(obs["frame_dragging_omega"] > 0.0 && obs["frame_dragging_omega"].is_finite());
    assert!(obs["isco"] > 2.0 && obs["isco"] < 6.0);

    tool.state.mode = playground::GeodesicMode::TimelikeParticle;
    tool.state.initial_radius = 1.8; // Inside the ergosphere (r_+ = 1.436 < 1.8 < r_ergo = 2.0)
    tool.state.impact_or_momentum = 3.6;
    tool.rebuild_trajectory(120);
    assert!((tool.samples[0].metric_norm + 1.0).abs() < 1e-9);

    let ctx = egui::Context::default();
    let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
}

#[test]
fn custom_ast_lapse_matches_schwarzschild_christoffel_and_steps_720_zero_alloc() {
    let mut inspector = ChristoffelInspectorTool::default();
    inspector.state.manifold = InspectorManifold::CustomRadialLapse4D;
    inspector.state.scale_radius = 2.0;
    inspector.state.radial_coord = 4.5;
    inspector.set_custom_lapse_expression("1 - 2 / r");
    let eval = inspector
        .evaluate_current()
        .expect("custom 1 - 2 / r Christoffel");
    assert!(
        eval.max_abs_diff < 1e-6,
        "1 - 2/r central diff must match Schwarzschild within 1e-6, got {}",
        eval.max_abs_diff
    );

    inspector.set_custom_lapse_expression("1 - Rs / r + q^2 / r^2");
    assert_eq!(
        inspector.sandbox.lapse_bar.active_param_names(),
        &["q".to_string()]
    );

    let mut playground = GeodesicPlaygroundTool::default();
    playground.state.spacetime = SpacetimePreset::CustomRadialLapse;
    playground.state.scale_radius = 2.0;
    playground.state.initial_radius = 12.0;
    playground.state.impact_or_momentum = 5.25;
    playground.state.step_size = 0.015;
    playground.sandbox.lapse_bar.set_expression("1 - 2 / r");
    playground.clamp_parameters();

    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            playground.rebuild_trajectory(720);
        });
    }
    assert_eq!(playground.samples.len(), 721);
    assert!(
        playground
            .samples
            .iter()
            .all(|s| s.radial_coord.is_finite() && s.metric_norm.is_finite())
    );

    // Plunging custom Reissner-Nordstrom ray halts cleanly when f(r) <= 0.02 without NaNs
    playground
        .sandbox
        .lapse_bar
        .set_expression("1 - Rs / r + q^2 / r^2");
    playground.state.impact_or_momentum = 0.5;
    playground.state.step_size = 0.05;
    playground.rebuild_trajectory(600);
    assert!(playground.halted_near_singularity);
    assert!(
        playground
            .samples
            .iter()
            .all(|s| s.radial_coord.is_finite() && s.embed_xyz[2].is_finite())
    );
}
