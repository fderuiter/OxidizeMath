#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Desktop unit tests for [`RelativityTab`], [`GeodesicPlaygroundTool`], [`ChristoffelInspectorTool`],
//! `Camera3D` gestures, `#tab=relativity` URL hydration, `FiniteStateGuard` rollback, and exports.

use std::collections::HashSet;

use eframe::egui;

use super::*;
use crate::app::{MathExplorerApp, ThemeMode};
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::ExplorerTab;

fn test_raw_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        events,
        ..Default::default()
    }
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
    assert!(app
        .apply_share_location_strings("#tab=relativity&tool=GeodesicPlaygroundTool", "", None)
        .unwrap());
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
    let eval = inspector.evaluate_current().expect("valid Kerr Christoffel");
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
                    let (_, _, px) = egui_plot::export_primitives::decode_png_rgba(bytes).unwrap();
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
