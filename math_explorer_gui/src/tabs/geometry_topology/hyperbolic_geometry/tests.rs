#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{PRESET_CHAOTIC_BILLIARDS_JSON, PRESET_ESCHER_TILING_JSON};
use super::*;
use crate::tabs::ExplorerTab;
use crate::tabs::geometry_topology::GeometryTopologyTab;
use crate::url_state::parse_share_location;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_save_load_state_and_deny_unknown_fields() {
    let mut tool = HyperbolicGeometryTool::default();
    tool.state.mode = LabMode::ChaoticBilliards;
    tool.state.billiard_preset = BilliardPreset::IdealTriangle;
    tool.state.billiard_seed = 777;
    tool.apply_disk_drag(DiskPoint::origin(), DiskPoint::new_clamped(0.22, -0.14));

    let saved = tool.save_state().expect("save_state must succeed");
    let mut restored = HyperbolicGeometryTool::default();
    restored
        .try_load_state(&saved)
        .expect("try_load_state must restore valid snapshot");
    assert_eq!(restored.state.mode, LabMode::ChaoticBilliards);
    assert_eq!(restored.state.billiard_preset, BilliardPreset::IdealTriangle);
    assert_eq!(restored.state.billiard_seed, 777);

    let unknown_json = r#"{"mode":"TilingExplorer","unknown_extra_key":123}"#;
    assert!(restored.try_load_state(unknown_json).is_err());

    let non_hyperbolic_json = saved
        .replace("\"Heptagonal73\"", "\"CustomSchlafli\"")
        .replace("\"schlafli_p\":7", "\"schlafli_p\":4")
        .replace("\"schlafli_q\":3", "\"schlafli_q\":4");
    assert!(restored.try_load_state(&non_hyperbolic_json).is_err());
}

#[test]
fn test_finite_state_guard_clamp_parameters_recovers_corrupted_values() {
    let mut tool = HyperbolicGeometryTool::default();
    tool.state.horocycle_radius = f64::NAN;
    tool.state.half_plane_span = f64::INFINITY;
    assert!(!tool.has_finite_parameters());
    assert!(tool.clamp_parameters());
    assert!(tool.has_finite_parameters());
}

#[test]
fn test_url_share_link_roundtrip_on_geometry_topology_tab() {
    let mut tab = GeometryTopologyTab::default();
    assert!(tab.select_tool_by_name("Hyperbolic Geometry"));
    {
        let fw = tab.framework_mut().expect("framework");
        let tool = fw.active_tool.as_mut().expect("active Hyperbolic Geometry");
        tool.import_preset_json(PRESET_ESCHER_TILING_JSON)
            .expect("starter preset loads");
    }
    let expected_state = tab.save_state().expect("tab save_state");
    let share_url = tab
        .framework()
        .expect("framework")
        .encode_share_url()
        .expect("encode_share_url must succeed");
    assert!(share_url.contains("#tab=geometry_topology&tool=HyperbolicGeometryTool&state="));

    let hash_idx = share_url.find('#').expect("URL hash fragment");
    let parsed = parse_share_location(&share_url[hash_idx..], "")
        .expect("parse_share_location ok")
        .expect("Some(ShareUrlParams)");
    assert_eq!(parsed.tab, "geometry_topology");
    assert_eq!(parsed.tool.as_deref(), Some("HyperbolicGeometryTool"));

    let mut restored_tab = GeometryTopologyTab::default();
    restored_tab
        .framework_mut()
        .expect("restored framework")
        .import_url_preset_json(parsed.tool.as_deref(), parsed.state_json.as_deref())
        .expect("import_url_preset_json must succeed");
    assert_eq!(
        restored_tab.save_state().as_deref(),
        Some(expected_state.as_str())
    );
}

#[test]
fn test_svg_png_and_csv_dataset_exports_across_modes() {
    let mut tool = HyperbolicGeometryTool::default();
    let tiling_ds = tool.export_datasets();
    assert!(tiling_ds.len() >= 3);
    let tiling_prims = tool.export_plot_primitives();
    let opts = egui_plot::export::ExportOptions::default();
    let svg = egui_plot::export_primitives::export_primitives_svg(&tiling_prims, &opts)
        .expect("tiling SVG export");
    assert!(svg.contains("<svg"));

    tool.import_preset_json(PRESET_CHAOTIC_BILLIARDS_JSON)
        .expect("load chaotic billiards preset");
    tool.step_simulation();
    let billiard_ds = tool.export_datasets();
    assert!(
        billiard_ds
            .iter()
            .any(|(name, pts)| name.contains("Log Separation") && !pts.is_empty())
    );
    let billiard_prims = tool.export_plot_primitives();
    let billiard_svg = egui_plot::export_primitives::export_primitives_svg(&billiard_prims, &opts)
        .expect("billiard SVG export");
    assert!(billiard_svg.contains("<svg"));
}

fn click_tool_at(
    ctx: &egui::Context,
    tool: &mut HyperbolicGeometryTool,
    pos: egui::Pos2,
    base: &egui::RawInput,
) {
    let _ = ctx.run(
        egui::RawInput {
            events: vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..base.clone()
        },
        |c| tool.show(c),
    );
}

#[test]
fn test_accesskit_canvas_focus_ring_and_arrow_key_panning_without_tab_trap() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = HyperbolicGeometryTool::default();

    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let out = ctx.run(raw.clone(), |c| tool.show(c));
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update");
    let canvas_bounds = ak
        .nodes
        .iter()
        .find(|(_, node)| {
            node.role() == egui::accesskit::Role::Canvas
                && node.label().unwrap_or("").contains("Poincaré Disk")
        })
        .and_then(|(_, node)| node.bounds())
        .expect("Poincare Disk Canvas must expose AccessKit node");

    let click_pos = egui::pos2(
        ((canvas_bounds.x0 + canvas_bounds.x1) * 0.5) as f32,
        ((canvas_bounds.y0 + canvas_bounds.y1) * 0.5) as f32,
    );
    click_tool_at(&ctx, &mut tool, click_pos, &raw);
    assert!(tool.focused_canvas_id().is_some());

    let before_b = tool.state.disk_isometry.b();
    let _ = ctx.run(
        egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::ArrowRight,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..raw
        },
        |c| tool.show(c),
    );
    assert_ne!(tool.state.disk_isometry.b(), before_b);
}

#[test]
fn test_zero_alloc_hot_loop_under_verified_allocator_in_gui_tool() {
    let mut tool = HyperbolicGeometryTool::default();
    tool.state.mode = LabMode::ChaoticBilliards;
    tool.state.billiards_running = true;

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    VerifiedEngine::run_verified(|| {
        for i in 0..100usize {
            let t = (i as f64) * 0.03;
            tool.apply_disk_drag(
                DiskPoint::new_clamped(0.1 * t.cos(), 0.1 * t.sin()),
                DiskPoint::new_clamped(0.12 * (t + 0.05).cos(), 0.12 * (t + 0.05).sin()),
            );
            tool.step_simulation();
        }
    });
}
