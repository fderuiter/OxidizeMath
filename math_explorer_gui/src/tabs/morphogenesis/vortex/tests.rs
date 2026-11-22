#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::MathExplorerApp;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::vortex_morphogenesis::VortexRegimePreset;

fn test_screen_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

#[test]
fn test_vortex_tool_presets_peclet_damkohler_and_kinetics_switching() {
    let mut tool = VortexMorphogenesisTool::default();
    assert_eq!(tool.name(), "Vortex Morphogenesis");
    assert_eq!(tool.domain(), "biology");

    let karman_metrics = tool.engine.metrics();
    assert!(karman_metrics.peclet > 1.0);
    assert!(karman_metrics.damkohler > 1.0);

    tool.select_regime_preset(VortexRegimePreset::ShearAlignedStripes);
    assert_eq!(tool.kinetics_kind(), VortexKineticsKind::Schnakenberg);
    assert!(tool.engine.config.shear_profile);

    tool.select_regime_preset(VortexRegimePreset::DamkohlerWashout);
    assert_eq!(tool.kinetics_kind(), VortexKineticsKind::GrayScott);
    assert!(
        tool.engine.metrics().damkohler < 1.0,
        "DamkohlerWashout preset must start in Da < 1 washout regime"
    );

    tool.set_kinetics_kind(VortexKineticsKind::Schnakenberg);
    assert_eq!(tool.kinetics_kind(), VortexKineticsKind::Schnakenberg);
    assert_eq!(tool.preset_tag, VortexPresetTag::Custom);
}

#[test]
fn test_vortex_tool_obstacle_and_emitter_brushes_mutate_coupled_masks() {
    let mut tool = VortexMorphogenesisTool::default();
    tool.engine.clear_geometry();

    tool.brush_mode = VortexBrushMode::Obstacle;
    tool.brush_radius = 2;
    tool.apply_brush_at(45, 25);
    let idx = flatten_2d_index(45, 25, tool.engine.config.width);
    assert!(tool.engine.obstacles_row()[idx]);
    assert!(tool.engine.lbm.is_obstacle(45, 25));

    tool.brush_mode = VortexBrushMode::Emitter;
    tool.apply_brush_at(60, 25);
    let emit_idx = flatten_2d_index(60, 25, tool.engine.config.width);
    assert!(tool.engine.emitters_row()[emit_idx]);
    assert!(tool.engine.morphogens.v()[emit_idx] > 0.1);

    tool.brush_mode = VortexBrushMode::Erase;
    tool.apply_brush_at(45, 25);
    assert!(!tool.engine.obstacles_row()[idx]);
    assert!(!tool.engine.lbm.is_obstacle(45, 25));
}

#[test]
fn test_vortex_tool_finite_state_guard_detects_nan_and_rolls_back_cleanly() {
    let mut tool = VortexMorphogenesisTool::default();
    tool.step_single_tick();
    assert!(FiniteStateGuard::is_finite_state(&tool));

    // Inject non-finite value into morphogen field and step
    tool.engine.morphogens.v_mut()[12] = f64::NAN;
    tool.step_single_tick();

    assert!(FiniteStateGuard::diverged(&tool));
    assert!(tool.diagnostic_report().is_some());
    assert!(tool.paused);

    // Roll back via FiniteStateGuard recovery
    tool.rollback_to_last_valid();
    assert!(!FiniteStateGuard::diverged(&tool));
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!(tool.diagnostic_report().is_none());
}

#[test]
fn test_vortex_tool_renders_accessible_canvas_and_badges() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = VortexMorphogenesisTool::default();
    tool.step_single_tick();
    tool.paused = true;

    let out = ctx.run(test_screen_input(), |ctx| {
        tool.show(ctx);
    });
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update expected");
    let has_canvas = ak.nodes.iter().any(|(_, n)| {
        n.role() == egui::accesskit::Role::Canvas
            && n.label() == Some("Coupled Vortex Morphogenesis Field")
            && n.value().is_some_and(|v| v.contains("Pe =") && v.contains("Da ="))
    });
    assert!(has_canvas, "Expected accessible Canvas node with Pe/Da value");

    let img = tool.export_raster_image().expect("raster export expected");
    assert_eq!(img.size, [100, 50]);
}

#[test]
fn test_cross_tab_navigation_bridges_fluid_dynamics_and_morphogenesis() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    crate::app::request_cross_tab_navigation(
        &ctx,
        "Morphogenesis",
        Some("VortexMorphogenesisTool"),
    );
    let _ = ctx.run(test_screen_input(), |ctx| {
        eframe::App::update(&mut app, ctx, &mut frame);
    });
    let active_tab = app.pane_tab(0).expect("primary tab");
    assert_eq!(active_tab.name(), "Morphogenesis");
    assert_eq!(
        active_tab
            .framework()
            .and_then(|fw| fw.active_tool.as_ref())
            .map(|t| t.name()),
        Some("Vortex Morphogenesis")
    );

    crate::app::request_cross_tab_navigation(
        &ctx,
        "Fluid Dynamics",
        Some("LatticeBoltzmannTool"),
    );
    let _ = ctx.run(test_screen_input(), |ctx| {
        eframe::App::update(&mut app, ctx, &mut frame);
    });
    let fluid_tab = app.pane_tab(0).expect("primary tab");
    assert_eq!(fluid_tab.name(), "Fluid Dynamics");
    assert_eq!(
        fluid_tab
            .framework()
            .and_then(|fw| fw.active_tool.as_ref())
            .map(|t| t.name()),
        Some("Lattice Boltzmann (Demo)")
    );
}

#[test]
fn test_export_all_four_vortex_regime_previews_to_png() {
    use egui_plot::export::{color32_image_to_png, ExportOptions};
    use egui_plot::export_primitives::scale_color_image;

    let opts = ExportOptions::default();
    let pid = std::process::id();
    let temp_dir = std::env::temp_dir();
    let cases = [
        (VortexRegimePreset::KarmanVortexSpots, "regime_1_karman_spots.png"),
        (VortexRegimePreset::ShearAlignedStripes, "regime_2_shear_stripes.png"),
        (VortexRegimePreset::StagnationPlumes, "regime_3_stagnation_plumes.png"),
        (VortexRegimePreset::DamkohlerWashout, "regime_4_damkohler_washout.png"),
    ];
    let mut tool = VortexMorphogenesisTool::default();
    for (preset, filename) in cases {
        tool.select_regime_preset(preset);
        tool.display_mode = if preset == VortexRegimePreset::ShearAlignedStripes {
            VortexDisplayMode::MorphogenU
        } else {
            VortexDisplayMode::CoupledWake
        };
        for _ in 0..3 {
            tool.step_single_tick();
        }
        assert!(FiniteStateGuard::is_finite_state(&tool));
        let raw = tool.export_raster_image().expect("raster image");
        let scaled = scale_color_image(&raw, 600, 300, &opts);
        assert_eq!(scaled.size, [600, 300]);
        let png = color32_image_to_png(&scaled).expect("valid png");
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        let out_path = temp_dir.join(format!("oxidize_{pid}_{filename}"));
        std::fs::write(&out_path, &png).expect("write preview png");
        let written_len = std::fs::metadata(&out_path).expect("png metadata").len();
        assert_eq!(written_len, png.len() as u64);
        let _ = std::fs::remove_file(&out_path);
    }
}
