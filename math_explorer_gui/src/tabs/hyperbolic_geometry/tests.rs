#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::HyperbolicGeometryTab;
use super::billiards_tool::HyperbolicBilliardsTool;
use super::disk_tool::HyperbolicDiskTool;
use crate::framework::{
    ExportFormat, ExportStatusSeverity, InteractiveTool, SimulationFramework,
};
use crate::tabs::ExplorerTab;
use eframe::egui;
use math_explorer::num_complex::Complex64;
use math_explorer::pure_math::differential_geometry::hyperbolic::{
    HyperbolicModel, POINCARE_DISK_RADIUS_MAX, gauss_bonnet_triangle,
};

fn viewport_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 820.0),
        )),
        events,
        ..Default::default()
    }
}

#[test]
fn test_hyperbolic_tab_domain_slug_and_registered_tools() {
    let mut tab = HyperbolicGeometryTab::default();
    assert_eq!(tab.name(), "Hyperbolic Geometry");
    assert_eq!(tab.academic_domain(), "Pure Math");
    assert_eq!(tab.url_slug(), "hyperbolic_geometry");
    assert!(tab.select_tool_by_name("HyperbolicDiskTool"));
    assert_eq!(tab.active_tool_id(), Some("HyperbolicDiskTool"));
    assert!(tab.select_tool_by_name("HyperbolicBilliardsTool"));
    assert_eq!(tab.active_tool_id(), Some("HyperbolicBilliardsTool"));
}

#[test]
fn test_disk_tool_pointer_drag_mobius_panning_preserves_gauss_bonnet_invariants() {
    let ctx = egui::Context::default();
    let mut tool = HyperbolicDiskTool::default();
    let gb_before = gauss_bonnet_triangle(
        tool.ruler_vertices[0],
        tool.ruler_vertices[1],
        tool.ruler_vertices[2],
    );
    let rev_before = tool.state_revision();

    // Frame 1: lay out side panel and disk canvas.
    let _ = ctx.run(viewport_input(Vec::new()), |ctx| tool.show(ctx));
    let start = egui::pos2(820.0, 410.0);
    let end = egui::pos2(890.0, 360.0);

    // Frame 2-4: press, drag while held, then release across the central disk canvas.
    let _ = ctx.run(
        viewport_input(vec![
            egui::Event::PointerMoved(start),
            egui::Event::PointerButton {
                pos: start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |ctx| tool.show(ctx),
    );
    let _ = ctx.run(
        viewport_input(vec![egui::Event::PointerMoved(end)]),
        |ctx| tool.show(ctx),
    );
    let _ = ctx.run(
        viewport_input(vec![egui::Event::PointerButton {
            pos: end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]),
        |ctx| tool.show(ctx),
    );

    assert!(tool.state_revision() > rev_before);
    assert!(tool.ruler_vertices[0].norm() > 1e-3);
    for v in tool.ruler_vertices {
        assert!(v.norm() <= POINCARE_DISK_RADIUS_MAX);
    }

    let gb_after = gauss_bonnet_triangle(
        tool.ruler_vertices[0],
        tool.ruler_vertices[1],
        tool.ruler_vertices[2],
    );
    assert!((gb_before.area - gb_after.area).abs() < 1e-10);
    assert!((gb_before.side_lengths[0] - gb_after.side_lengths[0]).abs() < 1e-10);
}

#[test]
fn test_poincare_vs_klein_projection_switching_and_state_roundtrip() {
    let mut tool = HyperbolicDiskTool::default();
    tool.apply_mobius_pan(Complex64::new(0.22, -0.17));
    let poincare_ds = tool.export_datasets();

    tool.model = HyperbolicModel::Klein;
    let klein_ds = tool.export_datasets();
    assert_eq!(poincare_ds.len(), klein_ds.len());

    // First polygon edge in Klein model must have collinear interior chord points,
    // whereas Poincare model curves along an orthogonal circular arc.
    let p_ring = &poincare_ds[1].1;
    let k_ring = &klein_ds[1].1;
    let k_dir = [k_ring[8][0] - k_ring[0][0], k_ring[8][1] - k_ring[0][1]];
    let k_mid_rel = [k_ring[4][0] - k_ring[0][0], k_ring[4][1] - k_ring[0][1]];
    let klein_cross = (k_mid_rel[0] * k_dir[1] - k_mid_rel[1] * k_dir[0]).abs();
    assert!(klein_cross < 1e-11);

    let p_dir = [p_ring[8][0] - p_ring[0][0], p_ring[8][1] - p_ring[0][1]];
    let p_mid_rel = [p_ring[4][0] - p_ring[0][0], p_ring[4][1] - p_ring[0][1]];
    let poincare_cross = (p_mid_rel[0] * p_dir[1] - p_mid_rel[1] * p_dir[0]).abs();
    assert!(poincare_cross > 1e-4);

    // Direct JSON state round-trip + PresetWrapper envelope round-trip + invalid input rejection.
    let direct_state = tool.save_state().expect("save_state");
    let mut direct_restored = HyperbolicDiskTool::default();
    direct_restored
        .try_load_state(&direct_state)
        .expect("try_load_state");
    let canonical_state = direct_restored.save_state().expect("canonical save_state");
    let mut second_restored = HyperbolicDiskTool::default();
    second_restored
        .try_load_state(&canonical_state)
        .expect("idempotent try_load_state");
    assert_eq!(second_restored.save_state(), Some(canonical_state));

    let preset = tool.export_preset_json().expect("export preset");
    let mut restored = HyperbolicDiskTool::default();
    restored
        .import_preset_json(&preset)
        .expect("import valid preset");
    assert_eq!(restored.model, tool.model);
    assert_eq!((restored.p, restored.q, restored.depth), (tool.p, tool.q, tool.depth));
    for idx in 0..3 {
        assert!((restored.ruler_vertices[idx] - tool.ruler_vertices[idx]).norm() < 1e-14);
    }
    assert!(!tool.apply_schlafli(4, 4, 2));
    assert!(restored
        .try_load_state(
            r#"{"model":"Poincare","p":7,"q":3,"depth":2,"ruler_re":[1.5,0.2,0.1],"ruler_im":[0.0,0.0,0.2],"view_coeffs":[1.0,0.0,0.0,0.0],"show_ruler":true}"#
        )
        .is_err());
}

#[test]
fn test_finite_state_guard_rollback_and_parameter_clamping_for_both_tools() {
    let ctx = egui::Context::default();
    let mut fw = SimulationFramework::new("hyperbolic_geometry");
    assert!(fw.select_tool_by_name("HyperbolicDiskTool"));
    let _ = ctx.run(viewport_input(Vec::new()), |ctx| {
        fw.show(ctx, "hyperbolic_geometry");
    });

    // Inject non-finite ruler coordinate -> FiniteStateGuard detects non-finite parameter and clamps/recovers.
    if let Some(disk) = fw
        .active_tool
        .as_mut()
        .and_then(|t| t.as_any_mut())
        .and_then(|a| a.downcast_mut::<HyperbolicDiskTool>())
    {
        disk.ruler_vertices[0] = Complex64::new(f64::NAN, 2.5);
        assert!(!disk.has_finite_parameters());
        assert!(disk.clamp_parameters());
        assert!(disk.has_finite_parameters());
    }

    let mut billiards = HyperbolicBilliardsTool {
        initial_re: f64::INFINITY,
        log10_epsilon: f64::NAN,
        ..Default::default()
    };
    assert!(!billiards.has_finite_parameters());
    assert!(billiards.clamp_parameters());
    assert!(billiards.has_finite_parameters());
}

#[test]
fn test_billiards_lyapunov_stepping_and_modal_exports() {
    let ctx = egui::Context::default();
    let mut fw = SimulationFramework::new("hyperbolic_geometry");
    assert!(fw.select_tool_by_name("HyperbolicBilliardsTool"));

    if let Some(tool) = fw.active_tool.as_mut() {
        for _ in 0..30 {
            tool.step_simulation();
        }
        let ds = tool.export_datasets();
        assert_eq!(ds.len(), 2);
        assert!(ds[0].1.len() >= 100);
    }

    for format in [ExportFormat::Csv, ExportFormat::Png, ExportFormat::Svg] {
        let _ = ctx.run(viewport_input(Vec::new()), |ctx| {
            egui_plot::clear_last_plot_datasets();
            fw.show_export_modal = true;
            fw.queue_export(format);
            fw.show(ctx, "hyperbolic_geometry");
        });
        assert_eq!(
            fw.export_status_severity,
            ExportStatusSeverity::Success,
            "Export {format:?} failed: {:?}",
            fw.export_status_message
        );
    }
}

// [cite:modular_polynomials_review]
