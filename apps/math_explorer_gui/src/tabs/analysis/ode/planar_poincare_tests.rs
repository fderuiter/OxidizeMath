#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::phase_plane::{MAX_QUIVER_GRID, MAX_SEEDED_ORBITS};
use super::poincare_map::{
    classify_equilibrium, companion_matrix_from_trace_det, trace_det_from_matrix, EquilibriumRegime,
};
use super::*;
use crate::framework::{InteractiveTool, SimulationFramework, TraceSeries};
use crate::presets::{OdePreset, MAX_ODE_STEPS};
use eframe::egui;

fn collect_shape_texts(shapes: &[egui::epaint::Shape], out: &mut Vec<(String, egui::Pos2)>) {
    for shape in shapes {
        match shape {
            egui::epaint::Shape::Text(t) => {
                let center = t.pos + 0.5 * t.galley.size();
                out.push((t.galley.job.text.clone(), center));
            }
            egui::epaint::Shape::Vec(inner) => collect_shape_texts(inner, out),
            _ => {}
        }
    }
}

fn render_tool_frame_with_positions(
    ctx: &egui::Context,
    tool: &mut OdeSolverTool,
    events: Vec<egui::Event>,
) -> Vec<(String, egui::Pos2)> {
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 820.0),
        )),
        events,
        ..Default::default()
    };
    let full = ctx.run(raw, |ctx| {
        tool.show(ctx);
    });
    let shapes: Vec<_> = full.shapes.into_iter().map(|cs| cs.shape).collect();
    let mut texts = Vec::new();
    collect_shape_texts(&shapes, &mut texts);
    texts
}

fn render_tool_frame(
    ctx: &egui::Context,
    tool: &mut OdeSolverTool,
    events: Vec<egui::Event>,
) -> Vec<String> {
    render_tool_frame_with_positions(ctx, tool, events)
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

#[test]
fn test_non_autonomous_custom_ode_advances_explicit_time_t() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        dt: 0.02,
        total_time: std::f64::consts::PI,
        param_k: 1.25,
        ic_y0: 1.5,
        ..Default::default()
    };
    tool.math_input.set_expression("-k * y * sin(t)");
    tool.recalculate();

    assert!(!tool.diverged());
    let (&last_t, &last_y) = tool
        .time_series
        .last()
        .zip(tool.y_series.last())
        .expect("non-empty trajectory");
    let exact = tool.ic_y0 * (tool.param_k * (last_t.cos() - 1.0)).exp();
    assert!(
        (last_y - exact).abs() < 1e-5,
        "expected analytical y(T)={exact}, got {last_y} (static t=0 would give {})",
        tool.ic_y0
    );
}

#[test]
fn test_planar_custom_vdp_and_lotka_volterra_quiver_and_nullclines() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::PlanarCustom,
        phase_plane_mode: true,
        show_quiver: true,
        show_nullclines: true,
        quiver_density: 120,
        ic_y0: 1.5,
        ic_v0: 0.5,
        param_k: 1.0,
        param_r: 1.2,
        ..Default::default()
    };
    tool.recalculate();
    assert!(tool.is_phase_plane_active());
    assert!(!tool.diverged());
    assert_eq!(tool.y_series.len(), tool.v_series.len());

    let (origins, tips) = tool.compute_quiver_arrows();
    assert_eq!(origins.len(), tips.len());
    assert!(!origins.is_empty());
    assert!(origins.len() <= MAX_QUIVER_GRID * MAX_QUIVER_GRID);

    let (x_null, y_null) = tool.compute_nullcline_segments();
    assert!(!x_null.is_empty() && !y_null.is_empty());

    tool.planar_x_input.set_expression("r * x - k * x * y");
    tool.planar_y_input.set_expression("k * x * y - y");
    tool.recalculate();
    assert!(!tool.diverged());
    assert!(tool
        .y_series
        .iter()
        .chain(&tool.v_series)
        .all(|v| v.is_finite()));
}

#[test]
fn test_phase_plane_seed_cap_and_pointer_click_seeding() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        phase_plane_mode: true,
        dt: 0.001,
        total_time: 100.0,
        ..Default::default()
    };
    assert_eq!(tool.preset.state_labels(), ("y(t)", "y'(t)", "y", "y'"));

    for i in 0..(MAX_SEEDED_ORBITS + 2) {
        tool.seed_orbit([0.5 + i as f64 * 0.25, -0.4]);
    }
    assert_eq!(tool.seeded_ic.len(), MAX_SEEDED_ORBITS);
    assert_eq!(tool.seeded_trajectories.len(), MAX_SEEDED_ORBITS);
    assert!((tool.seeded_ic[0][0] - 1.0).abs() < 1e-12);

    let total_seeded_steps: usize = tool
        .seeded_trajectories
        .iter()
        .map(|t| t.time_series.len().saturating_sub(1))
        .sum();
    assert!(total_seeded_steps <= MAX_ODE_STEPS);

    tool.clear_seeded_orbits();
    tool.dt = 0.05;
    tool.total_time = 6.0;
    tool.recalculate();

    let ctx = egui::Context::default();
    let _ = render_tool_frame(&ctx, &mut tool, Vec::new());
    let click_pos = egui::pos2(860.0, 260.0);
    let click_events = vec![
        egui::Event::PointerMoved(click_pos),
        egui::Event::PointerButton {
            pos: click_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: click_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = render_tool_frame(&ctx, &mut tool, click_events);
    assert_eq!(tool.seeded_ic.len(), 1);
    assert_eq!(tool.seeded_trajectories.len(), 1);
}

#[test]
fn test_poincare_all_six_regimes_and_minimap_labels() {
    let cases = [
        (0.5, -1.5, EquilibriumRegime::Saddle),
        (-3.0, 1.5, EquilibriumRegime::StableNode),
        (3.0, 1.5, EquilibriumRegime::UnstableNode),
        (-1.2, 2.5, EquilibriumRegime::StableSpiral),
        (1.2, 2.5, EquilibriumRegime::UnstableSpiral),
        (0.0, 2.0, EquilibriumRegime::Center),
    ];
    let mut tool = OdeSolverTool::default();
    for (tau, delta, expected) in cases {
        assert_eq!(classify_equilibrium(tau, delta), expected);
        tool.apply_poincare_point(tau, delta);
        assert_eq!(tool.preset, OdePreset::LinearSystem2D);
        assert!(tool.is_phase_plane_active());
        assert_eq!(tool.matrix_a, companion_matrix_from_trace_det(tau, delta));
        let (tr, det) = trace_det_from_matrix(tool.matrix_a);
        assert!((tr - tau).abs() < 1e-12 && (det - delta).abs() < 1e-12);
        assert!(!tool.diverged() && !tool.y_series.is_empty());
    }

    let ctx = egui::Context::default();
    let positioned = render_tool_frame_with_positions(&ctx, &mut tool, Vec::new());
    for regime in EquilibriumRegime::ALL {
        assert!(
            positioned.iter().any(|(t, _)| t.contains(regime.label())),
            "missing Poincaré minimap label '{}'",
            regime.label()
        );
    }
}

#[test]
fn test_poincare_minimap_click_syncs_companion_matrix_in_one_frame() {
    let mut tool = OdeSolverTool::default();
    tool.apply_poincare_point(0.0, 2.0);
    let ctx = egui::Context::default();
    let positioned = render_tool_frame_with_positions(&ctx, &mut tool, Vec::new());
    let saddle_pos = positioned
        .iter()
        .find(|(t, _)| t == "Saddle")
        .map(|(_, pos)| *pos)
        .expect("Saddle label on minimap");
    let before_y = tool.y_series.clone();
    let click = vec![
        egui::Event::PointerMoved(saddle_pos),
        egui::Event::PointerButton {
            pos: saddle_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: saddle_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = render_tool_frame(&ctx, &mut tool, click);
    let (tau, delta) = trace_det_from_matrix(tool.matrix_a);
    assert_eq!(classify_equilibrium(tau, delta), EquilibriumRegime::Saddle);
    assert_eq!(tool.matrix_a, companion_matrix_from_trace_det(tau, delta));
    assert_ne!(tool.y_series, before_y);
}

#[test]
fn test_incomplete_ast_retains_trajectory_without_false_divergence() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("-k * y * sin(t)");
    tool.recalculate();
    let valid_scalar = tool.y_series.clone();
    assert!(!valid_scalar.is_empty());

    tool.math_input.set_expression("cos(");
    tool.recalculate();
    assert!(tool.math_input.error_message().is_some());
    assert!(!tool.diverged());
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert_eq!(tool.y_series, valid_scalar);

    let ctx = egui::Context::default();
    let texts = render_tool_frame(&ctx, &mut tool, Vec::new());
    assert!(texts.iter().any(|t| t.contains("Syntax Error:")));
    assert!(!texts.iter().any(|t| t.contains("Auto-scale Step Size")));

    tool.preset = OdePreset::PlanarCustom;
    tool.phase_plane_mode = true;
    tool.recalculate();
    let valid_planar = (tool.y_series.clone(), tool.v_series.clone());
    tool.planar_y_input.set_expression("cos(");
    tool.recalculate();
    assert!(!tool.diverged());
    assert_eq!((tool.y_series, tool.v_series), valid_planar);
}

fn populate_planar_history_edits(tool: &mut OdeSolverTool) {
    tool.preset = OdePreset::PlanarCustom;
    tool.phase_plane_mode = true;
    tool.show_nullclines = true;
    tool.planar_x_input.set_expression("r * x - k * x * y");
    tool.planar_y_input.set_expression("k * x * y - y");
    tool.matrix_a = [[0.0, 1.0], [-2.0, -0.5]];
    tool.seed_orbit([1.2, 0.8]);
    tool.seed_orbit([0.6, 1.4]);
    for edit_idx in 0..6 {
        tool.param_k = 0.8 + 0.1 * edit_idx as f64;
        tool.recalculate();
        let pts: Vec<[f64; 2]> = tool
            .time_series
            .iter()
            .copied()
            .zip(tool.y_series.iter().copied())
            .map(|(t, y)| [t, y])
            .collect();
        tool.snapshot_manager
            .push_history_run(tool.param_summary(), vec![TraceSeries::new("x(t)", pts)]);
    }
    let full_preset = tool.export_preset_json().expect("full preset JSON");
    assert!(full_preset.contains("snapshot_manager"));
    assert!(full_preset.len() > crate::url_state::MAX_DECODED_URL_STATE_BYTES);
}

#[test]
fn test_url_share_after_many_edits_stays_under_budget_and_roundtrips() {
    let mut fw = SimulationFramework::new("analysis");
    let ode_idx = fw
        .available_tools
        .iter()
        .position(|m| m.id == "OdeSolverTool")
        .expect("OdeSolverTool registered");
    fw.select_tool(ode_idx);

    let tool = fw
        .active_tool
        .as_mut()
        .and_then(|t| t.as_any_mut())
        .and_then(|a| a.downcast_mut::<OdeSolverTool>())
        .expect("downcast OdeSolverTool");
    populate_planar_history_edits(tool);

    let url_json = fw
        .export_url_preset_json()
        .expect("url preset ok")
        .expect("some json");
    assert!(!url_json.contains("snapshot_manager"));
    assert!(url_json.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);

    let url = fw.encode_share_url().expect("encode_share_url succeeds");
    let (_, hash) = url.split_once('#').expect("share url fragment");
    let parsed = crate::url_state::parse_share_location(hash, "")
        .expect("parse_share_location ok")
        .expect("share params");
    let mut restored_fw = SimulationFramework::new("analysis");
    restored_fw
        .import_url_preset_json(parsed.tool.as_deref(), parsed.state_json.as_deref())
        .expect("import_url_preset_json succeeds");
    let restored = restored_fw
        .active_tool
        .as_ref()
        .and_then(|t| t.as_any())
        .and_then(|a| a.downcast_ref::<OdeSolverTool>())
        .expect("restored OdeSolverTool");
    assert_eq!(restored.preset, OdePreset::PlanarCustom);
    assert!(restored.phase_plane_mode && restored.show_nullclines);
    assert_eq!(restored.planar_x_input.raw_text, "r * x - k * x * y");
    assert_eq!(restored.planar_y_input.raw_text, "k * x * y - y");
    assert_eq!(restored.matrix_a, [[0.0, 1.0], [-2.0, -0.5]]);
    assert_eq!(restored.seeded_ic, vec![[1.2, 0.8], [0.6, 1.4]]);
}
