#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::atlas::{OdeAtlasRegime, OdeRegimeCartographer};
use super::phase::PlotViewMode;
use super::systems::eval_ode_final_state;
use super::*;
use crate::async_sim::atlas::{
    AtlasPassMeta, AtlasRenderRequest, AtlasRunner, AtlasTeleportEvent, RegimeCartographer,
    ATLAS_STRIDES,
};
use crate::async_sim::SimulationRunner;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::starter_gallery::StarterPresetMetadata;
use crate::url_state::{parse_share_location, MAX_DECODED_URL_STATE_BYTES};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

fn wcag_contrast_on_black(color: egui::Color32) -> f64 {
    let to_lin = |ch: u8| {
        let s = f64::from(ch) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let lum = 0.2126 * to_lin(color.r()) + 0.7152 * to_lin(color.g()) + 0.0722 * to_lin(color.b());
    (lum + 0.05) / 0.05
}

#[test]
fn test_non_autonomous_ode_clock_sine_decay_and_duffing_forcing() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        custom_coupled: false,
        method: IntegrationMethod::RungeKutta4,
        dt: 0.05,
        total_time: 10.0,
        param_k: 1.0,
        ic_y0: 1.0,
        ..Default::default()
    };
    tool.math_input.set_expression("-k * y * sin(t)");
    tool.math_input.set_param("k", 1.0);
    tool.recalculate();

    assert_eq!(tool.time_series.len(), 201);
    let mut max_err = 0.0_f64;
    for (&t, &y) in tool.time_series.iter().zip(&tool.y_series) {
        let exact = (t.cos() - 1.0).exp();
        max_err = max_err.max((y - exact).abs());
    }
    assert!(
        max_err < 1e-4,
        "RK4 non-autonomous y' = -k*y*sin(t) max error {max_err:.3e} exceeded 1e-4"
    );

    let spec = tool.build_custom_ode_spec();
    let p_map = HashMap::from([("k".to_string(), 1.0), ("y0".to_string(), 1.0)]);
    let final_y = eval_ode_final_state(
        OdePreset::Custom,
        IntegrationMethod::RungeKutta4,
        0.05,
        10.0,
        &p_map,
        spec.as_ref(),
    );
    assert!((final_y - (10.0_f64.cos() - 1.0).exp()).abs() < 1e-4);

    // Duffing forced oscillator from rest: advancing cos(w * t) reverses forcing sign across half-periods.
    let mut duffing = OdeSolverTool {
        preset: OdePreset::Custom,
        custom_coupled: true,
        dt: 0.02,
        total_time: 12.0,
        ic_y0: 0.0,
        ic_v0: 0.0,
        ..Default::default()
    };
    duffing
        .coupled_bank
        .set_expressions(&["v", "-c * v - a * y - b * y^3 + f * cos(w * t)"]);
    for (k, v) in [("c", 0.15), ("a", 0.0), ("b", 0.0), ("f", 1.0), ("w", 2.0)] {
        duffing.coupled_bank.set_param(k, v);
    }
    duffing.recalculate();
    let has_pos_v = duffing.v_series.iter().any(|&v| v > 0.15);
    let has_neg_v = duffing.v_series.iter().any(|&v| v < -0.15);
    assert!(
        has_pos_v && has_neg_v,
        "Duffing cos(w*t) forcing must alternate velocity sign as elapsed_time advances"
    );
}

#[test]
fn test_phase_portrait_quiver_slice_updates_and_multi_solver_projection() {
    // 1D non-autonomous quiver updates with t_slice on -k * y * sin(t)
    let mut scalar_tool = OdeSolverTool {
        preset: OdePreset::Custom,
        custom_coupled: false,
        view_mode: PlotViewMode::PhasePortrait,
        param_k: 1.0,
        ic_y0: 1.0,
        ..Default::default()
    };
    scalar_tool.math_input.set_expression("-k * y * sin(t)");
    scalar_tool.recalculate();
    scalar_tool.set_t_slice(0.0);
    let (_, tips_t0) = scalar_tool.quiver_vectors();
    scalar_tool.set_t_slice(std::f64::consts::FRAC_PI_2);
    let (_, tips_t_pi2) = scalar_tool.quiver_vectors();
    assert_ne!(tips_t0.as_slice(), tips_t_pi2.as_slice());

    // 2D coupled quiver updates with t_slice on Duffing f * cos(w * t)
    let mut duffing = OdeSolverTool {
        preset: OdePreset::Custom,
        custom_coupled: true,
        view_mode: PlotViewMode::PhasePortrait,
        ..Default::default()
    };
    duffing
        .coupled_bank
        .set_expressions(&["v", "-c * v - a * y - b * y^3 + f * cos(w * t)"]);
    for (k, v) in [("c", 0.2), ("a", 1.0), ("b", 0.5), ("f", 2.5), ("w", 1.0)] {
        duffing.coupled_bank.set_param(k, v);
    }
    duffing.recalculate();
    duffing.set_t_slice(0.0);
    let (_, d_tips_0) = duffing.quiver_vectors();
    duffing.set_t_slice(std::f64::consts::PI);
    let (_, d_tips_pi) = duffing.quiver_vectors();
    assert_ne!(d_tips_0.as_slice(), d_tips_pi.as_slice());

    // Multi-solver 2D Phase Portrait: Euler spirals outward vs RK4 closed orbit
    let mut osc = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        view_mode: PlotViewMode::PhasePortrait,
        dt: 0.08,
        total_time: 12.0,
        param_k: 1.0,
        ic_y0: 1.0,
        ic_v0: 0.0,
        ..Default::default()
    };
    osc.set_comparison_mode(true);
    let orbits = osc.projected_comparison_orbits();
    assert_eq!(orbits.len(), 3);
    let rk4_pts = &orbits
        .iter()
        .find(|(n, _)| n.contains("Runge-Kutta"))
        .unwrap()
        .1;
    let euler_pts = &orbits.iter().find(|(n, _)| n.contains("Euler")).unwrap().1;
    let rk4_end = rk4_pts.last().unwrap();
    let euler_end = euler_pts.last().unwrap();
    let rk4_r = rk4_end.x.hypot(rk4_end.y);
    let euler_r = euler_end.x.hypot(euler_end.y);
    assert!(
        (rk4_r - 1.0).abs() < 1e-2 && euler_r > 1.45,
        "Expected RK4 closed orbit (r={rk4_r:.4}) vs Euler outward spiral (r={euler_r:.4})"
    );
}

fn send_primary_click(
    ctx: &egui::Context,
    raw: &egui::RawInput,
    pos: egui::Pos2,
    tool: &mut OdeSolverTool,
) {
    for pressed in [true, false] {
        let mut inp = raw.clone();
        inp.events.push(egui::Event::PointerMoved(pos));
        inp.events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(inp, |ctx| tool.show(ctx));
    }
}

fn key_press_event(key: egui::Key) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
}

#[test]
fn test_phase_canvas_click_and_keyboard_seeding_with_accessibility() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        view_mode: PlotViewMode::PhasePortrait,
        ic_y0: 1.0,
        ic_v0: 0.0,
        ..Default::default()
    };
    tool.recalculate();

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000.0, 720.0),
        )),
        ..Default::default()
    };
    let out1 = ctx.run(raw.clone(), |ctx| tool.show(ctx));
    let ak = out1.platform_output.accesskit_update.as_ref().unwrap();
    let canvas_bounds = ak
        .nodes
        .iter()
        .find(|(_, n)| {
            n.role() == egui::accesskit::Role::Canvas
                && n.label().unwrap_or("").contains("Phase Portrait")
        })
        .and_then(|(_, n)| n.bounds())
        .expect("Phase Portrait Role::Canvas registered with bounds");

    let click_pos = egui::pos2(
        (0.62 * canvas_bounds.x1 + 0.38 * canvas_bounds.x0) as f32,
        (0.35 * canvas_bounds.y1 + 0.65 * canvas_bounds.y0) as f32,
    );
    send_primary_click(&ctx, &raw, click_pos, &mut tool);
    assert!(tool.focused_canvas_id().is_some());
    assert_eq!(tool.snapshot_manager.history_len(), 1);
    let seeded_after_click = (tool.ic_y0, tool.ic_v0);

    let mut key_input = raw;
    key_input.events.extend([
        key_press_event(egui::Key::ArrowRight),
        key_press_event(egui::Key::ArrowUp),
        key_press_event(egui::Key::Enter),
    ]);
    let _ = ctx.run(key_input, |ctx| tool.show(ctx));

    assert!(tool.ic_y0 > seeded_after_click.0 && tool.ic_v0 > seeded_after_click.1);
    assert_eq!(tool.snapshot_manager.history_len(), 2);
    assert_eq!(tool.projected_snapshot_orbits().len(), 2);
}

#[test]
fn test_ode_regime_cartographer_regimes_contour_and_zero_alloc() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        custom_coupled: true,
        ic_y0: 1.5,
        ic_v0: 0.0,
        ..Default::default()
    };
    tool.coupled_bank.set_expressions(&["v", "-k * y - c * v"]);
    tool.recalculate();

    let cart = OdeRegimeCartographer::from_tool(&tool, ("k", "c"));
    for (k, c, expected) in [
        (1.0, 2.6, OdeAtlasRegime::OverdampedDecay),
        (2.0, 0.4, OdeAtlasRegime::UnderdampedSpiral),
        (1.5, 0.0, OdeAtlasRegime::SustainedCycle),
        (1.0, -0.4, OdeAtlasRegime::UnstableDivergence),
    ] {
        assert_eq!(
            OdeAtlasRegime::from_u8(cart.evaluate_cell(k, c).regime),
            expected
        );
    }

    let (contour_label, contour_pts) = cart.overlay_contour().expect("c^2 = 4k contour");
    assert!(contour_label.contains("c² = 4k"));
    for &[k, c] in &contour_pts {
        assert!((c * c - 4.0 * k).abs() < 1e-9);
    }

    let axes = cart.axes();
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for sy in 0..24 {
            for sx in 0..24 {
                let (p1, p2) = axes.pixel_to_coords(sx, sy, 24, 24);
                let cell = cart.evaluate_cell(p1, p2);
                assert!(wcag_contrast_on_black(cart.color_for_cell(cell, true)) >= 3.0);
            }
        }
    });
}

#[test]
fn test_ode_regime_cartographer_progressive_passes_and_teleport() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        custom_coupled: true,
        ic_y0: 1.5,
        ic_v0: 0.0,
        ..Default::default()
    };
    tool.coupled_bank.set_expressions(&["v", "-k * y - c * v"]);
    tool.recalculate();
    let cart = OdeRegimeCartographer::from_tool(&tool, ("k", "c"));

    let req = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 32,
        height: 32,
        high_contrast: false,
        cartographer: Arc::new(cart),
    }));
    let mut runner = AtlasRunner::new(req, Arc::new(AtomicU64::new(1)));
    let mut observed_strides = Vec::new();
    for _ in 0..32 {
        runner.step();
        let snap = runner.get_snapshot();
        if let Some(meta) = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
        {
            if observed_strides.last().copied() != Some(meta.stride) {
                observed_strides.push(meta.stride);
            }
            if meta.is_final {
                break;
            }
        }
    }
    assert_eq!(observed_strides.first().copied(), Some(ATLAS_STRIDES[0]));
    assert_eq!(observed_strides.last().copied(), Some(1));

    tool.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "k",
        x_val: 3.25,
        y_key: "c",
        y_val: 0.75,
        regime_label: "Underdamped Spiral (c² < 4k)",
    });
    assert!((tool.param_k - 3.25).abs() < 1e-9);
    assert_eq!(tool.coupled_bank.params_map().get("c"), Some(&0.75));
    assert_eq!(tool.snapshot_manager.history_len(), 1);
}

fn populate_vdp_snapshots(tool: &mut OdeSolverTool) {
    tool.dt = 0.02;
    tool.total_time = 20.0;
    tool.set_t_slice(1.25);
    tool.atlas.set_axis_keys("mu", "dt").unwrap();
    tool.recalculate();
    for idx in 0..10 {
        let series = tool.build_current_trace_series();
        assert!(tool.snapshot_manager.add_snapshot(
            format!("Pinned {idx}"),
            tool.param_summary(),
            series,
        ));
    }
    for _ in 0..5 {
        let series = tool.build_current_trace_series();
        tool.snapshot_manager
            .push_history_run(tool.param_summary(), series);
    }
}

#[test]
fn test_export_url_state_omits_snapshots_under_1200b_and_roundtrips_starter() {
    let mut fw = SimulationFramework::new("analysis");
    let vdp = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .find(|p| p.id == "van_der_pol_phase_portrait")
        .expect("van_der_pol_phase_portrait starter registered");
    fw.apply_starter_card(vdp.title, vdp.tool_name, Some(vdp.generate_preset()))
        .expect("launch Van der Pol starter card");

    {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<OdeSolverTool>()
            .unwrap();
        assert_eq!(tool.view_mode(), PlotViewMode::PhasePortrait);
        assert!(tool.custom_coupled);
        populate_vdp_snapshots(tool);
        let full_file_json = tool.export_preset_json().expect("full file export");
        assert!(full_file_json.contains("snapshot_manager") && full_file_json.len() > 20_000);
    }

    let url_preset = fw.export_url_preset_json().unwrap().unwrap();
    assert!(!url_preset.contains("snapshot_manager"));
    assert!(url_preset.len() < 1_200 && url_preset.len() < MAX_DECODED_URL_STATE_BYTES);

    let share_url = fw.encode_share_url().expect("encode share url");
    let hash = &share_url[share_url.find('#').unwrap()..];
    let payload = parse_share_location(hash, "").unwrap().unwrap();
    let mut restored_fw = SimulationFramework::new("analysis");
    restored_fw
        .import_url_preset_json(payload.tool.as_deref(), payload.state_json.as_deref())
        .expect("import url preset");
    let restored = restored_fw
        .active_tool
        .as_ref()
        .unwrap()
        .as_any()
        .unwrap()
        .downcast_ref::<OdeSolverTool>()
        .unwrap();
    assert_eq!(restored.view_mode(), PlotViewMode::PhasePortrait);
    assert!((restored.t_slice() - 1.25).abs() < 1e-9);
    assert_eq!(restored.atlas.axis_keys(), ("mu", "dt"));
    assert!(restored.snapshot_manager.is_empty());
}

#[test]
fn test_ode_app_accesskit_exposes_phase_and_atlas_controls() {
    use eframe::App;

    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let mut app = crate::app::MathExplorerApp::default();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut frame = eframe::Frame::_new_kittest();
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0));
    let tab_idx = app
        .tabs
        .iter()
        .position(|t| t.name().contains("Analysis"))
        .expect("Analysis tab");
    app.selected_tab = tab_idx;
    assert!(app.tabs[tab_idx].select_tool_by_name("OdeSolverTool"));

    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(rect),
            ..Default::default()
        },
        |ctx| app.update(ctx, &mut frame),
    );
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let has_phase = ak
        .nodes
        .iter()
        .filter_map(|(_, n)| n.label())
        .any(|l| l.contains("Phase Portrait"));
    let has_atlas = ak
        .nodes
        .iter()
        .filter_map(|(_, n)| n.label())
        .any(|l| l.contains("2D Parameter Atlas"));
    assert!(has_phase && has_atlas);
}
