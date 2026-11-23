#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::framework::InteractiveTool;
use eframe::egui::{self, Event, Modifiers, PointerButton, Pos2, RawInput, Rect, vec2};
use math_commons::ast::parse_equation;
use std::sync::{Arc, RwLock};

fn wcag_contrast_on_black(color: egui::Color32) -> f64 {
    let channel = |c: u8| {
        let s = f64::from(c) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let lum =
        0.2126 * channel(color.r()) + 0.7152 * channel(color.g()) + 0.0722 * channel(color.b());
    (lum + 0.05) / 0.05
}

fn build_test_cartographers() -> (
    QuantumTunnelingCartographer,
    QuantumTunnelingCartographer,
    QuantumTunnelingCartographer,
) {
    let params = wave_parameter_constraints();
    let (_, custom_expr) =
        parse_equation("v0 * exp(-(x / width)^4)").expect("parse super-Gaussian");
    let cart_double = QuantumTunnelingCartographer::new(
        &params,
        ("k0", "v0"),
        WavePotentialPreset::DoubleBarrier,
        4.6385,
        20.0,
        1.0,
        0.55,
        &custom_expr,
        &[],
    );
    let cart_width = QuantumTunnelingCartographer::new(
        &params,
        ("k0", "width"),
        WavePotentialPreset::RectangularBarrier,
        5.0,
        15.0,
        0.5,
        0.5,
        &custom_expr,
        &[],
    );
    let cart_custom = QuantumTunnelingCartographer::new(
        &params,
        ("k0", "v0"),
        WavePotentialPreset::Custom,
        4.5,
        18.0,
        0.8,
        0.5,
        &custom_expr,
        &[(96, 12.0), (98, 12.0)],
    );
    (cart_double, cart_width, cart_custom)
}

#[test]
fn test_quantum_cartographer_zero_allocation_custom_ast_and_wcag_contrast() {
    let (cart_double, cart_width, cart_custom) = build_test_cartographers();
    let (axes_db, axes_w, axes_c) = (cart_double.axes(), cart_width.axes(), cart_custom.axes());
    assert!(cart_double.overlay_contour().is_some() && cart_width.overlay_contour().is_some());
    let shared_req = Arc::new(RwLock::new(crate::async_sim::atlas::AtlasRenderRequest {
        generation: 1,
        width: 128,
        height: 128,
        high_contrast: true,
        cartographer: Arc::new(cart_double.clone()),
    }));
    let latest_gen = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let mut atlas_runner = crate::async_sim::atlas::AtlasRunner::new(shared_req, latest_gen);

    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            while !atlas_runner.is_complete() {
                atlas_runner.step();
            }
            for sy in 0..24 {
                for sx in 0..24 {
                    let (k0, v0) = axes_db.pixel_to_coords(sx, sy, 24, 24);
                    let cell = cart_double.evaluate_cell(k0, v0);
                    assert!(cell.primary.is_finite() && (0.0..=1.0).contains(&cell.primary));
                    assert!(cell.secondary.is_finite() && (0.0..=1.0).contains(&cell.secondary));
                    assert!(wcag_contrast_on_black(cart_double.color_for_cell(cell, true)) >= 3.0);

                    let (kw, w) = axes_w.pixel_to_coords(sx, sy, 24, 24);
                    let cell_w = cart_width.evaluate_cell(kw, w);
                    assert!(wcag_contrast_on_black(cart_width.color_for_cell(cell_w, true)) >= 3.0);

                    let (kc, vc) = axes_c.pixel_to_coords(sx, sy, 24, 24);
                    let cell_c = cart_custom.evaluate_cell(kc, vc);
                    assert!(cell_c.primary.is_finite() && (0.0..=1.0).contains(&cell_c.primary));
                }
            }
        });
    }
    assert!(atlas_runner.is_complete());

    let res_cell = cart_double.evaluate_cell(4.6385, 20.0);
    assert!(
        res_cell.primary > 0.99,
        "Expected Fabry-Pérot stationary T > 0.99 at k0=4.6385, got {}",
        res_cell.primary
    );
    assert_eq!(
        QuantumTunnelingRegime::from_u8(res_cell.regime),
        QuantumTunnelingRegime::FabryPerotResonance
    );
}

#[test]
fn test_double_barrier_quasi_bound_resonance_latches_finite_pre_wall_transmission() {
    let cfg = WaveSimConfig {
        preset: WavePotentialPreset::DoubleBarrier,
        x0: -2.5,
        k0: 4.6385,
        sigma: 0.55,
        v0: 20.0,
        width: 1.0,
        dt: 0.02,
        steps_per_frame: 5,
        custom_expr: presets::default_compiled_expr(),
        sparse_potential: Vec::new(),
    };
    let mut runner = WaveRunner::new(Arc::new(RwLock::new(cfg)));
    assert!(runner.latched_transmission().is_none());

    let mut latched_at_wall: Option<f64> = None;
    for _ in 0..160 {
        runner.step();
        if let Some(val) = runner.latched_transmission() {
            latched_at_wall = Some(val);
            break;
        }
    }
    let latched = latched_at_wall.expect(
        "DoubleBarrier Fabry-Pérot wavepacket must latch finite transmission before wall bounce",
    );
    assert!(
        latched.is_finite() && latched > 0.08 && latched <= 1.0,
        "Expected finite resonant wavepacket transmission > 0.08, got {latched}"
    );
    assert!(runner.build_wave_data().quasi_bound_latched);

    // Off-resonant k0 = 3.80 must transmit > 5x less than resonant k0 = 4.6385
    let off_cfg = WaveSimConfig {
        preset: WavePotentialPreset::DoubleBarrier,
        x0: -2.5,
        k0: 3.80,
        sigma: 0.55,
        v0: 20.0,
        width: 1.0,
        dt: 0.02,
        steps_per_frame: 5,
        custom_expr: presets::default_compiled_expr(),
        sparse_potential: Vec::new(),
    };
    let mut off_runner = WaveRunner::new(Arc::new(RwLock::new(off_cfg)));
    for _ in 0..80 {
        off_runner.step();
    }
    assert!(latched > 5.0 * off_runner.build_wave_data().transmission.max(1e-6));

    // Step well past right-wall reflection and confirm the latched readout stays frozen.
    for _ in 0..80 {
        runner.step();
    }
    assert_eq!(runner.latched_transmission(), Some(latched));
}

#[test]
fn test_double_barrier_default_packet_latches_at_k0_4_6385() {
    let cfg = WaveSimConfig {
        preset: WavePotentialPreset::DoubleBarrier,
        k0: 4.6385,
        v0: 20.0,
        width: 1.0,
        ..Default::default()
    };
    let mut runner = WaveRunner::new(Arc::new(RwLock::new(cfg)));
    for _ in 0..160 {
        runner.step();
        if runner.latched_transmission().is_some() {
            break;
        }
    }
    let latched = runner
        .latched_transmission()
        .expect("Default-packet DoubleBarrier at k0=4.6385, v0=20.0, width=1.0 must latch");
    assert!(latched.is_finite() && latched > 0.03 && latched <= 1.0);
}

fn locate_k0_slider_center(
    ctx: &egui::Context,
    screen: Rect,
    tool: &mut WaveSimulatorTool,
) -> Pos2 {
    let out = ctx.run(
        RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit tree must be emitted");
    let b = ak
        .nodes
        .iter()
        .find_map(|(_, node)| {
            (node.role() == egui::accesskit::Role::Slider
                && node.label().unwrap_or_default().contains("Wave Number"))
            .then(|| node.bounds())
            .flatten()
        })
        .expect("k0 Slider node must have AccessKit bounds");
    Pos2::new(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32)
}

fn run_pointer_frame(
    ctx: &egui::Context,
    screen: Rect,
    events: Vec<Event>,
    tool: &mut WaveSimulatorTool,
) {
    let _ = ctx.run(
        RawInput {
            screen_rect: Some(screen),
            events,
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
}

#[test]
fn test_mid_drag_slider_motion_defers_propagator_rebuilds_until_release() {
    let mut tool = WaveSimulatorTool::new();
    tool.preset = WavePotentialPreset::RectangularBarrier;
    tool.sync_config_and_reset(None);
    tool.step_simulation();
    let baseline_rebuilds = tool.propagator_rebuild_count();
    let (baseline_k0, baseline_t) = (tool.k0, tool.stationary_and_wavepacket_transmission().0);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen = Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 900.0));
    let start_pos = locate_k0_slider_center(&ctx, screen, &mut tool);

    run_pointer_frame(
        &ctx,
        screen,
        vec![
            Event::PointerMoved(start_pos),
            Event::PointerButton {
                pos: start_pos,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: Modifiers::NONE,
            },
        ],
        &mut tool,
    );
    for step_idx in 1..=5 {
        let drag_pos = start_pos + vec2(10.0 * (step_idx as f32), 0.0);
        run_pointer_frame(&ctx, screen, vec![Event::PointerMoved(drag_pos)], &mut tool);
    }

    assert!((tool.k0 - baseline_k0).abs() > 1.0e-4 && tool.pending_slider_reset);
    assert_eq!(tool.propagator_rebuild_count(), baseline_rebuilds);
    assert!((tool.stationary_and_wavepacket_transmission().0 - baseline_t).abs() > 1.0e-4);

    let end_pos = start_pos + vec2(50.0, 0.0);
    run_pointer_frame(
        &ctx,
        screen,
        vec![Event::PointerButton {
            pos: end_pos,
            button: PointerButton::Primary,
            pressed: false,
            modifiers: Modifiers::NONE,
        }],
        &mut tool,
    );
    assert!(!tool.pending_slider_reset);
    tool.step_simulation();
    assert!(tool.propagator_rebuild_count() > baseline_rebuilds);
}

#[test]
fn test_fabry_perot_challenge_and_atlas_state_persistence() {
    let mut tool = WaveSimulatorTool::new();
    let default_json = tool.save_state().expect("save default wave state");
    assert!(!default_json.contains("atlas_visible"));
    assert!(!default_json.contains("atlas_axes"));

    tool.atlas.set_visible(true);
    tool.atlas
        .set_axis_keys("k0", "width")
        .expect("select (k0, width) axes");
    let atlas_json = tool.save_state().expect("save open atlas state");
    assert!(atlas_json.contains("atlas_visible"));
    assert!(atlas_json.contains("atlas_axes"));

    let mut restored = WaveSimulatorTool::new();
    restored
        .try_load_state(&atlas_json)
        .expect("deserialize atlas state");
    assert!(restored.atlas.is_visible());
    assert_eq!(restored.atlas.axis_keys(), ("k0", "width"));

    tool.apply_challenge(Some(CHALLENGE_FABRY_PEROT_RESONANCE))
        .expect("activate Fabry-Pérot challenge");
    tool.k0 = challenge::CHALLENGE_FP_BASELINE_K0;
    let baseline_eval = tool.evaluate_challenge().expect("baseline challenge eval");
    assert!(!baseline_eval.passed);

    tool.apply_atlas_teleport(
        AtlasTeleportEvent {
            x_key: "k0",
            x_val: 4.6385,
            y_key: "v0",
            y_val: 12.0, // Locked by challenge to V0 = 20.0
            regime_label: "Fabry-Pérot Resonance (E₀ < V₀)",
        },
        None,
    );
    assert!((tool.v0 - challenge::CHALLENGE_FP_V0).abs() < 1e-12);
    let solved_eval = tool.evaluate_challenge().expect("solved challenge eval");
    assert!(
        solved_eval.passed,
        "Expected challenge pass: {solved_eval:?}"
    );
}
