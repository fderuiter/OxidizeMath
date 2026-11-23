#![cfg_attr(any(), verified(opt_out = "tests"))]
//! Zero-allocation lattice recurrence, `DoubleBarrier` scatter latch, 2D Parameter Atlas,
//! unsolved inverse `ChallengeMetadata`, and `QuantumTab` `observables()` contract tests.

use super::super::atlas::{compute_lattice_steady_transmission_from_slice, WaveCartographer};
use super::super::challenges::{CHALLENGE_DOUBLE_BARRIER_RESONANCE, CHALLENGE_SUBBARRIER_50_SPLIT};
use super::super::presets::{evaluate_potential_grid, GRID_POINTS, GRID_X_MAX, GRID_X_MIN};
use super::super::*;
use crate::async_sim::atlas::{
    AtlasTeleportEvent, RegimeCartographer, ATLAS_STRIDES, DEFAULT_ATLAS_SIDE,
};
use crate::async_sim::SimulationRunner;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::starter_gallery::find_challenge_metadata;
use crate::tabs::quantum::{
    clebsch::ClebschGordanTool, spin_viz::SpinVisualizer,
    volumetric_studio::VolumetricFieldStudioTool,
};
use eframe::egui::{self, Color32};
use std::sync::{Arc, RwLock};

fn contrast_ratio(a: Color32, b: Color32) -> f64 {
    let chan = |c: u8| {
        let v = f64::from(c) / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let lum = |c: Color32| 0.2126 * chan(c.r()) + 0.7152 * chan(c.g()) + 0.0722 * chan(c.b());
    let (l1, l2) = (lum(a), lum(b));
    (l1.max(l2) + 0.05) / (l1.min(l2) + 0.05)
}

#[test]
fn test_zero_alloc_lattice_recurrence_and_double_barrier_resonances() {
    let dx = (GRID_X_MAX - GRID_X_MIN) / (GRID_POINTS as f64 - 1.0);
    let x_axis: Vec<f64> = (0..GRID_POINTS)
        .map(|i| GRID_X_MIN + i as f64 * dx)
        .collect();
    let pot_vec = evaluate_potential_grid(
        WavePotentialPreset::DoubleBarrier,
        &x_axis,
        28.0,
        0.8,
        dx,
        &presets::default_compiled_expr(),
    );
    let mut pot = [0.0_f64; GRID_POINTS];
    pot.copy_from_slice(pot_vec.as_slice());

    let carto = WaveCartographer::new(
        ("k0", "v0"),
        WavePotentialPreset::DoubleBarrier,
        28.0,
        0.8,
        &presets::default_compiled_expr(),
        &[],
    );

    let (mut t_r1, mut t_r2, mut t_off, mut cell_sample) = (0.0, 0.0, 1.0, 0.0);
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            t_r1 = compute_lattice_steady_transmission_from_slice(&pot, dx, 2.898);
            t_r2 = compute_lattice_steady_transmission_from_slice(&pot, dx, 5.671);
            t_off = compute_lattice_steady_transmission_from_slice(&pot, dx, 4.20);
            cell_sample = carto.evaluate_cell(5.671, 28.0).primary;
        });
    }

    assert!(
        t_r1 >= 0.99,
        "Expected k0=2.898 resonance >= 0.99, got {t_r1:.6}"
    );
    assert!(
        t_r2 >= 0.99,
        "Expected k0=5.671 resonance >= 0.99, got {t_r2:.6}"
    );
    assert!(
        t_off < 0.03,
        "Expected k0=4.20 off-resonance < 0.03, got {t_off:.6}"
    );
    assert!(cell_sample >= 0.95);
}

#[test]
fn test_double_barrier_central_well_ringing_latches_pre_wall_transmission() {
    let cfg = WaveSimConfig {
        preset: WavePotentialPreset::DoubleBarrier,
        x0: -2.5,
        k0: 5.671,
        sigma: 0.5,
        v0: 28.0,
        width: 0.8,
        dt: 0.02,
        steps_per_frame: 2,
        custom_expr: presets::default_compiled_expr(),
        sparse_potential: Vec::new(),
    };
    let mut runner = WaveRunner::new(Arc::new(RwLock::new(cfg)));
    assert!(runner.steady_transmission() >= 0.99);
    assert!(runner.latched_transmission().is_none());

    let mut latched_while_ringing = None;
    for _ in 0..120 {
        runner.step();
        if latched_while_ringing.is_none() {
            latched_while_ringing = runner.latched_transmission();
        }
    }
    let latched = latched_while_ringing
        .expect("DoubleBarrier must latch pre-wall transmission despite central-well ringing");
    assert!(
        latched.is_finite() && latched > 0.01 && latched <= 1.0,
        "Expected positive finite pre-wall latched_transmission on DoubleBarrier, got {latched}"
    );

    for _ in 0..60 {
        runner.step();
    }
    assert_eq!(runner.latched_transmission(), Some(latched));
}

#[cfg(not(target_arch = "wasm32"))]
fn save_atlas_preview_png(carto: &dyn RegimeCartographer, path: &str) {
    let side = 256usize;
    let axes = carto.axes();
    let (x0, x1) = (axes.x.constraint.min, axes.x.constraint.max);
    let (y0, y1) = (axes.y.constraint.min, axes.y.constraint.max);
    let mut rgba = vec![0u8; side * side * 4];
    for py in 0..side {
        let v0 = y1 - (py as f64 / (side - 1) as f64) * (y1 - y0);
        for px in 0..side {
            let k0 = x0 + (px as f64 / (side - 1) as f64) * (x1 - x0);
            let e0 = 0.5 * k0 * k0;
            let c = if (e0 - v0).abs() < 0.45 {
                Color32::from_rgb(255, 255, 255)
            } else {
                carto.color_for_cell(carto.evaluate_cell(k0, v0), false)
            };
            let idx = (py * side + px) * 4;
            rgba[idx..idx + 4].copy_from_slice(&[c.r(), c.g(), c.b(), 255]);
        }
    }
    if let Ok(file) = std::fs::File::create(path) {
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), side as u32, side as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        if let Ok(mut writer) = enc.write_header() {
            let _ = writer.write_image_data(&rgba);
        }
    }
}

#[test]
fn test_wave_parameter_atlas_axes_contour_contrast_and_teleport_all_presets() {
    assert_eq!(ATLAS_STRIDES, [8, 4, 2, 1]);
    assert_eq!(DEFAULT_ATLAS_SIDE, 128);
    let mut tool = WaveSimulatorTool::new();
    tool.atlas.set_visible(true);
    for (idx, expected_axes) in [(0, ("k0", "v0")), (1, ("k0", "width"))] {
        tool.atlas.select_axis_pair_index(idx);
        assert_eq!(tool.atlas.axis_keys(), expected_axes);
        let carto = tool.build_wave_cartographer(expected_axes);
        let curve = carto.overlay_contour();
        if expected_axes.1 == "v0" {
            let (_, pts) = curve.expect("E0 = V0 contour must be present on (k0, v0) slice");
            assert!(pts.iter().all(|p| (p[1] - 0.5 * p[0] * p[0]).abs() < 1e-9));
        }
        let c_low_hc = carto.color_for_cell(carto.evaluate_cell(0.2, 35.0), true);
        let c_high_hc = carto.color_for_cell(carto.evaluate_cell(11.5, 2.0), true);
        assert!(contrast_ratio(c_low_hc, Color32::BLACK) >= 3.0);
        assert!(contrast_ratio(c_high_hc, Color32::BLACK) >= 3.0);
    }

    for preset in WavePotentialPreset::ALL {
        tool.preset = preset;
        tool.width = 0.8;
        tool.apply_atlas_teleport(
            AtlasTeleportEvent {
                x_key: "k0",
                y_key: "v0",
                x_val: 5.671,
                y_val: 28.0,
                regime_label: "Resonant Peak",
            },
            None,
        );
        let obs: std::collections::HashMap<_, _> = tool.observables().into_iter().collect();
        assert!(obs["steady_transmission"].is_finite());
        assert!(obs["latched_transmission"].is_finite());
        if preset == WavePotentialPreset::DoubleBarrier {
            assert!(obs["steady_transmission"] >= 0.99);
            #[cfg(not(target_arch = "wasm32"))]
            save_atlas_preview_png(
                &*tool.build_wave_cartographer(("k0", "v0")),
                "/tmp/quantum_double_barrier_atlas.png",
            );
        }
    }

    tool.atlas.select_axis_pair_index(1);
    let saved = tool.save_state().expect("save atlas state");
    let mut restored = WaveSimulatorTool::new();
    restored
        .try_load_state(&saved)
        .expect("restore atlas state");
    assert!(restored.atlas.is_visible());
    assert_eq!(restored.atlas.axis_keys(), ("k0", "width"));
}

#[test]
fn test_subbarrier_50_split_and_double_barrier_challenge_hud_and_url_clamping() {
    let mut fw = SimulationFramework::new("quantum");
    let split_meta =
        find_challenge_metadata(CHALLENGE_SUBBARRIER_50_SPLIT).expect("subbarrier 50 split card");
    fw.launch_challenge_card(&split_meta.to_card())
        .expect("launch subbarrier 50 split");

    let eval_init = fw
        .active_tool
        .as_ref()
        .unwrap()
        .evaluate_challenge()
        .expect("active evaluation");
    assert!(!eval_init.passed, "Challenge must start unsolved");

    {
        let wave = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<WaveSimulatorTool>()
            .unwrap();
        wave.k0 = 5.0;
        wave.v0 = 11.35;
        wave.width = 0.5;
        wave.sync_config_and_reset(None);
        assert!(
            !wave.evaluate_challenge().unwrap().passed,
            "Over-barrier E0=12.50 >= V0=11.35 must be rejected"
        );

        wave.v0 = 21.05;
        wave.width = 0.25;
        wave.k0 = 5.0;
        wave.sync_config_and_reset(None);
        let solved = wave.evaluate_challenge().unwrap();
        assert!(
            solved.passed,
            "Calibrated sub-barrier 50/50 split must pass: {}",
            solved.detail
        );
        assert!(solved.residual_value <= 0.03);
    }

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| fw.show(ctx, "quantum"));

    let db_meta = find_challenge_metadata(CHALLENGE_DOUBLE_BARRIER_RESONANCE)
        .expect("double barrier resonance card");
    let tampered =
        r#"{"preset":"InfiniteWell","x0":1.2,"k0":5.671,"sigma":1.2,"v0":2.0,"width":0.2}"#;
    let clamped = db_meta
        .clamp_state_json(tampered)
        .expect("clamp locked fields");
    let parsed: serde_json::Value = serde_json::from_str(&clamped).unwrap();
    assert_eq!(parsed["preset"], "DoubleBarrier");
    assert_eq!(parsed["v0"], 28.0);
    assert_eq!(parsed["width"], 0.8);
}

#[test]
fn test_all_four_quantum_tab_tools_expose_finite_named_observables() {
    let tools: [Box<dyn InteractiveTool>; 4] = [
        Box::new(WaveSimulatorTool::new()),
        Box::new(VolumetricFieldStudioTool::default()),
        Box::new(SpinVisualizer::default()),
        Box::new(ClebschGordanTool::default()),
    ];
    for tool in &tools {
        let obs = tool.observables();
        assert!(
            !obs.is_empty(),
            "QuantumTab tool '{}' must expose non-empty observables()",
            tool.name()
        );
        for (key, val) in obs {
            assert!(!key.is_empty());
            assert!(
                val.is_finite(),
                "Non-finite observable {key}={val} on {}",
                tool.name()
            );
        }
    }
}
