#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{evaluate_potential_grid, POTENTIAL_CLAMP_MAX, POTENTIAL_CLAMP_MIN};
use super::*;
use crate::async_sim::SimulationRunner;
use crate::framework::{InteractiveTool, SimulationFramework};
use eframe::egui;
use math_commons::ast::parse_equation;
use math_explorer::physics::quantum::evolve_state;
use std::sync::{Arc, RwLock};

#[test]
fn test_legacy_starter_json_and_custom_url_state_roundtrip() {
    let mut tool = WaveSimulatorTool::new();
    tool.try_load_state(r#"{"potential_type": 1.0, "dt": 0.02}"#)
        .expect("legacy WaveUnified JSON must deserialize cleanly");
    assert_eq!(tool.preset, WavePotentialPreset::HarmonicOscillator);
    assert!((tool.dt - 0.02).abs() < 1e-12);

    tool.import_preset_json(STARTER_PRESET_JSON)
        .expect("StarterPresetMetadata JSON must import");
    assert_eq!(tool.preset, WavePotentialPreset::HarmonicOscillator);

    let mut fw = SimulationFramework::new("quantum");
    assert!(fw.select_tool_by_name("WaveSimulator"));
    {
        let wave = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<WaveSimulatorTool>()
            .unwrap();
        wave.preset = WavePotentialPreset::Custom;
        wave.x0 = -2.25;
        wave.k0 = 5.5;
        wave.sigma = 0.45;
        wave.v0 = 14.0;
        wave.width = 0.6;
        wave.dt = 0.015;
        wave.math_input.set_expression("v0 * exp(-(x / width)^4)");
        wave.last_valid_expr = wave.math_input.expr().unwrap().clone();
        wave.sync_config_and_reset(None);
    }
    let expected_state = fw.active_tool.as_ref().unwrap().save_state();
    let share_url = fw.encode_share_url().expect("encode share URL");
    assert!(share_url.contains("#tab=quantum&tool=WaveSimulator&state="));

    let fragment = format!("#{}", share_url.split_once('#').unwrap().1);
    let params = crate::url_state::parse_share_location(&fragment, "")
        .expect("decode fragment")
        .expect("parsed share params");
    let mut restored_fw = SimulationFramework::new("quantum");
    restored_fw
        .import_url_preset_json(params.tool.as_deref(), params.state_json.as_deref())
        .expect("import shared URL preset");
    assert_eq!(
        restored_fw.active_tool.as_ref().unwrap().save_state(),
        expected_state
    );

    let oversized = format!(
        r#"{{"preset":"Custom","custom_expr":"{}"}}"#,
        "x+".repeat(300)
    );
    assert!(tool.try_load_state(&oversized).is_err());
}

#[test]
fn test_wave_simulator_sonification_and_audio_roundtrip() {
    let mut tool = WaveSimulatorTool::new();
    tool.preset = WavePotentialPreset::Custom;
    tool.math_input.set_expression("v0 * cos(x)^2");
    tool.last_valid_expr = tool.math_input.expr().unwrap().clone();
    tool.set_acoustic_settings(crate::audio::AcousticSettings {
        gain: 0.68,
        pitch_scale: 1.35,
    });
    tool.sync_config_and_reset(None);
    tool.step_simulation();

    let live_frame = tool.sonification_frame().expect("sonification frame");
    assert!(live_frame.is_audible());
    assert!(live_frame.custom_harmonics.is_some());

    let saved = tool.save_state().expect("save wave simulator state");
    assert!(saved.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);
    let mut restored = WaveSimulatorTool::new();
    restored
        .try_load_state(&saved)
        .expect("load wave simulator state");
    assert_eq!(restored.preset, WavePotentialPreset::Custom);
    assert!((restored.acoustic_settings().gain - 0.68).abs() < 1e-5);
    assert!((restored.acoustic_settings().pitch_scale - 1.35).abs() < 1e-5);
}

#[test]
fn test_rectangular_barrier_tunneling_benchmarks_and_pre_wall_latch() {
    for (v0, target_t) in [(11.35, 0.50), (15.0, 0.32)] {
        let cfg = WaveSimConfig {
            preset: WavePotentialPreset::RectangularBarrier,
            x0: -2.0,
            k0: 5.0,
            sigma: 0.5,
            v0,
            width: 0.5,
            dt: 0.02,
            steps_per_frame: 5,
            custom_expr: presets::default_compiled_expr(),
        };
        let mut runner = WaveRunner::new(Arc::new(RwLock::new(cfg)));
        assert!(runner.latched_transmission().is_none());

        // Step through packet transit until T_scatter latches pre-wall (around t ~ 0.82)
        let mut pre_wall_checked = false;
        for _ in 0..50 {
            let prev_rw = runner.build_wave_data().right_wall_prob;
            runner.step();
            if !pre_wall_checked && runner.latched_transmission().is_some() {
                assert!(prev_rw < 0.02, "Latch must occur from a pre-wall state");
                pre_wall_checked = true;
            }
        }
        let latched_pre_wall = runner
            .latched_transmission()
            .expect("T_scatter must latch after barrier transit and before wall arrival");
        assert!(
            (latched_pre_wall - target_t).abs() <= 0.02,
            "Expected T_scatter within {target_t:.2} +/- 0.02 for V0={v0}, got {latched_pre_wall:.6}"
        );
        let snap_pre = runner.build_wave_data();
        assert!(
            (snap_pre.reflection + snap_pre.barrier_prob + snap_pre.transmission - 1.0).abs()
                < 1e-9
        );

        // Continue past right-wall bounce (t = 2.60) and verify latched value stays frozen
        for _ in 0..80 {
            runner.step();
        }
        assert_eq!(runner.latched_transmission(), Some(latched_pre_wall));
    }
}

#[test]
fn test_ast_syntax_aria_announcement_and_singular_potential_clamping() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    crate::accessibility::clear_test_announcements();

    let mut tool = WaveSimulatorTool::new();
    tool.preset = WavePotentialPreset::Custom;
    tool.sync_config_and_reset(None);
    let baseline_expr = tool.last_valid_expr.clone();

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // Incomplete syntax triggers assertive ARIA announcement while preserving last_valid_expr
    tool.math_input.set_expression("1 +");
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
    let announcements = crate::accessibility::all_announcements();
    assert!(
        announcements
            .iter()
            .any(|(msg, prio)| prio == "assertive" && msg.starts_with("Syntax Error:")),
        "Expected assertive ARIA syntax error announcement, got {announcements:?}"
    );
    assert_eq!(tool.last_valid_expr, baseline_expr);

    // Singular 1 / x and scaled pole 5000 / x clamp cleanly to [-500.0, 500.0]
    let (_, pole_expr) = parse_equation("1 / x").expect("parse 1 / x");
    let mut dense_grid: Vec<f64> = (0..200).map(|i| -5.0 + i as f64 * (10.0 / 199.0)).collect();
    dense_grid.extend_from_slice(&[-1e-4, 0.0, 1e-4]);
    let clamped = evaluate_potential_grid(
        WavePotentialPreset::Custom,
        &dense_grid,
        20.0,
        0.5,
        0.05,
        &pole_expr,
    );
    assert!(clamped
        .iter()
        .all(|&v| v.is_finite() && (POTENTIAL_CLAMP_MIN..=POTENTIAL_CLAMP_MAX).contains(&v)));
    assert!((clamped[dense_grid.len() - 3] - (-500.0)).abs() < 1e-12);
    assert!((clamped[dense_grid.len() - 2] - 0.0).abs() < 1e-12);
    assert!((clamped[dense_grid.len() - 1] - 500.0).abs() < 1e-12);

    let cfg = WaveSimConfig {
        preset: WavePotentialPreset::Custom,
        custom_expr: pole_expr,
        ..Default::default()
    };
    let mut runner = WaveRunner::new(Arc::new(RwLock::new(cfg)));
    runner.step();
    assert!(runner.is_finite_state());
}

#[test]
fn test_sample_animation_frames_monotonic_time_and_distinct_wavepackets() {
    let mut tool = WaveSimulatorTool::new();
    tool.apply_tunneling_split_challenge(None);

    let frames = tool.sample_animation_frames(10);
    assert_eq!(frames.len(), 10);

    let mut prev_time = 0.0;
    let mut prev_psi: Option<Vec<[f64; 2]>> = None;
    for frame in &frames {
        assert!(frame.len() >= 3);
        let (label, psi_pts) = &frame[0];
        let t_str = label
            .split("t=")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .expect("series label must encode simulation time t");
        let t_val: f64 = t_str.parse().expect("valid float time");
        assert!(
            t_val > prev_time,
            "Expected strictly increasing frame time: {t_val} <= {prev_time}"
        );
        prev_time = t_val;
        if let Some(ref prev) = prev_psi {
            assert_ne!(prev, psi_pts, "Consecutive wavepacket frames must differ");
        }
        prev_psi = Some(psi_pts.clone());
    }
}

#[test]
fn test_zero_allocation_hot_path_propagator_cache_and_state_guard_rollback() {
    let cfg = Arc::new(RwLock::new(WaveSimConfig {
        preset: WavePotentialPreset::DoubleBarrier,
        dt: 0.02,
        ..Default::default()
    }));
    let mut runner = WaveRunner::new(Arc::clone(&cfg));
    assert_eq!(runner.cached_dt, None);

    // Warm propagator cache on step 1
    let init_psi = runner.psi.clone();
    runner.step();
    assert_eq!(runner.cached_dt, Some(0.02));
    let ptr_before = runner.propagator.matrix.as_ptr();

    // Verify steps 2..=6 run with zero heap allocations under VerifiedAllocator
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..5 {
                runner.step();
            }
        });
    }
    assert_eq!(runner.propagator.matrix.as_ptr(), ptr_before);

    let expected_psi = evolve_state(&init_psi, &runner.hamiltonian, 0.12, 1.0).normalize();
    assert!((&runner.psi.vector - &expected_psi.vector).norm() < 1e-11);

    // Verify SimulationFramework state_guard rollback and parameter clamping
    let mut fw = SimulationFramework::new("quantum");
    assert!(fw.select_tool_by_name("WaveSimulator"));
    fw.cache_valid_snapshot();
    let baseline = fw.active_tool.as_ref().unwrap().save_state();

    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&[(
        "Probability |ψ|²".to_string(),
        vec![[0.0, f64::NAN]],
    )]);
    assert!(fw.inspect_runtime_state().is_err());
    assert!(fw.is_diverged());
    assert!(fw.rollback_step_back());
    assert!(!fw.is_diverged());
    assert_eq!(fw.active_tool.as_ref().unwrap().save_state(), baseline);
}
