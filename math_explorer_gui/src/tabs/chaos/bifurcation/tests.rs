#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use crate::reflective_ui::AnimatedSliderState;
use crate::web_io::PlatformInstant;
use std::time::Duration;

#[test]
fn test_bifurcation_recompute_and_borrowed_slice() {
    let mut tool = BifurcationDiagram::default();
    assert!(!tool.points.is_empty());
    assert!(!tool.lyapunov_points.is_empty());

    let count_initial = tool.points.len();
    tool.steps = 200;
    tool.recompute();
    assert_ne!(tool.points.len(), count_initial);

    let slice: &[PlotPoint] = &tool.points[..];
    let plot_points: egui_plot::PlotPoints<'_> = slice.into();
    assert!(matches!(plot_points, egui_plot::PlotPoints::Borrowed(_)));
    assert_eq!(plot_points.points().len(), tool.points.len());
}

#[test]
fn test_bifurcation_animated_slider_sweep_under_16_67ms() {
    let mut tool = BifurcationDiagram {
        steps: 2000,
        ..Default::default()
    };
    let ctx = egui::Context::default();
    let initial_r_min = tool.r_min;
    let screen = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1280.0, 800.0),
    ));

    // Warm up layout/fonts on initial frame
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        |c| tool.show(c),
    );

    let id = egui::Id::new("bifurcation_r_min");
    ctx.data_mut(|d| {
        d.insert_temp(
            id,
            AnimatedSliderState {
                playing: true,
                direction: 1.0,
            },
        );
    });

    let t0 = PlatformInstant::now();
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            predicted_dt: 0.1,
            ..Default::default()
        },
        |c| tool.show(c),
    );
    let sweep_elapsed = t0.elapsed();

    assert!(tool.r_min > initial_r_min);
    assert!(tool.request_generation >= 1);
    assert!(
        sweep_elapsed < Duration::from_micros(16_670),
        "AnimatedSlider sweep at steps=2000 stalled UI thread: {sweep_elapsed:?}"
    );

    let datasets = tool.export_datasets();
    if let Ok(bytes) = egui_plot::export::export_custom_png(
        &datasets,
        &egui_plot::export::ExportOptions::default(),
    ) {
        let _ = std::fs::write("/tmp/bifurcation_diagram_animated.png", bytes);
    }
}

#[test]
fn test_bifurcation_runner_progressive_sweep_and_lyapunov_signs() {
    let req = BifurcationSweepRequest {
        generation: 1,
        r_min: 2.5,
        r_max: 4.0,
        steps: 300,
    };
    let shared = Arc::new(RwLock::new(BifurcationSweepRequest {
        generation: 0,
        ..req
    }));
    let latest = Arc::new(AtomicU64::new(0));
    let mut runner = BifurcationRunner::new(
        Arc::clone(&shared),
        Arc::clone(&latest),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    *shared.write().unwrap() = req;
    latest.store(1, Ordering::Release);

    for _ in 0..32 {
        runner.step();
        if runner.is_complete() {
            break;
        }
    }
    assert!(runner.is_complete());
    let snap = runner.get_snapshot();
    let data = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<BifurcationSweepSnapshot>())
        .expect("BifurcationSweepSnapshot expected");
    assert!(data.is_final);
    assert_eq!(data.raw_lyapunov.len(), 301);
    assert!(data.raw_lyapunov.iter().all(|pair| pair[1].is_finite()));

    let mut tool = BifurcationDiagram::default();
    tool.select_r(2.8);
    assert!(
        tool.selected_lyapunov() < 0.0,
        "Fixed-point regime r=2.8 must have negative Lyapunov exponent"
    );
    tool.select_r(3.83);
    assert!(
        tool.selected_lyapunov() < 0.0,
        "Period-3 window r=3.83 must have negative Lyapunov exponent"
    );
    tool.select_r(3.95);
    assert!(
        tool.selected_lyapunov() > 0.0,
        "Chaotic band r=3.95 must have positive Lyapunov exponent"
    );
}

#[test]
fn test_bifurcation_cobweb_orbit_and_backward_compatible_state() {
    let series = build_cobweb_and_orbit(2.8, 0.2, 40);
    assert_eq!(series.orbit.len(), 41);
    assert_eq!(series.cobweb.len(), 81);
    let tail = series.orbit.last().unwrap()[1];
    assert!(
        (tail - (1.0 - 1.0 / 2.8)).abs() < 1e-3,
        "r=2.8 orbit should converge near x* = 1 - 1/r, got {tail}"
    );

    let legacy_json = r#"{"r_min":1.0,"r_max":3.6,"steps":200}"#;
    let mut tool = BifurcationDiagram::default();
    tool.try_load_state(legacy_json)
        .expect("legacy JSON without selected_r must load");
    assert_eq!(tool.steps, 200);
    assert!((tool.r_min..=tool.r_max).contains(&tool.selected_r));

    tool.select_r(3.2);
    let saved = tool.save_state().expect("save_state must succeed");
    let mut restored = BifurcationDiagram::default();
    restored.try_load_state(&saved).expect("roundtrip load");
    assert!((restored.selected_r - 3.2).abs() < 1e-9);
}

#[test]
fn test_bifurcation_custom_1d_map_zero_alloc_and_state_roundtrip() {
    let mut tool = BifurcationDiagram {
        map_mode: BifurcationMapMode::Custom,
        r_min: 0.5,
        r_max: 2.0,
        steps: 240,
        x0: 0.25,
        ..Default::default()
    };
    tool.math_input.set_expression("a * sin(pi * x) * r / 2");
    let mut pmap = std::collections::BTreeMap::new();
    pmap.insert("a".to_string(), 1.8);
    tool.math_input.set_params_map(&pmap);
    tool.recompute();
    assert!(!tool.points.is_empty());
    assert!(tool
        .points
        .iter()
        .all(|pt| pt.x.is_finite() && pt.y.is_finite()));

    // Verify transient syntax error keeps last valid cascade points
    let prev_len = tool.points.len();
    tool.math_input.set_expression("a * sin(pi * x) +");
    tool.recompute();
    assert_eq!(tool.points.len(), prev_len);

    // Restore valid quadratic map x^2 - r on [0.0, 2.0] and verify zero heap allocations
    tool.math_input.set_expression("x^2 - r");
    tool.r_min = 0.0;
    tool.r_max = 1.9;
    tool.x0 = 0.1;
    tool.points.reserve(tool.steps * CUSTOM_ORBIT_SAMPLES);
    tool.recompute();
    assert!(!tool.points.is_empty());

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let _ = verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.recompute();
    });

    // Verify compact JSON / #state= persistence round-trip (< 4096 bytes)
    tool.math_input.set_expression("a * x * (1 - x) * r");
    tool.math_input.set_params_map(&pmap);
    let saved = tool.save_state().expect("custom bifurcation state saves");
    assert!(saved.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);
    let mut restored = BifurcationDiagram::default();
    restored
        .try_load_state(&saved)
        .expect("custom bifurcation state loads");
    assert_eq!(restored.map_mode, BifurcationMapMode::Custom);
    assert_eq!(restored.math_input.raw_text, "a * x * (1 - x) * r");
    assert_eq!(restored.math_input.params_map(), pmap);
    assert!((restored.r_min - 0.0).abs() < 1e-12);
    assert!((restored.r_max - 1.9).abs() < 1e-12);
    assert!((restored.x0 - 0.1).abs() < 1e-12);
}
