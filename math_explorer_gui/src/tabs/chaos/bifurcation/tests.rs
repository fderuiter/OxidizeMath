#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_bifurcation_recompute_and_borrowed_slice() {
    let mut tool = BifurcationDiagram::default();
    assert!(!tool.points.is_empty());

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
fn test_bifurcation_animated_slider_sweep() {
    use crate::reflective_ui::slider::AnimatedSliderState;
    let mut tool = BifurcationDiagram::default();
    let ctx = egui::Context::default();
    let initial_r_min = tool.r_min;

    // Simulate activating animation on r_min slider
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

    let raw_input = egui::RawInput {
        predicted_dt: 0.1,
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        tool.show(ctx);
    });

    // r_min should have increased, and recompute was triggered
    assert!(tool.r_min > initial_r_min);

    let points_vec: Vec<[f64; 2]> = tool.points.iter().map(|p| [p.x, p.y]).collect();
    let datasets = vec![("Bifurcation Diagram (Logistic Map)".to_string(), points_vec)];
    if let Ok(bytes) = egui_plot::export::export_custom_png(
        &datasets,
        &egui_plot::export::ExportOptions::default(),
    ) {
        let _ = std::fs::write("/tmp/bifurcation_diagram_animated.png", bytes);
    }
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
