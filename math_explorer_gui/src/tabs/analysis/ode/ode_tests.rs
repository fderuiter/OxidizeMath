#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, TraceSeries};

#[test]
fn test_ode_divergence_detection() {
    let mut tool = OdeSolverTool::default();
    assert!(!tool.diverged());

    // Set exponential growth rate and initial value that causes f64 overflow/divergence
    tool.param_k = 1e300;
    tool.ic_y0 = 1e300;
    tool.recalculate();

    assert!(tool.diverged());
    // Verify time_series and y_series do not contain non-finite values
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
    assert!(tool.time_series.iter().all(|t| t.is_finite()));

    // Reset parameters recovers from divergence
    tool.reset_parameters();
    assert!(!tool.diverged());
    assert!(!tool.y_series.is_empty());
}

#[test]
fn test_ode_nan_initial_condition() {
    let mut tool = OdeSolverTool {
        ic_y0: f64::NAN,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.is_empty());
}

#[test]
fn test_ode_harmonic_oscillator_divergence() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
    assert!(tool.v_series.iter().all(|v| v.is_finite()));
}

#[test]
fn test_ode_logistic_growth_divergence() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::LogisticGrowth,
        param_r: 1e300,
        param_cap_k: 1.0,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
}

#[test]
fn test_ode_trace_snapshot_pinning_and_export() {
    let mut tool = OdeSolverTool {
        param_k: 1.0,
        ..Default::default()
    };
    tool.recalculate();

    // Pin baseline trajectory k=1.0
    let summary = tool.param_summary();
    let pts: Vec<[f64; 2]> = tool
        .time_series
        .iter()
        .zip(&tool.y_series)
        .map(|(&t, &y)| [t, y])
        .collect();
    tool.snapshot_manager.add_snapshot(
        "Baseline 1",
        summary,
        vec![TraceSeries::new("y(t)", pts.clone())],
    );

    assert_eq!(tool.snapshot_manager.len(), 1);

    // Change parameter k=2.5 and recalculate active simulation
    tool.param_k = 2.5;
    tool.recalculate();

    // Pinned snapshot retains baseline curve points
    assert_eq!(
        tool.snapshot_manager.snapshots()[0].param_summary,
        r"k=1.00, \Delta t=0.05"
    );
    assert_eq!(tool.snapshot_manager.snapshots()[0].series[0].points, pts);

    // Export datasets includes active + pinned traces
    let exported = tool.export_datasets();
    assert!(exported.iter().any(|(name, _)| name.contains("Baseline 1")));

    // State serialization roundtrip preserves snapshot
    let state = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&state);
    assert_eq!(restored.snapshot_manager.len(), 1);
}

#[test]
fn test_ode_auto_history_buffering() {
    let mut tool = OdeSolverTool::default();
    assert_eq!(tool.snapshot_manager.history_len(), 0);

    for i in 1..=7 {
        let old_summary = tool.param_summary();
        let pts: Vec<[f64; 2]> = tool
            .time_series
            .iter()
            .zip(&tool.y_series)
            .map(|(&t, &y)| [t, y])
            .collect();
        tool.snapshot_manager
            .push_history_run(old_summary, vec![TraceSeries::new("y(t)", pts)]);
        tool.param_k = i as f64 * 0.5;
        tool.recalculate();
    }

    assert_eq!(tool.snapshot_manager.history_len(), 5);
    assert!(tool.snapshot_manager.history_runs()[0]
        .param_summary
        .contains("k=1.00"));
    assert!(tool.snapshot_manager.history_runs()[4]
        .param_summary
        .contains("k=3.00"));

    let exported = tool.export_datasets();
    assert!(exported.iter().any(|(name, _)| name.contains("Run")));
}

#[test]
fn test_ode_custom_equation_rk4() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        param_k: 1.5,
        ic_y0: 2.0,
        dt: 0.05,
        total_time: 5.0,
        ..Default::default()
    };
    tool.math_input.set_expression("-k * y * sin(t)");
    tool.recalculate();

    assert!(!tool.diverged());
    assert!(!tool.y_series.is_empty());
    assert_eq!(tool.time_series.len(), tool.y_series.len());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));

    // Verify changing k produces different trajectory
    let original_final_y = *tool.y_series.last().unwrap();
    tool.param_k = 3.0;
    tool.recalculate();
    let new_final_y = *tool.y_series.last().unwrap();
    assert_ne!(original_final_y, new_final_y);
}

#[test]
fn test_ode_custom_equation_divergence_handling() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ic_y0: 1.0,
        ..Default::default()
    };
    tool.math_input.set_expression("1 / (y - 1)");
    tool.recalculate();

    assert!(tool.diverged());
}

#[test]
fn test_ode_custom_state_roundtrip() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("y' = -k * y * sin(t)");
    tool.recalculate();

    let state = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&state);

    assert_eq!(restored.preset, OdePreset::Custom);
    assert_eq!(restored.math_input.raw_text, "y' = -k * y * sin(t)");
    assert!(!restored.diverged());
}

#[test]
fn test_generate_custom_ode_and_complex_screenshots() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        param_k: 2.0,
        ic_y0: 1.0,
        dt: 0.02,
        total_time: 10.0,
        ..Default::default()
    };
    tool.math_input.set_expression("y' = -k * y * sin(t)");
    tool.recalculate();

    let datasets = tool.export_datasets();
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/ast_expression_parser_ode.png", png);
    }

    // Complex conformal mapping datasets: w = z^3 - 1/z
    let expr = math_commons::ast::parse("z^3 - 1/z").expect("parse complex");
    let mut complex_datasets = Vec::new();
    let grid_density = 10;
    let grid_min = -2.0;
    let grid_max = 2.0;
    let step = (grid_max - grid_min) / (grid_density as f64 - 1.0);
    let resolution = 40;

    for i in 0..grid_density {
        let x = grid_min + (i as f64) * step;
        let mut pts = Vec::with_capacity(resolution);
        for j in 0..resolution {
            let y = grid_min + (grid_max - grid_min) * (j as f64) / (resolution as f64 - 1.0);
            let z = num_complex::Complex64::new(x, y);
            let w = expr.eval_complex_z(z);
            if w.re.is_finite() && w.im.is_finite() && w.norm() < 20.0 {
                pts.push([w.re, w.im]);
            }
        }
        if !pts.is_empty() {
            complex_datasets.push((format!("Re(z)={x:.2}"), pts));
        }
    }

    if let Ok(png) = egui_plot::export::export_custom_png(&complex_datasets, &options) {
        let _ = std::fs::write("/tmp/ast_expression_parser_complex.png", png);
    }

    assert!(std::path::Path::new("/tmp/ast_expression_parser_ode.png").exists());
    assert!(std::path::Path::new("/tmp/ast_expression_parser_complex.png").exists());
}

#[test]
fn test_ode_solver_method_selection_and_accuracy_ordering() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Exponential,
        param_k: 1.0,
        ic_y0: 1.0,
        dt: 0.1,
        total_time: 1.0,
        ..Default::default()
    };

    // 1. RungeKutta4
    tool.set_method(IntegrationMethod::RungeKutta4);
    assert_eq!(tool.method(), IntegrationMethod::RungeKutta4);
    let rk4_final = *tool.y_series.last().expect("rk4 final");

    // 2. Heun
    tool.set_method(IntegrationMethod::Heun);
    assert_eq!(tool.method(), IntegrationMethod::Heun);
    let heun_final = *tool.y_series.last().expect("heun final");

    // 3. Euler
    tool.set_method(IntegrationMethod::Euler);
    assert_eq!(tool.method(), IntegrationMethod::Euler);
    let euler_final = *tool.y_series.last().expect("euler final");

    // Analytical solution: e^1 ≈ 2.718281828
    let true_val = 1.0_f64.exp();
    let rk4_err = (rk4_final - true_val).abs();
    let heun_err = (heun_final - true_val).abs();
    let euler_err = (euler_final - true_val).abs();

    // Order of accuracy: RK4 < Heun < Euler
    assert!(
        rk4_err < heun_err,
        "RK4 err {rk4_err} should be < Heun err {heun_err}"
    );
    assert!(
        heun_err < euler_err,
        "Heun err {heun_err} should be < Euler err {euler_err}"
    );
}

#[test]
fn test_ode_multi_solver_comparison_mode_overlays() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1.0,
        ic_y0: 1.0,
        ic_v0: 0.0,
        dt: 0.05,
        total_time: 5.0,
        ..Default::default()
    };

    assert!(!tool.comparison_mode());
    assert_eq!(tool.comparison_manager().len(), 0);

    // Enable multi-solver comparison mode
    tool.set_comparison_mode(true);
    assert!(tool.comparison_mode());

    // Three solvers should be present in comparison manager
    let mgr = tool.comparison_manager();
    assert_eq!(mgr.len(), 3);

    let labels: Vec<String> = mgr.snapshots().iter().map(|s| s.label.clone()).collect();
    assert!(labels.contains(&"Runge-Kutta 4".to_string()));
    assert!(labels.contains(&"Euler".to_string()));
    assert!(labels.contains(&"Heun (RK2)".to_string()));

    // For harmonic oscillator, both y(t) and y'(t) are present in each snapshot
    for snapshot in mgr.snapshots() {
        assert_eq!(snapshot.series.len(), 2);
        assert_eq!(snapshot.series[0].name, "y(t)");
        assert_eq!(snapshot.series[1].name, "y'(t)");
        assert!(!snapshot.series[0].points.is_empty());
    }

    // Export includes all solver trajectories
    let exported = tool.export_datasets();
    assert!(exported.iter().any(|(name, _)| name.contains("Euler")));
    assert!(exported.iter().any(|(name, _)| name.contains("Heun")));
    assert!(exported.iter().any(|(name, _)| name.contains("Runge-Kutta 4")));

    // State persistence roundtrip preserves comparison mode and active method
    tool.set_method(IntegrationMethod::Heun);
    let state = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&state);
    assert!(restored.comparison_mode());
    assert_eq!(restored.method(), IntegrationMethod::Heun);
    assert_eq!(restored.comparison_manager().len(), 3);
}

#[test]
fn test_ode_discrepancy_metrics_calculation() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Exponential,
        param_k: 1.0,
        ic_y0: 1.0,
        dt: 0.1,
        total_time: 2.0,
        ..Default::default()
    };

    tool.set_comparison_mode(true);

    let euler_rk4 = tool.discrepancy_euler_rk4();
    let heun_rk4 = tool.discrepancy_heun_rk4();
    let euler_heun = tool.discrepancy_euler_heun();

    // Verify positive non-zero error discrepancies
    assert!(euler_rk4.max_error > 0.0);
    assert!(euler_rk4.rms_error > 0.0);
    assert!(heun_rk4.max_error > 0.0);
    assert!(heun_rk4.rms_error > 0.0);
    assert!(euler_heun.max_error > 0.0);
    assert!(euler_heun.rms_error > 0.0);

    // Euler error vs RK4 should exceed Heun error vs RK4
    assert!(
        euler_rk4.max_error > heun_rk4.max_error,
        "Euler max error should be greater than Heun max error"
    );
    assert!(
        euler_rk4.rms_error > heun_rk4.rms_error,
        "Euler RMS error should be greater than Heun RMS error"
    );
}

#[test]
fn test_ode_multi_solver_divergence_warning() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1.0,
        ic_y0: 1.0,
        ic_v0: 0.0,
        dt: 0.05,
        total_time: 10.0,
        ..Default::default()
    };
    tool.set_comparison_mode(true);
    assert!(!tool.diverged());

    // Large time step causes lower-order Euler to diverge or produce non-finite values
    tool.dt = 2.5;
    tool.param_k = 1e150;
    tool.ic_y0 = 1e150;
    tool.recalculate();

    assert!(tool.diverged());
    // Ensure all stored points remain finite
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
    for snapshot in tool.comparison_manager().snapshots() {
        for series in &snapshot.series {
            assert!(series
                .points
                .iter()
                .all(|pt| pt[0].is_finite() && pt[1].is_finite()));
        }
    }
}

#[test]
fn test_ode_annotations_toggle_and_rendering() {
    let mut tool = OdeSolverTool::default();
    assert!(tool.show_annotations);

    // Toggle off and on
    tool.show_annotations = false;
    assert!(!tool.show_annotations);
    tool.show_annotations = true;

    // Test serialization roundtrip with show_annotations
    tool.show_annotations = false;
    let saved = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&saved);
    assert!(!restored.show_annotations);

    // Test rendering with each preset
    let ctx = egui::Context::default();

    // 1. Exponential
    tool.show_annotations = true;
    tool.preset = OdePreset::Exponential;
    tool.param_k = -0.5;
    tool.recalculate();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // 2. Harmonic Oscillator
    tool.preset = OdePreset::HarmonicOscillator;
    tool.recalculate();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // 3. Logistic Growth
    tool.preset = OdePreset::LogisticGrowth;
    tool.param_cap_k = 12.0;
    tool.ic_y0 = 2.0;
    tool.recalculate();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // Disable annotations and ensure clean rendering
    tool.show_annotations = false;
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // Export plot screenshot for verification
    let datasets = tool.export_datasets();
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/ode_solver_trajectory.png", png_data);
    }
}
