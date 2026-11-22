#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, TraceSeries};
use scientific_metadata::theory::TheoryDescribable;

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
    let mut tool = OdeSolverTool { param_k: 1.0, ..Default::default() };
    tool.recalculate();

    // Pin baseline trajectory k=1.0
    let summary = tool.param_summary();
    let pts: Vec<[f64; 2]> = tool.time_series.iter().zip(&tool.y_series).map(|(&t, &y)| [t, y]).collect();
    tool.snapshot_manager.add_snapshot("Baseline 1", summary, vec![TraceSeries::new("y(t)", pts.clone())]);
    assert_eq!(tool.snapshot_manager.len(), 1);

    // Change parameter k=2.5 and recalculate active simulation
    tool.param_k = 2.5;
    tool.recalculate();

    // Pinned snapshot retains baseline curve points
    assert_eq!(tool.snapshot_manager.snapshots()[0].param_summary, r"k=1.00, \Delta t=0.05");
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
        let pts: Vec<[f64; 2]> = tool.time_series.iter().zip(&tool.y_series).map(|(&t, &y)| [t, y]).collect();
        tool.snapshot_manager.push_history_run(old_summary, vec![TraceSeries::new("y(t)", pts)]);
        tool.param_k = i as f64 * 0.5;
        tool.recalculate();
    }

    assert_eq!(tool.snapshot_manager.history_len(), 5);
    assert!(tool.snapshot_manager.history_runs()[0].param_summary.contains("k=1.00"));
    assert!(tool.snapshot_manager.history_runs()[4].param_summary.contains("k=3.00"));

    let exported = tool.export_datasets();
    assert!(exported.iter().any(|(name, _)| name.contains("Run")));
}

#[test]
fn test_ode_auto_callout_annotations_and_persistence() {
    let mut tool = OdeSolverTool::default();
    assert_eq!(tool.callout_manager.len(), 2);
    let initial_callout = tool
        .callout_manager
        .callouts()
        .iter()
        .find(|c| c.tag.as_deref() == Some("initial_condition"))
        .expect("initial callout");
    assert_eq!(initial_callout.target_point, [0.0, 1.0]);
    assert!(initial_callout.text.contains("Initial"));

    let final_callout = tool
        .callout_manager
        .callouts()
        .iter()
        .find(|c| c.tag.as_deref() == Some("final_state"))
        .expect("final callout");
    assert_eq!(final_callout.target_point[0], 10.0);
    assert!(final_callout.text.contains("Final"));

    // Add a custom user callout
    tool.callout_manager
        .add_callout([5.0, 2.0], "Equilibrium Y=2.0", [15.0, -20.0]);
    assert_eq!(tool.callout_manager.len(), 3);

    // Save and load state
    let state = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&state);
    assert_eq!(restored.callout_manager.len(), 3);
    assert!(restored
        .callout_manager
        .callouts()
        .iter()
        .any(|c| c.text == "Equilibrium Y=2.0"));
}

#[test]
fn test_ode_custom_equation_rk4() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom, param_k: 1.5, ic_y0: 2.0, dt: 0.05, total_time: 5.0, ..Default::default()
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
    let mut tool = OdeSolverTool { preset: OdePreset::Custom, ic_y0: 1.0, ..Default::default() };
    tool.math_input.set_expression("1 / (y - 1)");
    tool.recalculate();
    assert!(tool.diverged());
}

#[test]
fn test_ode_custom_state_roundtrip() {
    let mut tool = OdeSolverTool { preset: OdePreset::Custom, ..Default::default() };
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
fn test_ode_deep_custom_expr_preset_and_state_rejected() {
    let deep_expr = format!("{}1", "1+".repeat(100_000));
    let preset_payload = format!(
        r#"{{"metadata":{{"schema_version":"1.0","title":"Deep","domain":"analysis","timestamp":"2026-11-18T00:00:00Z"}},"config":{{"preset":"Custom","dt":0.05,"total_time":0.1,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"custom_expr":"{deep_expr}"}}}}"#
    );
    let mut tool = OdeSolverTool::default();
    let _ = tool.import_preset_json(&preset_payload);
    assert!(!tool.math_input.is_valid());
    let mut restored = OdeSolverTool::default();
    restored.load_state(&tool.save_state().expect("save state"));
    assert!(!restored.math_input.is_valid());
}

#[test]
fn test_generate_custom_ode_and_complex_screenshots() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom, param_k: 2.0, ic_y0: 1.0, dt: 0.02, total_time: 10.0, ..Default::default()
    };
    tool.math_input.set_expression("y' = -k * y * sin(t)");
    tool.recalculate();

    let datasets = tool.export_datasets();
    let options = egui_plot::export::ExportOptions::default();
    let ode_png = egui_plot::export::export_custom_png(&datasets, &options);
    assert!(ode_png.is_ok());
    if let Ok(png) = ode_png {
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

    let complex_png = egui_plot::export::export_custom_png(&complex_datasets, &options);
    assert!(complex_png.is_ok());
    if let Ok(png) = complex_png {
        let _ = std::fs::write("/tmp/ast_expression_parser_complex.png", png);
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

#[test]
fn test_ode_presets_theory_metadata() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Exponential,
        ..Default::default()
    };
    assert_eq!(tool.theory_latex_equation(), Some(r"\frac{dy}{dt} = k y".into()));
    assert!(tool.theory_citation().contains("Perko"));
    assert!(!tool.theory_description().is_empty());

    tool.preset = OdePreset::HarmonicOscillator;
    assert_eq!(tool.theory_latex_equation(), Some(r"\frac{d^2 y}{dt^2} = -k y".into()));
    assert!(tool.theory_citation().contains("Goldstein"));
    assert!(!tool.theory_description().is_empty());

    tool.preset = OdePreset::LogisticGrowth;
    assert_eq!(tool.theory_latex_equation(), Some(r"\frac{dy}{dt} = r y (1 - \frac{y}{K})".into()));
    assert!(tool.theory_citation().contains("Lotka") || tool.theory_citation().contains("Verhulst"));
    assert!(!tool.theory_description().is_empty());
}

#[test]
fn test_harmonic_equilibrium_root_dt_025_and_nonlinear_preset_fits() {
    let mut osc = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1.0,
        ic_y0: 1.0,
        ic_v0: 0.0,
        dt: 0.25,
        total_time: 10.0,
        ..Default::default()
    };
    osc.recalculate();
    let root_t = annotations::harmonic_equilibrium_root(&osc).expect("harmonic root");
    assert!((root_t - 5.0).abs() > 0.1, "must not anchor at arbitrary midpoint t=5.0");
    let nearest_analytical = 1.5 * std::f64::consts::PI;
    assert!(
        (root_t - nearest_analytical).abs() < 1e-3,
        "dt=0.25 RK4 + Hermite root error = {:.6e}s",
        (root_t - nearest_analytical).abs()
    );

    let osc_pts = &osc.export_datasets()[0].1;
    let sin_fit = egui_plot::fit_model(osc_pts, egui_plot::FitType::Sinusoidal, "y(t)", false)
        .expect("sinusoidal fit");
    assert!(sin_fit.r_squared >= 0.99);

    let mut log_tool = OdeSolverTool {
        preset: OdePreset::LogisticGrowth,
        param_r: 0.8,
        param_cap_k: 12.0,
        ic_y0: 0.5,
        dt: 0.1,
        total_time: 12.0,
        ..Default::default()
    };
    log_tool.recalculate();
    let log_pts = &log_tool.export_datasets()[0].1;
    let logistic_fit =
        egui_plot::fit_model(log_pts, egui_plot::FitType::Logistic, "y(t)", false)
            .expect("logistic fit");
    assert!(logistic_fit.r_squared >= 0.99);
}


