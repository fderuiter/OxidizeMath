#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

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
    assert!(exported
        .iter()
        .any(|(name, _)| name.contains("Runge-Kutta 4")));

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
fn test_symplectic_solvers_preserve_harmonic_oscillator_amplitude() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1.0,
        ic_y0: 1.0,
        ic_v0: 0.0,
        dt: 0.25,
        total_time: 100.0,
        ..Default::default()
    };

    tool.set_method(IntegrationMethod::RungeKutta4);
    let rk4_amp = tool
        .y_series
        .last()
        .unwrap()
        .hypot(*tool.v_series.last().unwrap());

    tool.set_comparison_mode(true);
    tool.set_method(IntegrationMethod::ImplicitMidpoint);
    let mid_amp = tool
        .y_series
        .last()
        .unwrap()
        .hypot(*tool.v_series.last().unwrap());
    assert_eq!(tool.comparison_manager().len(), 4);

    tool.set_method(IntegrationMethod::GaussLegendre4);
    let gl4_amp = tool
        .y_series
        .last()
        .unwrap()
        .hypot(*tool.v_series.last().unwrap());
    assert_eq!(tool.comparison_manager().len(), 4);

    assert!(
        (mid_amp - 1.0).abs() < 1e-7,
        "ImplicitMidpoint must preserve unit amplitude (<1e-7 drift), got {mid_amp}"
    );
    assert!(
        (gl4_amp - 1.0).abs() < 1e-7,
        "GaussLegendre4 must preserve unit amplitude (<1e-7 drift), got {gl4_amp}"
    );
    assert!(
        (1.0 - rk4_amp) > 5e-4,
        "Explicit RK4 at dt=0.25 over t=100 should damp amplitude (>5e-4 loss), got {rk4_amp}"
    );
}

fn reference_oscillator_trajectory(method: IntegrationMethod) -> comparison::SolverTrajectory {
    comparison::simulate_preset(
        OdePreset::HarmonicOscillator,
        method,
        0.02,
        4.0,
        2.5,
        1.0,
        10.0,
        1.2,
        -0.4,
        None,
    )
}

fn assert_snapshot_matches_reference(
    snap: &crate::framework::TraceSnapshot,
    expected: &comparison::SolverTrajectory,
) {
    assert_eq!(snap.series[0].points.len(), expected.y_series.len());
    assert_eq!(snap.series[1].points.len(), expected.v_series.len());
    let last_pt = snap.series[0].points.last().expect("last y point");
    let last_ref = *expected.y_series.last().expect("expected final y");
    assert_eq!(last_pt[1].to_bits(), last_ref.to_bits());
}

#[test]
fn test_ode_comparison_reuses_active_trajectory_with_exact_parity_across_methods() {
    use crate::web_io::PlatformInstant;
    use comparison::{compute_discrepancy, compute_series_discrepancy};

    let rk4_ref = reference_oscillator_trajectory(IntegrationMethod::RungeKutta4);
    let euler_ref = reference_oscillator_trajectory(IntegrationMethod::Euler);
    let heun_ref = reference_oscillator_trajectory(IntegrationMethod::Heun);
    assert_eq!(
        compute_discrepancy(&euler_ref, &rk4_ref),
        compute_series_discrepancy(
            &euler_ref.y_series,
            &euler_ref.v_series,
            &rk4_ref.y_series,
            &rk4_ref.v_series,
        )
    );

    let start = PlatformInstant::now();
    for active_method in IntegrationMethod::ALL {
        let mut tool = OdeSolverTool {
            preset: OdePreset::HarmonicOscillator,
            method: active_method,
            param_k: 2.5,
            ic_y0: 1.2,
            ic_v0: -0.4,
            dt: 0.02,
            total_time: 4.0,
            comparison_mode: true,
            ..Default::default()
        };
        tool.recalculate();

        let snaps = tool.comparison_manager().snapshots();
        let labels: Vec<&str> = snaps.iter().map(|s| s.label.as_str()).collect();
        let mut expected_labels = vec!["Runge-Kutta 4", "Euler", "Heun (RK2)"];
        if active_method.is_symplectic() {
            expected_labels.push(active_method.name());
        }
        assert_eq!(labels, expected_labels);
        assert_eq!(
            tool.discrepancy_euler_rk4(),
            compute_discrepancy(&euler_ref, &rk4_ref)
        );
        assert_eq!(
            tool.discrepancy_heun_rk4(),
            compute_discrepancy(&heun_ref, &rk4_ref)
        );
        assert_eq!(
            tool.discrepancy_euler_heun(),
            compute_discrepancy(&euler_ref, &heun_ref)
        );

        for (snap, expected) in snaps.iter().zip([&rk4_ref, &euler_ref, &heun_ref]) {
            assert_snapshot_matches_reference(snap, expected);
        }
        if active_method.is_symplectic() {
            let symp_ref = reference_oscillator_trajectory(active_method);
            assert_snapshot_matches_reference(&snaps[3], &symp_ref);
        }
    }
    assert!(start.elapsed().as_secs_f64() >= 0.0);
}
