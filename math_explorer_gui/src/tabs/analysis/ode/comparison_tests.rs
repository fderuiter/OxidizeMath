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

    // All seven solvers should be present in comparison manager
    let mgr = tool.comparison_manager();
    assert_eq!(mgr.len(), 7);

    let labels: Vec<String> = mgr.snapshots().iter().map(|s| s.label.clone()).collect();
    for expected in [
        "Runge-Kutta 4",
        "Euler",
        "Heun (RK2)",
        "Dormand-Prince 4(5)",
        "Backward Euler",
        "Implicit Trapezoidal",
        "Rosenbrock 2(3)",
    ] {
        assert!(labels.contains(&expected.to_string()), "missing {expected}");
    }

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
    assert!(exported
        .iter()
        .any(|(name, _)| name.contains("Dormand-Prince")));
    assert!(exported.iter().any(|(name, _)| name.contains("Rosenbrock")));

    // State persistence roundtrip preserves comparison mode and active method
    tool.set_method(IntegrationMethod::DormandPrince45);
    let state = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&state);
    assert!(restored.comparison_mode());
    assert_eq!(restored.method(), IntegrationMethod::DormandPrince45);
    assert_eq!(restored.comparison_manager().len(), 7);
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
fn test_ode_tool_non_autonomous_and_stiff_implicit_solvers() {
    // 1. Non-autonomous y'(t) = -k * y * sin(t), y(0) = 1 => y(T) = exp(k * (cos(T) - 1))
    let mut mod_tool = OdeSolverTool {
        preset: OdePreset::Custom,
        method: IntegrationMethod::DormandPrince45,
        param_k: 1.5,
        ic_y0: 1.0,
        dt: 0.05,
        total_time: 4.0,
        ..Default::default()
    };
    mod_tool.math_input.set_expression("-k * y * sin(t)");
    mod_tool.math_input.set_param("k", 1.5);
    mod_tool.recalculate();
    let exact_mod = (1.5 * (4.0_f64.cos() - 1.0)).exp();
    let got_mod = *mod_tool.y_series.last().expect("modulated final");
    assert!(
        (got_mod - exact_mod).abs() < 1e-6,
        "got {got_mod} vs {exact_mod}"
    );

    // 2. Non-autonomous y'(t) = cos(t) - y, y(0) = 0 => y(T) = 0.5 * (sin(T) + cos(T) - exp(-T))
    let mut driven = OdeSolverTool {
        preset: OdePreset::Custom,
        method: IntegrationMethod::RungeKutta4,
        ic_y0: 0.0,
        dt: 0.02,
        total_time: 5.0,
        ..Default::default()
    };
    driven.math_input.set_expression("cos(t) - y");
    driven.recalculate();
    let exact_driven = 0.5 * (5.0_f64.sin() + 5.0_f64.cos() - (-5.0_f64).exp());
    let got_driven = *driven.y_series.last().expect("driven final");
    assert!((got_driven - exact_driven).abs() < 1e-6);

    // 3. Stiff y' = -1000 * y at dt = 0.05 remains bounded and strictly positive for implicit solvers
    for stiff_method in [
        IntegrationMethod::BackwardEuler,
        IntegrationMethod::ImplicitTrapezoidal,
        IntegrationMethod::Rosenbrock23,
    ] {
        let mut stiff = OdeSolverTool {
            preset: OdePreset::Custom,
            method: stiff_method,
            ic_y0: 1.0,
            dt: 0.05,
            total_time: 0.5,
            ..Default::default()
        };
        stiff.math_input.set_expression("-1000 * y");
        stiff.recalculate();
        assert!(!stiff.diverged(), "{stiff_method:?} diverged");
        assert!(
            stiff
                .y_series
                .iter()
                .all(|&y| y.is_finite() && y > 0.0 && y <= 1.0),
            "{stiff_method:?} lost positivity or boundedness: {:?}",
            stiff.y_series
        );
    }
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

#[test]
fn test_ode_comparison_reuses_active_trajectory_with_exact_parity_across_methods() {
    use crate::web_io::PlatformInstant;
    use comparison::{compute_discrepancy, compute_series_discrepancy};

    let refs = IntegrationMethod::ALL.map(reference_oscillator_trajectory);
    assert_eq!(
        compute_discrepancy(&refs[1], &refs[0]),
        compute_series_discrepancy(
            &refs[1].y_series,
            &refs[1].v_series,
            &refs[0].y_series,
            &refs[0].v_series,
        )
    );

    let expected_labels = IntegrationMethod::ALL.map(|m| m.name());
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
        assert_eq!(labels, expected_labels);
        assert_eq!(
            tool.discrepancy_euler_rk4(),
            compute_discrepancy(&refs[1], &refs[0])
        );
        assert_eq!(
            tool.discrepancy_heun_rk4(),
            compute_discrepancy(&refs[2], &refs[0])
        );
        assert_eq!(
            tool.discrepancy_euler_heun(),
            compute_discrepancy(&refs[1], &refs[2])
        );
        assert_eq!(
            tool.discrepancy_rk4_dp45,
            compute_discrepancy(&refs[0], &refs[3])
        );
        assert_eq!(
            tool.discrepancy_be_ros23,
            compute_discrepancy(&refs[4], &refs[6])
        );

        for (snap, expected) in snaps.iter().zip(&refs) {
            assert_eq!(snap.series[0].points.len(), expected.y_series.len());
            assert_eq!(snap.series[1].points.len(), expected.v_series.len());
            let last_pt = snap.series[0].points.last().expect("last y point");
            let last_ref = *expected.y_series.last().expect("expected final y");
            assert_eq!(last_pt[1].to_bits(), last_ref.to_bits());
        }
    }
    assert!(start.elapsed().as_secs_f64() >= 0.0);
}
