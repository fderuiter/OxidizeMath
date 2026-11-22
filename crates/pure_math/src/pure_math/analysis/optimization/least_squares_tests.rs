use super::*;
use nalgebra::{DMatrix, DVector};
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_michaelis_menten_and_hill_kinetics_zero_alloc_recovery() {
    let s_vals: [f64; 12] = [
        0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0,
    ];
    let true_vmax = 12.5_f64;
    let true_km = 2.4_f64;
    let true_n = 1.85_f64;
    let mut y_hill = [0.0_f64; 12];
    for i in 0..s_vals.len() {
        let sn = s_vals[i].powf(true_n);
        y_hill[i] = (true_vmax * sn) / (true_km.powf(true_n) + sn);
    }

    let mut objective = ClosureLeastSquaresObjective::new(&s_vals, &y_hill, |s, p| {
        let vmax = p[0];
        let km = p[1];
        let n = p[2];
        let sn = s.powf(n);
        (vmax * sn) / (km.powf(n) + sn)
    })
    .expect("valid hill objective");

    let bounds = [
        ParameterBounds::new(0.1, 100.0).unwrap(),
        ParameterBounds::new(0.05, 50.0).unwrap(),
        ParameterBounds::new(0.2, 6.0).unwrap(),
    ];
    let initial = [8.0, 5.0, 1.1];
    let mut ws = LevenbergMarquardtWorkspace::new(16, 4).expect("workspace");
    let solver = LevenbergMarquardtSolver::default();

    let (result, _metrics) = VerifiedEngine::run_verified(|| {
        solver.solve_in_place(&mut objective, &initial, &bounds, &mut ws)
    });
    let metrics = result.expect("hill fit succeeded with zero heap allocations");
    let summary = ws.to_summary(&metrics);

    assert!(((summary.parameters[0] - true_vmax) / true_vmax).abs() < 1e-3);
    assert!(((summary.parameters[1] - true_km) / true_km).abs() < 1e-3);
    assert!(((summary.parameters[2] - true_n) / true_n).abs() < 1e-3);
    assert!(summary.rmse < 1e-7);
    assert!((summary.r_squared - 1.0).abs() < 1e-9);
    assert!(summary.aic.is_finite());
    assert_eq!(summary.residuals.len(), s_vals.len());
}

#[test]
#[verified_engine::verified]
fn test_power_law_capacity_fade_zero_alloc_recovery() {
    let cycles: [f64; 10] = [
        100.0, 250.0, 400.0, 600.0, 800.0, 150.0, 350.0, 600.0, 900.0, 1200.0,
    ];
    let dods: [f64; 10] = [80.0, 80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 40.0, 40.0, 40.0];
    let true_alpha = 85_000.0_f64;
    let true_beta = -1.32_f64;
    let mut cap_obs = [0.0_f64; 10];
    for i in 0..cycles.len() {
        let n70 = true_alpha * dods[i].powf(true_beta);
        cap_obs[i] = 0.7_f64.powf(cycles[i] / n70);
    }

    let idx_grid: [f64; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut objective = ClosureLeastSquaresObjective::new(&idx_grid, &cap_obs, |idx, p| {
        let k = idx as usize;
        let n70 = p[0] * dods[k].powf(p[1]);
        if n70 <= 0.0 || !n70.is_finite() {
            return f64::NAN;
        }
        0.7_f64.powf(cycles[k] / n70)
    })
    .expect("power-law objective");

    let bounds = [
        ParameterBounds::new(1_000.0, 1_000_000.0).unwrap(),
        ParameterBounds::new(-4.0, -0.1).unwrap(),
    ];
    let initial = [101_900.0, -1.2639];
    let mut ws = LevenbergMarquardtWorkspace::new(16, 4).unwrap();
    let solver = LevenbergMarquardtSolver::default();

    let (res, _) = VerifiedEngine::run_verified(|| {
        solver.solve_in_place(&mut objective, &initial, &bounds, &mut ws)
    });
    let summary = ws.to_summary(&res.expect("power-law fit succeeded"));

    assert!(((summary.parameters[0] - true_alpha) / true_alpha).abs() < 1e-3);
    assert!(((summary.parameters[1] - true_beta) / true_beta.abs()).abs() < 1e-3);
    assert!(summary.rmse < 1e-6);
}

#[test]
#[verified_engine::verified]
fn test_box_bounds_and_noisy_confidence_intervals() {
    let x: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let noise: [f64; 8] = [0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.02, -0.02];
    let mut y = [0.0_f64; 8];
    for i in 0..8 {
        y[i] = 3.0 * (-0.5 * x[i]).exp() + noise[i];
    }

    let mut objective =
        ClosureLeastSquaresObjective::new(&x, &y, |t, p| p[0] * (-p[1] * t).exp()).unwrap();
    let tight_bounds = [
        ParameterBounds::new(0.5, 2.5).unwrap(),
        ParameterBounds::new(0.05, 5.0).unwrap(),
    ];
    let mut ws = LevenbergMarquardtWorkspace::new(8, 2).unwrap();
    let solver = LevenbergMarquardtSolver::default();
    let clamped_fit = solver
        .fit(&mut objective, &[1.5, 1.0], &tight_bounds, &mut ws)
        .unwrap();
    assert!((clamped_fit.parameters[0] - 2.5).abs() < 1e-6);

    let wide_bounds = [
        ParameterBounds::new(0.5, 10.0).unwrap(),
        ParameterBounds::new(0.05, 5.0).unwrap(),
    ];
    let open_fit = solver
        .fit(&mut objective, &[2.0, 0.8], &wide_bounds, &mut ws)
        .unwrap();
    assert!((open_fit.parameters[0] - 3.0).abs() < 0.05);
    assert!((open_fit.parameters[1] - 0.5).abs() < 0.05);
    assert!(
        open_fit
            .standard_errors
            .iter()
            .all(|&se| se > 0.0 && se.is_finite())
    );
    assert!(
        open_fit
            .ci_95_half_widths
            .iter()
            .all(|&ci| ci > 0.0 && ci.is_finite())
    );
    assert!(open_fit.r_squared > 0.99);
}

#[test]
#[verified_engine::verified]
fn test_dimension_caps_and_dmatrix_step_consistency() {
    assert!(LevenbergMarquardtWorkspace::new(10, 17).is_err());
    assert!(LevenbergMarquardtWorkspace::new(10_001, 2).is_err());

    let j = DMatrix::from_row_slice(3, 2, &[1.0, 0.5, 1.0, 1.5, 1.0, 2.5]);
    let r = DVector::from_column_slice(&[-0.2, 0.1, -0.05]);
    let step = LevenbergMarquardtSolver::solve_damped_step_matrix(&j, &r, 0.01, 1e-10);
    assert_eq!(step.len(), 2);
    assert!(step.iter().all(|v| v.is_finite()));
}
