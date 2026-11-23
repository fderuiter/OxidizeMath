#![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

use crate::{
    DiffeqError, GaussKronrod15, Integrator, Simpson, SolverWorkspace, Trapezoidal,
    solve_hermite_root, solve_hermite_unit_bracket, solve_monotone_hermite_crossing,
    trapezoidal_nonuniform, trapezoidal_uniform_grid,
};
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_trapezoidal_error_estimate_and_eps_refinement() {
    let exact = 1000.0 / 3.0; // integral of x^2 over [0, 10]
    let fixed = Trapezoidal::new(10).integrate(|x| x * x, 0.0, 10.0, 0.0);
    assert!(fixed.error > 0.0, "expected non-zero Richardson error");
    assert!(
        fixed.error >= (fixed.value - exact).abs() - 1e-12,
        "Richardson error {} must bound true error {}",
        fixed.error,
        (fixed.value - exact).abs()
    );

    let refined = Trapezoidal::new(10).integrate(|x| x * x, 0.0, 10.0, 1e-3);
    assert!(refined.error <= 1e-3, "refined error = {}", refined.error);
    assert!(
        (refined.value - exact).abs() <= refined.error,
        "true error {} must be bounded by {}",
        (refined.value - exact).abs(),
        refined.error
    );
}

#[test]
fn test_simpson_and_gauss_kronrod15_smooth_and_oscillatory_accuracy() {
    let exact_poly = 1.0 / 7.0; // integral of x^6 on [0, 1]
    let simp = Simpson::new(20).integrate(|x| x.powi(6), 0.0, 1.0, 1e-8);
    assert!((simp.value - exact_poly).abs() <= 1e-8);
    assert!(simp.error >= (simp.value - exact_poly).abs() * 0.5);

    let gk = GaussKronrod15::new();
    // Smooth exponential: integral of exp(x) on [0, 3] = e^3 - 1
    let exact_exp = 3.0_f64.exp() - 1.0;
    let res_exp = gk.integrate(f64::exp, 0.0, 3.0, 1e-12);
    let true_err_exp = (res_exp.value - exact_exp).abs();
    assert!(true_err_exp < 1e-12, "GK15 exp err = {true_err_exp:e}");
    assert!(res_exp.error >= true_err_exp);

    // Rapidly oscillatory: integral of cos(25 * x) on [0, pi] = sin(25 * pi) / 25 = 0.0,
    // and sin(15 * x) on [0, pi] = (1 - cos(15 * pi)) / 15 = 2 / 15
    let exact_osc = 2.0 / 15.0;
    let res_osc = gk.integrate(|x| (15.0 * x).sin(), 0.0, core::f64::consts::PI, 1e-10);
    let true_err_osc = (res_osc.value - exact_osc).abs();
    assert!(true_err_osc <= 1e-10, "GK15 osc err = {true_err_osc:e}");
    assert!(
        res_osc.error >= true_err_osc,
        "estimated {} < true {}",
        res_osc.error,
        true_err_osc
    );
}

#[test]
fn test_zero_heap_allocations_for_quadrature_and_hermite_kernels() {
    let xs = [0.0, 1.0, 2.5, 4.0];
    let ys = [1.0, 3.0, 2.0, 5.0];
    let mut gk_scratch = [0.0_f64; 128];

    VerifiedEngine::run_verified(|| {
        let r_trap = Trapezoidal::new(16).integrate(|x| x * x + 1.0, 0.0, 4.0, 1e-3);
        let r_simp = Simpson::new(16).integrate(|x| (3.0 * x).cos(), 0.0, 2.0, 1e-6);
        let r_gk =
            GaussKronrod15::new().integrate(|x| (8.0 * x).sin() * (-x).exp(), 0.0, 5.0, 1e-9);
        let mut ws = SolverWorkspace::from_slice(&mut gk_scratch);
        let r_gk_ws = GaussKronrod15::new()
            .integrate_with_workspace(|x| x * x * x, 0.0, 2.0, 1e-11, &mut ws)
            .expect("GK15 workspace");
        assert!((r_gk_ws.value - 4.0).abs() < 1e-12);

        let a_non = trapezoidal_nonuniform(&xs, &ys);
        let a_uni = trapezoidal_uniform_grid(&ys, 0.5);
        let s_root = solve_hermite_root(0.1, -0.4, 0.6, 10.0, 10.0);
        let x_cross = solve_monotone_hermite_crossing([1.0, -1.0], [3.0, 1.0], 2.0, 1.0, 1.0, 0.0);

        assert!(r_trap.value.is_finite() && r_simp.value.is_finite() && r_gk.value.is_finite());
        assert!(a_non > 0.0 && a_uni > 0.0);
        assert!((s_root - 0.4).abs() < 1e-12);
        assert_eq!(x_cross.map(|v| (v - 2.0).abs() < 1e-12), Some(true));
    });

    assert_eq!(
        solve_hermite_unit_bracket(0.1, 1.0, 2.0, 0.0, 0.0, 1e-12, 16),
        Err(DiffeqError::NotBracketed { fa: 1.0, fb: 2.0 })
    );
}
