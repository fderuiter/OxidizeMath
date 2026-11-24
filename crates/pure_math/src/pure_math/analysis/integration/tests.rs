#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::pure_math::statistics::monte_carlo::MonteCarloError;
use std::f64::consts::PI;
use verified_engine::engine::VerifiedEngine;

fn clamp_pulse(x: f64) -> f64 {
    (3.0 * x).sin().clamp(-0.5, 0.8)
}

fn rect_pulse(x: f64) -> f64 {
    if (0.25..=0.75).contains(&x) { 1.0 } else { 0.0 }
}

fn shifted_rect_pulse(x: f64) -> f64 {
    if (0.256..=0.748).contains(&x) {
        1.0
    } else {
        0.0
    }
}

#[test]
fn trapezoidal_simpson_and_gk15_bound_true_error_on_sin() {
    let exact = 2.0;

    let trap = Trapezoidal::new(200).integrate(|x| x.sin(), 0.0, PI, 1e-8);
    let trap_err = (trap.value - exact).abs();
    assert!(trap.error > 0.0);
    assert!(
        trap_err <= trap.error,
        "Trapezoidal true error {trap_err} exceeded estimate {}",
        trap.error
    );
    assert!(trap_err < 1e-4);

    let simp = Simpson::new(100).integrate(|x| x.sin(), 0.0, PI, 1e-10);
    let simp_err = (simp.value - exact).abs();
    assert!(simp.error > 0.0);
    assert!(
        simp_err <= simp.error,
        "Simpson true error {simp_err} exceeded estimate {}",
        simp.error
    );
    assert!(simp_err < 1e-7);

    let gk = GaussKronrod15::new().integrate(|x| x.sin(), 0.0, PI, 1e-12);
    let gk_err = (gk.value - exact).abs();
    assert!(gk.error > 0.0);
    assert!(
        gk_err <= gk.error,
        "GaussKronrod15 true error {gk_err} exceeded estimate {}",
        gk.error
    );
    assert!(gk_err < 1e-12);
}

#[test]
fn quadrature_error_estimates_bound_true_error_on_clamp_pulse() {
    let x1 = 0.8_f64.asin() / 3.0;
    let x2 = (PI - 0.8_f64.asin()) / 3.0;
    let x3 = (PI + 0.5_f64.asin()) / 3.0;
    let x4 = (2.0 * PI - 0.5_f64.asin()) / 3.0;
    let int_sin = |a: f64, b: f64| ((3.0 * a).cos() - (3.0 * b).cos()) / 3.0;
    let exact_clamp = int_sin(0.0, x1)
        + 0.8 * (x2 - x1)
        + int_sin(x2, x3)
        + (-0.5) * (x4 - x3)
        + int_sin(x4, 2.0);

    for (name, res) in [
        (
            "Trapezoidal(clamp)",
            Trapezoidal::new(200).integrate(clamp_pulse, 0.0, 2.0, 1e-6),
        ),
        (
            "Simpson(clamp)",
            Simpson::new(200).integrate(clamp_pulse, 0.0, 2.0, 1e-6),
        ),
        (
            "GK15(clamp)",
            GaussKronrod15::new().integrate(clamp_pulse, 0.0, 2.0, 1e-8),
        ),
    ] {
        let true_err = (res.value - exact_clamp).abs();
        assert!(res.error > 0.0, "{name} returned non-positive error");
        assert!(
            true_err <= res.error,
            "{name} true error {true_err} > estimated {}",
            res.error
        );
        assert!(true_err < 1e-3, "{name} inaccurate: {true_err}");
    }
}

#[test]
fn quadrature_error_estimates_bound_true_error_on_rect_pulses() {
    for (exact_rect, pulse_fn) in [
        (0.50, rect_pulse as fn(f64) -> f64),
        (0.492, shifted_rect_pulse as fn(f64) -> f64),
    ] {
        for (name, res) in [
            (
                "Trapezoidal(rect)",
                Trapezoidal::new(100).integrate(pulse_fn, 0.0, 1.0, 1e-6),
            ),
            (
                "Simpson(rect)",
                Simpson::new(100).integrate(pulse_fn, 0.0, 1.0, 1e-6),
            ),
            (
                "GK15(rect)",
                GaussKronrod15::new().integrate(pulse_fn, 0.0, 1.0, 1e-8),
            ),
        ] {
            let true_err = (res.value - exact_rect).abs();
            assert!(res.error > 0.0, "{name} returned non-positive error");
            assert!(
                true_err <= res.error,
                "{name} true error {true_err} > estimated {}",
                res.error
            );
            assert!(true_err < 2e-2, "{name} inaccurate: {true_err}");
        }
    }
}

#[test]
fn gk15_panel_export_is_sorted_and_covers_domain() {
    let mut panels = [GkSubinterval::default(); GK15_MAX_PANELS];
    let (res, count) =
        GaussKronrod15::new().integrate_panels(rect_pulse, 0.0, 1.0, 1e-8, &mut panels);
    assert!((4..=GK15_MAX_PANELS).contains(&count));
    assert!((panels[0].left - 0.0).abs() < 1e-15);
    assert!((panels[count - 1].right - 1.0).abs() < 1e-15);
    for w in panels[..count].windows(2) {
        assert!((w[0].right - w[1].left).abs() < 1e-15);
    }
    assert!((res.value - 0.5).abs() < 1e-6);
}

#[test]
fn test_trapezoidal_richardson_error_estimate() {
    let trap_even = Trapezoidal::new(100);
    let lin = trap_even.integrate(|x| 3.0 * x + 2.0, 0.0, 2.0, 1e-6);
    assert!((lin.value - 10.0).abs() < 1e-12);
    assert!(lin.error < 1e-12);

    let exact = std::f64::consts::E - 1.0;
    let res_even = trap_even.integrate(f64::exp, 0.0, 1.0, 1e-6);
    let true_err_even = (res_even.value - exact).abs();
    assert!(res_even.error > 0.0);
    assert!((res_even.error - true_err_even).abs() < true_err_even * 0.05);

    let trap_odd = Trapezoidal::new(101);
    let res_odd = trap_odd.integrate(f64::exp, 0.0, 1.0, 1e-6);
    let true_err_odd = (res_odd.value - exact).abs();
    assert!(res_odd.error > 0.0);
    assert!((res_odd.error - true_err_odd).abs() < true_err_odd * 0.05);
}

#[test]
fn test_multivariate_integrator_hyperrectangle_and_zero_alloc() {
    let exact = 104.0 / 3.0;
    let lower = [0.0, 1.0, -1.0];
    let upper = [2.0, 3.0, 1.0];

    let mut sobol_int = MultivariateIntegrator::new_sobol(3, 1024)
        .unwrap()
        .with_seed(99);
    let mut halton_int = MultivariateIntegrator::new_halton(3, 1024)
        .unwrap()
        .with_seed(99);

    let (sobol_res, _) = VerifiedEngine::run_verified(|| {
        sobol_int
            .integrate(&lower, &upper, |x| x[0] * x[1] * x[1] + x[2])
            .unwrap()
    });
    let halton_res = halton_int
        .integrate(&lower, &upper, |x| x[0] * x[1] * x[1] + x[2])
        .unwrap();

    let mut mc_int = MultivariateIntegrator::new_monte_carlo(3, 1024)
        .unwrap()
        .with_seed(99);
    let mc_res = mc_int
        .integrate(&lower, &upper, |x| x[0] * x[1] * x[1] + x[2])
        .unwrap();

    assert!(
        (sobol_res.value - exact).abs() < 0.05,
        "Sobol integral {} vs exact {}",
        sobol_res.value,
        exact
    );
    assert!(
        sobol_res.error > 0.0 && sobol_res.error < mc_res.error * 0.5,
        "Sobol SE {} should beat IID SE {}",
        sobol_res.error,
        mc_res.error
    );
    assert!(
        (halton_res.value - exact).abs() < 0.35,
        "Halton integral {} vs exact {}",
        halton_res.value,
        exact
    );
    assert!(halton_res.error > 0.0 && halton_res.error < mc_res.error * 0.5);

    assert!(matches!(
        sobol_int.integrate(&[2.0, 0.0, 0.0], &[1.0, 1.0, 1.0], |_| 1.0),
        Err(MonteCarloError::InvalidBounds { dimension: 0, .. })
    ));
}

// [cite:graph_parameters_rust]
