use super::*;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_trapezoidal_richardson_error_estimate() {
    // Linear integrand has zero curvature => zero Richardson error.
    let trap_even = Trapezoidal::new(100);
    let lin = trap_even.integrate(|x| 3.0 * x + 2.0, 0.0, 2.0, 1e-6);
    assert!((lin.value - 10.0).abs() < 1e-12);
    assert!(lin.error < 1e-12);

    // Smooth nonlinear integrand exp(x) over [0, 1]: true integral = e - 1.
    let exact = std::f64::consts::E - 1.0;
    let res_even = trap_even.integrate(f64::exp, 0.0, 1.0, 1e-6);
    let true_err_even = (res_even.value - exact).abs();
    assert!(res_even.error > 0.0);
    assert!((res_even.error - true_err_even).abs() < true_err_even * 0.05);

    // Odd step count also produces accurate Richardson error.
    let trap_odd = Trapezoidal::new(101);
    let res_odd = trap_odd.integrate(f64::exp, 0.0, 1.0, 1e-6);
    let true_err_odd = (res_odd.value - exact).abs();
    assert!(res_odd.error > 0.0);
    assert!((res_odd.error - true_err_odd).abs() < true_err_odd * 0.05);
}

#[test]
#[verified_engine::verified]
fn test_multivariate_integrator_hyperrectangle_and_zero_alloc() {
    // Integrate f(x, y, z) = x * y^2 + z over [0, 2] x [1, 3] x [-1, 1].
    // Volume = 2 * 2 * 2 = 8.
    // int_0^2 x dx = 2; int_1^3 y^2 dy = 26/3; int_{-1}^1 1 dz = 2 => first term = 2 * (26/3) * 2 = 104/3.
    // int z over [-1, 1] is 0 => exact integral = 104 / 3.
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

    // Inverted bounds rejected cleanly.
    assert!(matches!(
        sobol_int.integrate(&[2.0, 0.0, 0.0], &[1.0, 1.0, 1.0], |_| 1.0),
        Err(MonteCarloError::InvalidBounds { dimension: 0, .. })
    ));
}

// [cite:graph_parameters_rust]
