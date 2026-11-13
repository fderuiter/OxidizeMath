#![allow(missing_docs)]
use approx::assert_abs_diff_eq;
use pure_math::pure_math::analysis::pde::{
    greens::GreenFunctionSolver1D, heat::HeatEquation1D, laplace::LaplaceEquation2D,
    wave::WaveEquation1D,
};
use pure_math::pure_math::special_functions::{bessel, legendre, polynomials};

#[test]
#[verified_engine::verified]
fn test_wave_equation_dalembert() {
    let wave = WaveEquation1D::new(2.0); // c = 2.0

    let f = |x: f64| x * x;
    let g = |x: f64| x;

    // u(x,t) = f(x-ct) + g(x+ct)
    // At x=1, t=1: f(1-2) + g(1+2) = f(-1) + g(3) = (-1)^2 + 3 = 1 + 3 = 4
    let val = wave.dalembert_solution(f, g, 1.0, 1.0);
    assert_abs_diff_eq!(
        val,
        4.0,
        epsilon = math_commons::registry::TOLERANCE_STANDARD
    );
}

#[test]
#[verified_engine::verified]
fn test_heat_equation_separated() {
    let heat = HeatEquation1D::new(0.5); // kappa = 0.5

    // u(x,t) = (A cos(kx) + B sin(kx)) * exp(-k^2 kappa t)
    // Let A=1, B=0, k=2.
    // u(x,t) = cos(2x) * exp(-4 * 0.5 * t) = cos(2x) * exp(-2t)

    let val = heat.separated_mode(2.0, (1.0, 0.0), 0.0, 1.0);
    // x=0 => cos(0) = 1.
    // t=1 => exp(-2) = 0.135335
    assert_abs_diff_eq!(
        val,
        (-2.0f64).exp(),
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
}

#[test]
#[verified_engine::verified]
fn test_laplace_equation() {
    // u(x,y) = sin(x) * sinh(y) => A=0, B=1, C=0, D=1, lambda=1
    let val = LaplaceEquation2D::separated_mode_cartesian(
        1.0,
        (0.0, 1.0),
        (0.0, 1.0),
        std::f64::consts::PI / 2.0,
        1.0,
    );
    // sin(pi/2) = 1. sinh(1) = 1.1752
    assert_abs_diff_eq!(
        val,
        1.0f64.sinh(),
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
}

#[test]
#[verified_engine::verified]
fn test_greens_function_simple() {
    // Solve y'' = f(x) with y(0)=0, y(1)=0.
    // Green's function is G(x, xi) = x(xi - 1) for x < xi, xi(x - 1) for x > xi.

    let g = |x: f64, xi: f64| -> f64 {
        if x <= xi {
            x * (xi - 1.0)
        } else {
            xi * (x - 1.0)
        }
    };

    // Let f(x) = -2. Then y'' = -2 => y(x) = -x^2 + x.
    // check at x=0.5. y(0.5) = -0.25 + 0.5 = 0.25.

    let solver = GreenFunctionSolver1D::new();
    let solution = solver.solve_at(0.5, (0.0, 1.0), g, |_| -2.0);

    assert_abs_diff_eq!(solution, 0.25, epsilon = 1e-4);
}

#[test]
#[verified_engine::verified]
fn test_legendre_polynomials() {
    // P_0(x) = 1
    assert_abs_diff_eq!(legendre::legendre_p(0, 0.5), 1.0);
    // P_1(x) = x
    assert_abs_diff_eq!(legendre::legendre_p(1, 0.5), 0.5);
    // P_2(x) = 0.5(3x^2 - 1) = 0.5(3*0.25 - 1) = 0.5(-0.25) = -0.125
    assert_abs_diff_eq!(legendre::legendre_p(2, 0.5), -0.125);
}

#[test]
#[verified_engine::verified]
fn test_legendre_orthogonality() {
    // P_1 and P_2 should be orthogonal.
    let dot = legendre::check_orthogonality_legendre(1, 2);
    assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-3);

    // Norm of P_1: integral P_1^2 = 2/(2*1+1) = 2/3 = 0.6666
    let norm = legendre::check_orthogonality_legendre(1, 1);
    assert_abs_diff_eq!(norm, 2.0 / 3.0, epsilon = 1e-3);
}

#[test]
#[verified_engine::verified]
fn test_hermite_polynomials() {
    // H_0 = 1
    assert_abs_diff_eq!(polynomials::hermite(0, 2.0), 1.0);
    // H_1 = 2x = 4
    assert_abs_diff_eq!(polynomials::hermite(1, 2.0), 4.0);
    // H_2 = 4x^2 - 2 = 16 - 2 = 14
    assert_abs_diff_eq!(polynomials::hermite(2, 2.0), 14.0);
}

#[test]
#[verified_engine::verified]
fn test_laguerre_polynomials() {
    // L_0 = 1
    assert_abs_diff_eq!(polynomials::laguerre(0, 2.0), 1.0);
    // L_1 = 1 - x = -1
    assert_abs_diff_eq!(polynomials::laguerre(1, 2.0), -1.0);
    // L_2 = (1/2)(x^2 - 4x + 2) = 0.5(4 - 8 + 2) = 0.5(-2) = -1
    assert_abs_diff_eq!(polynomials::laguerre(2, 2.0), -1.0);
}

#[test]
#[verified_engine::verified]
fn test_bessel_orthogonality_check() {
    // Roots of J_0: 2.4048, 5.5201 (approx)
    let alpha = 2.4048255577;
    let beta = 5.5200781103;
    let val = bessel::check_orthogonality_bessel(0.0, alpha, beta);
    // Should be close to 0
    assert!(val.abs() < 1e-3, "Bessel orthogonality failed: {}", val);
}

#[test]
#[verified_engine::verified]
fn test_bessel_y_stub_returns_nan() {
    assert!(bessel::bessel_y(0.0, 1.0).is_nan());
    assert!(bessel::bessel_y(1.0, 2.5).is_nan());
    assert!(bessel::bessel_y(-1.0, 2.0).is_nan());
    assert!(bessel::bessel_y(0.0, 0.0).is_nan());
    assert!(bessel::bessel_y(f64::INFINITY, 1.0).is_nan());
    assert!(bessel::bessel_y(0.0, f64::INFINITY).is_nan());
}
