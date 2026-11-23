//! Unit tests for scalar and multivariate root-finding algorithms.

use super::*;
use crate::pure_math::analysis::ode::{ArrayState, VecState};
use nalgebra::{DVector, Vector2};
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_newton_raphson_sqrt() -> Result<(), AnalysisError> {
    let solver = NewtonRaphson::default();
    let root = solver.find_root_with_derivative(|x| x * x - 2.0, |x| 2.0 * x, 1.5)?;
    assert!((root - std::f64::consts::SQRT_2).abs() < math_commons::registry::TOLERANCE_FAST);
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_newton_raphson_numerical() -> Result<(), AnalysisError> {
    let solver = NewtonRaphson::default();
    let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
    assert!((root - std::f64::consts::SQRT_2).abs() < 1e-5);
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_bisection_square_root_and_linear() -> Result<(), AnalysisError> {
    let solver = Bisection::default();
    let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
    assert!((root - std::f64::consts::SQRT_2).abs() < math_commons::registry::TOLERANCE_FAST);

    let lin = solver.find_root(|x| 2.0 * x - 4.0, 0.0, 5.0)?;
    assert!((lin - 2.0).abs() < math_commons::registry::TOLERANCE_FAST);
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_bisection_nan_rejection_and_negative_zero_root() {
    let solver = Bisection::new(100, 0.0);

    // Domain error at left endpoint: ln(-1.0) - 1.0 = NaN must return RootNotBracketed
    let nan_res = Bisection::default().find_root(|x| x.ln() - 1.0, -1.0, 5.0);
    assert!(matches!(
        nan_res,
        Err(AnalysisError::RootNotBracketed { .. })
    ));

    // Infinite endpoint value must also be rejected
    let inf_res = Bisection::default().find_root(|x| 1.0 / x, 0.0, 2.0);
    assert!(matches!(
        inf_res,
        Err(AnalysisError::RootNotBracketed { .. })
    ));

    // Exact -0.0 at endpoint (even with tolerance = 0.0 and endpoint = -0.0) returns Ok(0.0)
    let neg_zero_left = solver
        .find_root(|x| if x == 0.0 { -0.0 } else { x }, -0.0, 2.0)
        .expect("exact -0.0 root at lower endpoint");
    assert_eq!(neg_zero_left, 0.0);
    assert!(neg_zero_left.is_sign_positive());

    let neg_zero_right = solver
        .find_root(|x| if x == 0.0 { -0.0 } else { x }, -2.0, -0.0)
        .expect("exact -0.0 root at upper endpoint");
    assert_eq!(neg_zero_right, 0.0);
    assert!(neg_zero_right.is_sign_positive());

    // Non-bracketed curve returns RootNotBracketed
    let unbracketed = solver.find_root(|x| x * x + 1.0, -2.0, 2.0);
    assert!(matches!(
        unbracketed,
        Err(AnalysisError::RootNotBracketed { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_brent_convergence_fallbacks_and_secant() -> Result<(), AnalysisError> {
    let tol = math_commons::registry::TOLERANCE_FAST;
    let brent = Brent::default();
    let secant = Secant::default();

    // Algebraic cubic: x^3 - 2x - 5 = 0 (Wallis's cubic, root ~ 2.0945514815423265)
    let expected_cubic = 2.094_551_481_542_326_5;
    let r_brent_alg = brent.find_root(|x| x * x * x - 2.0 * x - 5.0, 1.5, 3.0)?;
    let r_secant_alg = secant.find_root(|x| x * x * x - 2.0 * x - 5.0, 1.5, 3.0)?;
    assert!((r_brent_alg - expected_cubic).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!((r_secant_alg - expected_cubic).abs() < tol);

    // Transcendental Dottie equation: cos(x) - x = 0 (root ~ 0.7390851332151607)
    let dottie = 0.739_085_133_215_160_7;
    let r_brent_tr = brent.find_root(|x| x.cos() - x, 0.0, 1.5)?;
    let r_secant_tr = secant.find_root(|x| x.cos() - x, 0.0, 1.5)?;
    assert!((r_brent_tr - dottie).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!((r_secant_tr - dottie).abs() < tol);

    // Reversed bounds, flat triple root (forces bisection fallback), and exact endpoint hit
    let reversed = brent.find_root(|x| x * x - 2.0, 2.0, 1.0)?;
    assert!(
        (reversed - std::f64::consts::SQRT_2).abs() < math_commons::registry::TOLERANCE_STANDARD
    );
    let cubic = brent.find_root(|x| (x - 0.3).powi(3), -1.0, 2.0)?;
    assert!((cubic - 0.3).abs() < math_commons::registry::TOLERANCE_STANDARD);
    let exact_min = brent.find_root(|x| x - 1.0, 1.0, 3.0)?;
    assert!((exact_min - 1.0).abs() < math_commons::registry::TOLERANCE_HIGH);

    // Kepler equation: x - 0.8 * sin(x) - 1.0 = 0
    let r_kepler = brent.find_root(|x| x - 0.8 * x.sin() - 1.0, 0.0, 3.0)?;
    assert!((r_kepler - 0.8 * r_kepler.sin() - 1.0).abs() < tol);

    // Invalid NaN evaluation rejected by Brent
    assert!(matches!(
        brent.find_root(|x| x.ln() - 1.0, -1.0, 5.0),
        Err(AnalysisError::RootNotBracketed { .. })
    ));
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_all_root_finders_zero_allocation_on_convergence_and_errors() {
    let ((r_bis, r_newt, r_brent, err_bis, err_newt, err_brent, err_bounds), _) =
        VerifiedEngine::run_verified(|| {
            let bis = Bisection::default();
            let newt = NewtonRaphson::default();
            let brent = Brent::new(200, 1e-11);

            let ok_bis = bis.find_root(|x| x * x - 2.0, 1.0, 2.0);
            let ok_newt = newt.find_root_with_derivative(|x| x * x - 2.0, |x| 2.0 * x, 1.5);
            let ok_brent = brent.find_root(|x| x.exp() - 3.0, 0.0, 2.0);

            let bad_bis = bis.find_root(|x| x * x + 1.0, -1.0, 1.0);
            let bad_newt = newt.find_root_with_derivative(|x| x * x + 1.0, |_| 0.0, 0.0);
            let bad_brent = brent.find_root(|x| x * x + 1.0, -1.0, 1.0);
            let bad_bounds = brent.find_root(|x| x, f64::NAN, 1.0);

            (
                ok_bis, ok_newt, ok_brent, bad_bis, bad_newt, bad_brent, bad_bounds,
            )
        });

    assert!((r_bis.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    assert!((r_newt.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    assert!((r_brent.unwrap() - 3.0_f64.ln()).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!(matches!(
        err_bis,
        Err(AnalysisError::RootNotBracketed { .. })
    ));
    assert!(matches!(
        err_newt,
        Err(AnalysisError::DerivativeTooSmall { .. })
    ));
    assert!(matches!(
        err_brent,
        Err(AnalysisError::RootNotBracketed { .. })
    ));
    assert!(matches!(
        err_bounds,
        Err(AnalysisError::InvalidBracket { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_multivariate_newton_stack_and_dynamic_coupled_2d() -> Result<(), AnalysisError> {
    let tol = math_commons::registry::TOLERANCE_FAST;
    let solver = MultivariateNewton::default();

    let r_fixed = solver.solve_fixed(
        |v| [v[0] * v[0] + v[1] * v[1] - 5.0, v[0] * v[1] - 2.0],
        [2.2, 0.8],
    )?;
    assert!((r_fixed[0] - 2.0).abs() < tol && (r_fixed[1] - 1.0).abs() < tol);

    let r_jac = solver.solve_fixed_with_jacobian(
        |v| [v[0].exp() + v[1] - 2.0, v[0] + v[1] * v[1] - 1.0],
        |v| [[v[0].exp(), 1.0], [1.0, 2.0 * v[1]]],
        [0.4, 0.7],
    )?;
    assert!(r_jac[0].abs() < tol && (r_jac[1] - 1.0).abs() < tol);

    let r_svec = solver.solve_svector(
        |v| Vector2::new(v[1] - v[0] * v[0] + 1.0, v[0] - v[1].sin() - 0.5),
        Vector2::new(1.0, 0.2),
    )?;
    assert!((r_svec[1] - r_svec[0] * r_svec[0] + 1.0).abs() < tol);
    assert!((r_svec[0] - r_svec[1].sin() - 0.5).abs() < tol);

    let mut dyn_solver = MultivariateNewton::with_dimension(2, 100, tol);
    let mut dv = DVector::from_vec(vec![2.3, 0.7]);
    dyn_solver.solve_dvector_in_place(
        |x, out| {
            out[0] = x[0] * x[0] + x[1] * x[1] - 5.0;
            out[1] = x[0] * x[1] - 2.0;
        },
        &mut dv,
    )?;
    assert!((dv[0] - 2.0).abs() < tol && (dv[1] - 1.0).abs() < tol);

    let mut vec_state = VecState(vec![0.8, 2.2]);
    dyn_solver.solve_state(
        |s, out| {
            out.0[0] = s.0[0] * s.0[0] + s.0[1] * s.0[1] - 5.0;
            out.0[1] = s.0[0] * s.0[1] - 2.0;
        },
        &mut vec_state,
    )?;
    assert!((vec_state.0[0] - 1.0).abs() < tol && (vec_state.0[1] - 2.0).abs() < tol);

    let mut arr_state = ArrayState([2.2, 0.8]);
    dyn_solver.solve_state(
        |s, out| {
            out.0[0] = s.0[0] * s.0[0] + s.0[1] * s.0[1] - 5.0;
            out.0[1] = s.0[0] * s.0[1] - 2.0;
        },
        &mut arr_state,
    )?;
    assert!((arr_state.0[0] - 2.0).abs() < tol && (arr_state.0[1] - 1.0).abs() < tol);
    Ok(())
}

// [cite:graph_parameters_rust]
