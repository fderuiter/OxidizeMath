// [cite:math_commons]
use super::*;
use std::collections::HashSet;

#[test]
#[verified_engine::verified]
fn test_canonical_fit_model_hash_and_degree_bounds() {
    let mut set = HashSet::new();
    for preset in FitModel::MENU_PRESETS {
        assert!(set.insert(preset), "Duplicate preset in MENU_PRESETS");
    }
    assert_eq!(set.len(), 9);
    assert_eq!(FitModel::from_degree(1), Ok(FitModel::Linear));
    assert_eq!(FitModel::from_degree(2), Ok(FitModel::Quadratic));
    assert_eq!(FitModel::from_degree(3), Ok(FitModel::Cubic));
    assert_eq!(FitModel::from_degree(4), Ok(FitModel::Quartic));
    assert_eq!(
        FitModel::from_degree(0),
        Err(FitError::UnsupportedDegree { degree: 0 })
    );
    assert_eq!(
        FitModel::from_degree(5),
        Err(FitError::UnsupportedDegree { degree: 5 })
    );
}

#[test]
#[verified_engine::verified]
fn test_polynomial_regression_degrees_1_through_4() {
    let pts: Vec<[f64; 2]> = (-10..=10)
        .map(|i| {
            let x = i as f64 * 0.25;
            [
                x,
                1.5 - 2.0 * x + 0.75 * x * x - 0.2 * x.powi(3) + 0.05 * x.powi(4),
            ]
        })
        .collect();

    let fit = fit_polynomial(&pts, 4).expect("quartic fit");
    assert_eq!(fit.fit_model, FitModel::Quartic);
    assert!(fit.r_squared > 0.999_999);
    assert!((fit.coefficients()[0] - 1.5).abs() < 1e-8);
    assert!((fit.coefficients()[4] - 0.05).abs() < 1e-8);
    assert_eq!(
        fit_polynomial(&pts, 6),
        Err(FitError::UnsupportedDegree { degree: 6 })
    );

    let x0 = 1_000_000.0_f64;
    let shifted: Vec<[f64; 2]> = (0..=40)
        .map(|i| {
            let u = -2.0 + (i as f64) * 0.1;
            let x = x0 + u;
            let y = 1.5 - 0.8 * u + 2.2 * u * u - 0.35 * u.powi(3) + 0.12 * u.powi(4);
            [x, y]
        })
        .collect();
    let shifted_fit = fit_polynomial(&shifted, 4).expect("centered high-offset quartic");
    assert!(shifted_fit.r_squared > 0.999_999);
    assert!((shifted_fit.eval(x0) - 1.5).abs() < 1e-6);
}

#[test]
#[verified_engine::verified]
fn test_nonlinear_exponential_power_log_logistic_and_sinusoidal() {
    let exp_pts: Vec<(f64, f64)> = (0..=20)
        .map(|i| {
            let x = i as f64 * 0.2;
            (x, 2.5 * (-0.4 * x).exp())
        })
        .collect();
    let exp_fit = fit_model(&exp_pts, FitModel::Exponential, false).expect("exp fit");
    assert!((exp_fit.coefficients()[0] - 2.5).abs() < 1e-6);
    assert!((exp_fit.coefficients()[1] + 0.4).abs() < 1e-6);

    let log_pts: Vec<[f64; 2]> = (1..=30)
        .map(|i| {
            let x = i as f64 * 0.5;
            [x, 12.0 / (1.0 + (-0.8 * (x - 6.0)).exp())]
        })
        .collect();
    let logistic_fit = fit_model(&log_pts, FitModel::Logistic, false).expect("logistic fit");
    assert!((logistic_fit.coefficients()[0] - 12.0).abs() < 1e-4);
    assert!((logistic_fit.coefficients()[1] - 0.8).abs() < 1e-4);

    let xs: Vec<f64> = (0..=200).map(|i| i as f64 * 0.05).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| 3.2 * (1.7 * x + 0.4).sin() - 1.1)
        .collect();
    let sin_fit = fit_model(&(xs.as_slice(), ys.as_slice()), FitModel::Sinusoidal, false)
        .expect("sinusoidal fit");
    assert!((sin_fit.coefficients()[0] - 3.2).abs() < 1e-4);
    assert!((sin_fit.coefficients()[1] - 1.7).abs() < 1e-4);
    assert!((sin_fit.coefficients()[3] + 1.1).abs() < 1e-4);
}

#[test]
#[verified_engine::verified]
fn test_fit_error_on_invalid_domains_singular_and_non_convergence() {
    let neg = [[-2.0, -3.0], [-1.0, -1.0], [0.0, -0.5]];
    assert_eq!(
        fit_model(&neg, FitModel::PowerLaw, false),
        Err(FitError::InvalidDomain {
            reason: "power-law fit requires at least 2 points with x > 0 and y > 0"
        })
    );

    let dup_x = [[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]];
    assert_eq!(fit_polynomial(&dup_x, 1), Err(FitError::SingularMatrix));

    let flat_pts = [[0.0, 1.0], [1.0, 2.0], [2.0, 5.0], [3.0, 10.0]];
    let stalled = levenberg_marquardt(&flat_pts, &[0.5, 0.5], |x, _p| (0.0, [1.0, x, 0.0, 0.0]));
    assert_eq!(
        stalled,
        Err(FitError::NonConvergence {
            iterations: MAX_LM_ITERATIONS
        })
    );
}

#[test]
#[verified_engine::verified]
fn test_six_parameter_levenberg_marquardt_numeric() {
    let true_p = [1.2, -0.35, 2.4, 0.6, 0.9, 1.7];
    let eval_wave = |x: f64, p: &[f64]| -> Option<f64> {
        if p.len() < 6 {
            return None;
        }
        let exp_arg = (-p[3] * x).clamp(-700.0, 700.0);
        let y = p[0] + p[1] * x + p[2] * exp_arg.exp() + p[4] * (p[5] * x).sin();
        y.is_finite().then_some(y)
    };
    let mut pts = [[0.0_f64; 2]; 61];
    for (i, slot) in pts.iter_mut().enumerate() {
        let x = (i as f64) * 0.15;
        *slot = [x, eval_wave(x, &true_p).expect("finite y")];
    }

    let seeds = [1.0, -0.2, 2.0, 0.4, 0.6, 1.5];
    let fitted = levenberg_marquardt_numeric(&pts, &seeds, eval_wave).expect("6-param numeric LM");
    let sse = sum_squared_errors_numeric(&pts, &fitted, eval_wave).expect("finite sse");
    assert!(sse < 1e-8, "expected SSE < 1e-8, got {sse}");
    assert_eq!(
        levenberg_marquardt_numeric(&pts, &[1.0; 7], eval_wave),
        Err(FitError::UnsupportedDegree { degree: 7 })
    );
}

#[test]
fn test_six_parameter_lm_single_reverse_pass_per_point_and_zero_alloc() {
    use crate::ast::{CompiledExpr, SolverWorkspace, parse};
    use std::cell::Cell;
    use verified_engine::engine::VerifiedEngine;

    let expr = parse("p0 + p1 * x + p2 * exp(-p3 * x) + p4 * sin(p5 * x)").expect("parse");
    let compiled = CompiledExpr::compile_with_vars_and_params(
        &expr,
        &["x"],
        &["p0", "p1", "p2", "p3", "p4", "p5"],
    )
    .expect("compile");

    let true_p = [1.2, -0.35, 2.4, 0.6, 0.9, 1.7];
    let mut pts = [[0.0_f64; 2]; 41];
    for (i, slot) in pts.iter_mut().enumerate() {
        let x = (i as f64) * 0.15;
        *slot = [x, compiled.eval_1d(x, &true_p)];
    }
    let seeds = [1.0, -0.2, 2.0, 0.4, 0.6, 1.5];

    // Count reverse passes on the first normal-equation sweep: 1 pass/point (not 13 passes/point).
    let grad_passes = Cell::new(0usize);
    let mut ws = SolverWorkspace::new();
    let cb = ReverseGradFit {
        eval_val: |x: f64, p: &[f64]| Some(compiled.eval_1d(x, p)),
        eval_grad: |x: f64, p: &[f64], grad_out: &mut [f64], work: &mut SolverWorkspace| {
            grad_passes.set(grad_passes.get() + 1);
            let mut vars = [0.0_f64; 7];
            let mut grads = [0.0_f64; 7];
            vars[0] = x;
            vars[1..7].copy_from_slice(&p[..6]);
            let y = compiled.eval_grad_into(&vars, work, &mut grads);
            y.is_finite()
                .then(|| grad_out[..6].copy_from_slice(&grads[1..7]))
                .map(|()| y)
        },
    };
    let mut seed_arr = [0.0_f64; MAX_FIT_PARAMS];
    seed_arr.copy_from_slice(&seeds);
    for &[x, _] in &pts {
        let (y, jac) = cb.eval_point_and_jac(x, &seed_arr, 6, &mut ws);
        assert!(y.is_finite() && jac.iter().all(|v| v.is_finite()));
    }
    assert_eq!(
        grad_passes.get(),
        pts.len(),
        "expected exactly 1 reverse pass per point"
    );

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (fitted, _) = VerifiedEngine::run_verified(|| {
        levenberg_marquardt_numeric(&pts, &seeds, &compiled).expect("6-param reverse-mode LM")
    });
    let sse = sum_squared_errors_numeric(&pts, &fitted, &compiled).expect("finite sse");
    assert!(
        sse < 1e-18,
        "expected machine-precision SSE < 1e-18, got {sse}"
    );
}
