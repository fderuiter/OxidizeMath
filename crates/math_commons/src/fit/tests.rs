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
