use super::*;

const GISTEMP_1880_2023: [f64; 144] = [
    -0.16, -0.08, -0.11, -0.17, -0.28, -0.33, -0.31, -0.36, -0.17, -0.10, -0.35, -0.22, -0.27,
    -0.31, -0.30, -0.23, -0.11, -0.11, -0.27, -0.18, -0.08, -0.16, -0.28, -0.37, -0.47, -0.26,
    -0.22, -0.39, -0.43, -0.48, -0.43, -0.44, -0.36, -0.34, -0.15, -0.14, -0.36, -0.46, -0.30,
    -0.27, -0.27, -0.19, -0.28, -0.26, -0.27, -0.22, -0.11, -0.21, -0.20, -0.36, -0.16, -0.09,
    -0.16, -0.28, -0.12, -0.20, -0.15, -0.03, 0.00, -0.02, 0.12, 0.18, 0.07, 0.09, 0.20, 0.09,
    -0.07, -0.03, -0.11, -0.11, -0.17, -0.07, 0.01, 0.08, -0.13, -0.14, -0.19, 0.05, 0.06, 0.03,
    -0.03, 0.06, 0.03, 0.05, -0.20, -0.11, -0.06, -0.02, -0.08, 0.05, 0.03, -0.08, 0.01, 0.16,
    -0.07, -0.01, -0.10, 0.18, 0.07, 0.16, 0.26, 0.32, 0.14, 0.31, 0.16, 0.12, 0.18, 0.32, 0.39,
    0.27, 0.45, 0.41, 0.22, 0.23, 0.31, 0.45, 0.33, 0.46, 0.61, 0.38, 0.39, 0.54, 0.63, 0.62, 0.53,
    0.68, 0.64, 0.66, 0.54, 0.66, 0.72, 0.61, 0.65, 0.68, 0.75, 0.90, 1.01, 0.92, 0.85, 0.98, 1.02,
    0.85, 0.89, 1.17,
];

#[test]
fn test_gistemp_degree4_fit_accuracy_and_scientific_formatting() {
    let pts: Vec<[f64; 2]> = GISTEMP_1880_2023
        .iter()
        .enumerate()
        .map(|(i, &y)| [(1880 + i) as f64, y])
        .collect();
    let res = fit_model_with_axes(&pts, FitType::Quartic, "GISTEMP", false, false, false)
        .expect("GISTEMP Degree 4 fit should converge");
    assert!(
        (res.r_squared - 0.9158).abs() < 0.001,
        "Expected GISTEMP Degree 4 R^2 ~ 0.9158, got {} (formula: {})",
        res.r_squared,
        res.formula
    );
    assert!(
        !res.formula.contains("0.0000"),
        "Formula must not truncate small coefficients to 0.0000: {}",
        res.formula
    );
    assert!(
        res.formula.contains("x⁴") && res.formula.contains("1.680e-8"),
        "Formula must retain x⁴ term in 4-sig-fig scientific notation: {}",
        res.formula
    );
}

#[test]
fn test_calendar_year_shifted_exponential_logistic_and_sinusoidal() {
    let exp_pts: Vec<[f64; 2]> = (1880..=2023)
        .map(|yr| {
            let x = yr as f64;
            [x, 0.15 * (0.025 * (x - 1880.0)).exp()]
        })
        .collect();
    let exp_res =
        fit_model_with_axes(&exp_pts, FitType::Exponential, "Exp", false, false, false).unwrap();
    assert!(exp_res.r_squared > 0.9999, "Exp R^2: {}", exp_res.r_squared);
    assert!(!exp_res.formula.contains("0.0000"));

    let log_pts: Vec<[f64; 2]> = (1880..=2023)
        .map(|yr| {
            let x = yr as f64;
            [x, 420.0 / (1.0 + (-0.055 * (x - 1960.0)).exp())]
        })
        .collect();
    let log_res =
        fit_model_with_axes(&log_pts, FitType::Logistic, "Log", false, false, false).unwrap();
    assert!(log_res.r_squared > 0.9999, "Log R^2: {}", log_res.r_squared);
    assert!(!log_res.formula.contains("0.0000"));

    let sin_pts: Vec<[f64; 2]> = (1880..=2023)
        .map(|yr| {
            let x = yr as f64;
            [x, 1.4 * (0.12 * (x - 1880.0) + 0.7).sin() + 0.35]
        })
        .collect();
    let sin_res =
        fit_model_with_axes(&sin_pts, FitType::Sinusoidal, "Sin", false, false, false).unwrap();
    assert!(sin_res.r_squared > 0.999, "Sin R^2: {}", sin_res.r_squared);
    assert!(!sin_res.formula.contains("0.0000"));
}
