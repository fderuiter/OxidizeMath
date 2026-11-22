use std::time::Instant;

use super::*;

#[test]
fn test_linear_and_polynomial_fits_exact() {
    let lin = vec![[0.0, 1.0], [1.0, 3.5], [2.0, 6.0], [3.0, 8.5]];
    let res = fit_dataset(&lin, 1, "TestLine").expect("Fit failed");
    assert_eq!(res.degree, 1);
    assert!((res.r_squared - 1.0).abs() < 1e-12);
    assert!(res.rmse < 1e-10);
    assert_eq!(res.residuals.len(), 4);

    let quad: Vec<[f64; 2]> = (-2..=2)
        .map(|i| {
            let x = i as f64;
            [x, -1.5 * x * x + 2.0 * x + 4.0]
        })
        .collect();
    let res_q = fit_dataset(&quad, 2, "TestQuad").expect("Quad fit failed");
    assert!((res_q.r_squared - 1.0).abs() < 1e-12);

    let quartic: Vec<[f64; 2]> = (0..=6)
        .map(|i| {
            let x = i as f64;
            [x, 0.25 * x.powi(4) - x.powi(3) + 2.0]
        })
        .collect();
    let res_4 = fit_dataset(&quartic, 4, "Quartic").expect("Quartic fit failed");
    assert_eq!(res_4.fit_type, FitType::Quartic);
    assert!((res_4.r_squared - 1.0).abs() < 1e-9);
    assert_eq!(
        fit_dataset(&quartic, 5, "Deg5"),
        Err(FitError::UnsupportedDegree { degree: 5 })
    );
}

#[test]
fn test_edge_cases_and_nan_rejection() {
    assert!(matches!(
        fit_dataset(&[[1.0, 2.0]], 1, "Single"),
        Err(FitError::InsufficientPoints {
            required: 2,
            actual: 1
        })
    ));
    let singular = vec![
        [1.0, 2.0],
        [1.0, 5.0],
        [f64::NAN, 3.0],
        [2.0, f64::INFINITY],
    ];
    assert_eq!(
        fit_dataset(&singular, 1, "Singular"),
        Err(FitError::SingularMatrix)
    );
    assert!(fit_model(&singular, FitType::Exponential, "BadExp", false).is_err());
    assert!(fit_model(&singular, FitType::Sinusoidal, "BadSin", false).is_err());
}

#[test]
fn test_fit_dataset_scaled_log_axes() {
    let pts = vec![
        [0.0, 0.0],
        [-5.0, 10.0],
        [1.0, 2.5],
        [2.0, 20.0],
        [4.0, 160.0],
        [10.0, 2500.0],
    ];
    let res = fit_dataset_scaled(&pts, 1, "PowerLaw", true, true).expect("Log-log fit failed");
    assert!(res.log_x && res.log_y);
    assert!((res.r_squared - 1.0).abs() < 1e-9);
    assert!((res.eval(4.0) - 160.0).abs() < 1e-6);
    assert!(res.eval(0.0).is_nan());
}

#[test]
fn benchmark_harmonic_oscillator_sinusoidal_r2_above_099() {
    let pts: Vec<[f64; 2]> = (0..=100)
        .map(|i| {
            let t = i as f64 * 0.1;
            [t, 1.5 * (1.2 * t + 0.4).sin() - 0.3]
        })
        .collect();
    let fit = fit_model(&pts, FitType::Sinusoidal, "HarmonicOscillator", false)
        .expect("Sinusoidal fit failed");
    assert!(fit.r_squared >= 0.99, "got R² = {}", fit.r_squared);
    assert!(fit.rmse < 1e-4);
}

#[test]
fn benchmark_logistic_growth_and_sir_recovered_r2_above_099() {
    let logistic_pts: Vec<[f64; 2]> = (0..=80)
        .map(|i| {
            let t = i as f64 * 0.25;
            let k = 10.0;
            let y0 = 0.5;
            let r = 0.6;
            let y = k / (1.0 + ((k - y0) / y0) * (-r * t).exp());
            [t, y]
        })
        .collect();
    let fit_lg = fit_model(&logistic_pts, FitType::Logistic, "LogisticGrowth", false)
        .expect("Logistic fit failed");
    assert!(
        fit_lg.r_squared >= 0.99,
        "LogisticGrowth R² = {}",
        fit_lg.r_squared
    );

    // Classic Kermack-McKendrick SIR simulation (N=1000, I0=10, R0=0, beta=0.3, gamma=0.1)
    let (mut s, mut inf, mut rec, n_pop, dt) = (990.0_f64, 10.0_f64, 0.0_f64, 1000.0_f64, 0.5_f64);
    let mut sir_rec = Vec::with_capacity(321);
    for step in 0..=320 {
        let t = step as f64 * dt;
        sir_rec.push([t, rec]);
        let new_inf = 0.3 * s * inf / n_pop;
        let new_rec = 0.1 * inf;
        s -= new_inf * dt;
        inf += (new_inf - new_rec) * dt;
        rec += new_rec * dt;
    }
    let fit_sir =
        fit_model(&sir_rec, FitType::Logistic, "SIR Recovered", false).expect("SIR fit failed");
    assert!(fit_sir.r_squared >= 0.99, "SIR R² = {}", fit_sir.r_squared);
}

#[test]
fn benchmark_beam_profiling_and_power_law_n70_r2_above_099() {
    let mu = 0.08_f64;
    let psi0 = 120.0_f64;
    let beam_pts: Vec<[f64; 2]> = (0..=60)
        .map(|i| {
            let d = i as f64 * 0.5;
            [d, mu * psi0 * (-mu * d).exp()]
        })
        .collect();
    let fit_beam = fit_model(&beam_pts, FitType::Exponential, "BeamProfiling", false)
        .expect("Beam exponential fit failed");
    assert!(
        fit_beam.r_squared >= 0.99,
        "Beam R² = {}",
        fit_beam.r_squared
    );

    let n70_pts: Vec<[f64; 2]> = (1..=20)
        .map(|i| {
            let dod = i as f64 * 0.05;
            [dod, 850.0 * dod.powf(-1.35)]
        })
        .collect();
    let fit_pw = fit_model(&n70_pts, FitType::PowerLaw, "PowerLawModel::n70", false)
        .expect("PowerLaw fit failed");
    assert!(
        fit_pw.r_squared >= 0.99,
        "PowerLaw R² = {}",
        fit_pw.r_squared
    );

    let log_pts: Vec<[f64; 2]> = (1..=30)
        .map(|i| {
            let x = i as f64 * 0.5;
            [x, 3.2 * x.ln() - 1.4]
        })
        .collect();
    let fit_ln = fit_model(&log_pts, FitType::Logarithmic, "LogCurve", false)
        .expect("Logarithmic fit failed");
    assert!(fit_ln.r_squared >= 0.99);
}

#[test]
fn log_mode_vs_untransformed_exponential_and_10k_latency() {
    // Heteroscedastic dataset where Log Mode and untransformed LM produce distinct fits
    let noisy = vec![[0.0, 1.0], [1.0, 2.7], [2.0, 7.4], [3.0, 20.1], [4.0, 58.0]];
    let untransformed = fit_model(&noisy, FitType::Exponential, "Exp", false).unwrap();
    let log_mode = fit_model(&noisy, FitType::Exponential, "Exp", true).unwrap();
    assert!(!untransformed.log_mode && log_mode.log_mode);
    assert!(untransformed.rmse <= log_mode.rmse + 1e-9);

    let pts_10k: Vec<[f64; 2]> = (0..10_000)
        .map(|i| {
            let x = i as f64 * 0.001;
            [x, 2.5 * (1.8 * x + 0.3).sin() + 1.1]
        })
        .collect();
    let start = Instant::now();
    let res = fit_model(&pts_10k, FitType::Sinusoidal, "Sin10k", false).expect("10k fit failed");
    let elapsed = start.elapsed();
    assert!(res.r_squared >= 0.99);
    #[cfg(not(debug_assertions))]
    assert!(
        elapsed.as_millis() < 16,
        "release 10k sinusoidal fit exceeded 16ms frame budget: {elapsed:?}"
    );
    assert!(
        elapsed.as_millis() < 150,
        "10k sinusoidal fit took {elapsed:?}"
    );
}
