use super::*;

#[test]
#[verified_engine::verified]
fn test_momentum_type_classification() {
    assert_eq!(MomentumType::classify(3.0), MomentumType::FlashInPan);
    assert_eq!(MomentumType::classify(1.0), MomentumType::Normal);
    assert_eq!(MomentumType::classify(0.3), MomentumType::HeatCheck);
}

#[test]
#[verified_engine::verified]
fn test_ou_analyzer_creation() {
    let params = OuParams::from_values(0.45, 1.0, 0.15).unwrap();
    let dt = TimeStep::new(0.01).unwrap();
    let analyzer = OuAnalyzer::new(params, dt);

    assert_eq!(analyzer.momentum_type(), MomentumType::Normal);
}

#[test]
#[verified_engine::verified]
fn test_monte_carlo_forecast() {
    let params = OuParams::from_values(0.5, 1.0, 0.2).unwrap();
    let dt = TimeStep::new(0.01).unwrap();
    let analyzer = OuAnalyzer::new(params, dt);

    let mut rng = oxidize_core::rng::OxidizeRng::default();
    let stats = analyzer.monte_carlo_forecast(0.6, 1.0, 1000, Some(0.5), &mut rng);

    // Mean should be close to long-term mean
    assert!((stats.mean_final - 0.5).abs() < 0.1);
    assert!(stats.std_final > 0.0);
    assert!(stats.prob_above_threshold.is_some());
}

#[test]
#[verified_engine::verified]
fn test_comeback_probability() {
    let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
    let dt = TimeStep::new(0.01).unwrap();
    let analyzer = OuAnalyzer::new(params, dt);

    let mut rng = oxidize_core::rng::OxidizeRng::default();
    let prob = analyzer.comeback_probability(0.3, 0.5, 1.0, 1000, &mut rng);

    // Should be between 0 and 1
    assert!((0.0..=1.0).contains(&prob));
    // Should be non-zero since 0.5 is the mean
    assert!(prob > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_estimate_ou_params() {
    let true_params = OuParams::from_values(0.5, 1.0, 0.2).unwrap();
    let dt = TimeStep::new(0.02).unwrap();
    let solver = EulerMaruyama::new(true_params, dt);

    let mut rng = oxidize_core::rng::OxidizeRng::new(12345);
    let trajectory = solver.simulate(0.5, 2_500, &mut rng);

    let estimated = estimate_ou_params(&trajectory, 0.02).unwrap();
    assert!((estimated.mu.value() - 0.5).abs() < 0.15);
    assert!(estimated.theta.value() > 0.0);
    assert!(
        (estimated.sigma.value() - 0.2).abs() / 0.2 < 0.12,
        "expected sigma ~0.2, got {}",
        estimated.sigma.value()
    );
}

#[test]
#[verified_engine::verified]
fn test_estimate_ou_params_recovers_sigma_across_slow_and_fast_theta() {
    let slow_params = OuParams::from_values(1.2, 0.05, 0.25).unwrap();
    let fast_params = OuParams::from_values(-0.8, 5.0, 0.35).unwrap();
    let dt = TimeStep::new(0.01).unwrap();

    let mut rng_slow = oxidize_core::rng::OxidizeRng::new(2026_1123);
    let slow_traj = EulerMaruyama::new(slow_params, dt).simulate(1.2, 4_000, &mut rng_slow);
    let est_slow = estimate_ou_params(&slow_traj, 0.01).unwrap();
    assert!(
        (est_slow.sigma.value() - 0.25).abs() / 0.25 < 0.10,
        "slow theta sigma={} expected ~0.25",
        est_slow.sigma.value()
    );

    let mut rng_fast = oxidize_core::rng::OxidizeRng::new(2026_1124);
    let fast_traj = EulerMaruyama::new(fast_params, dt).simulate(-0.8, 4_000, &mut rng_fast);
    let est_fast = estimate_ou_params(&fast_traj, 0.01).unwrap();
    assert!((est_fast.mu.value() - (-0.8)).abs() < 0.10);
    assert!((est_fast.theta.value() - 5.0).abs() / 5.0 < 0.18);
    assert!(
        (est_fast.sigma.value() - 0.35).abs() / 0.35 < 0.10,
        "fast theta sigma={} expected ~0.35",
        est_fast.sigma.value()
    );
}

#[test]
#[verified_engine::verified]
fn test_estimate_ou_params_insufficient_data() {
    let observations = vec![0.5, 0.6];
    let result = estimate_ou_params(&observations, 1.0);
    assert!(result.is_err());
}

#[test]
#[verified_engine::verified]
fn test_estimate_ou_params_recovers_sigma_within_five_percent() {
    let dt_val = 0.02;
    let dt = TimeStep::new(dt_val).unwrap();
    let true_sigma = 0.35;

    for &theta in &[4.0_f64, 0.10_f64] {
        let params = OuParams::from_values(1.0, theta, true_sigma).unwrap();
        let solver = super::super::ExactOuSolver::new(params, dt);
        let mut rng = oxidize_core::rng::OxidizeRng::new(2026);
        let traj = solver.simulate(1.0, 8_000, &mut rng);

        let est = estimate_ou_params(&traj, dt_val).unwrap();
        let rel_err = (est.sigma.value() - true_sigma).abs() / true_sigma;
        assert!(
            rel_err < 0.05,
            "theta={theta}: estimated sigma {} exceeded 5% relative error vs {true_sigma} (rel_err={rel_err})",
            est.sigma.value()
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_monte_carlo_forecast_zero_paths_nan_and_standard_errors() {
    let params = OuParams::from_values(0.5, 1.0, 0.2).unwrap();
    let dt = TimeStep::new(0.01).unwrap();
    let analyzer = OuAnalyzer::new(params, dt);
    let mut rng = oxidize_core::rng::OxidizeRng::new(7);

    let zero_stats = analyzer.monte_carlo_forecast(0.6, 1.0, 0, Some(0.5), &mut rng);
    assert!(zero_stats.mean_final.is_finite());
    assert_eq!(zero_stats.standard_error, 0.0);
    assert_eq!(zero_stats.prob_above_threshold, Some(0.0));
    assert!(
        analyzer
            .try_monte_carlo_forecast(0.6, 1.0, 0, None, &mut rng)
            .is_err()
    );

    let nan_stats = analyzer.monte_carlo_forecast(f64::NAN, 1.0, 64, Some(f64::NAN), &mut rng);
    assert!(nan_stats.mean_final.is_finite());
    assert_eq!(nan_stats.standard_error, 0.0);
    assert!(nan_stats.prob_above_threshold.is_none());
    assert!(
        analyzer
            .try_monte_carlo_forecast(f64::NAN, 1.0, 64, None, &mut rng)
            .is_err()
    );

    let stats = analyzer.monte_carlo_forecast(0.5, 1.0, 500, Some(0.5), &mut rng);
    assert!(stats.standard_error > 0.0);
    assert!(stats.ci_95_lower < stats.mean_final && stats.mean_final < stats.ci_95_upper);
    assert!(stats.prob_standard_error.unwrap() > 0.0);
}

// [cite:clinical_trials_statistics]
