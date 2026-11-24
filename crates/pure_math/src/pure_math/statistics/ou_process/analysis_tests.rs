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
fn test_monte_carlo_forecast_and_safety_guards() {
    let params = OuParams::from_values(0.5, 1.0, 0.2).unwrap();
    let dt = TimeStep::new(0.01).unwrap();
    let analyzer = OuAnalyzer::new(params, dt);

    let mut rng = oxidize_core::rng::OxidizeRng::default();
    let stats = analyzer.monte_carlo_forecast(0.6, 1.0, 1000, Some(0.5), &mut rng);

    assert!((stats.mean_final - 0.5).abs() < 0.1);
    assert!(stats.std_final > 0.0);
    assert!(stats.mean_std_error > 0.0);
    assert!(stats.prob_above_threshold.is_some());
    assert!(stats.prob_above_threshold_std_error.unwrap() > 0.0);

    // Zero-paths and NaN inputs never panic and return fallible errors via try_monte_carlo_forecast
    let zero_stats = analyzer.monte_carlo_forecast(0.6, 1.0, 0, Some(0.5), &mut rng);
    assert_eq!(zero_stats.mean_std_error, 0.0);
    assert!(
        analyzer
            .try_monte_carlo_forecast(0.6, 1.0, 0, Some(0.5), &mut rng)
            .is_err()
    );

    let nan_stats = analyzer.monte_carlo_forecast(f64::NAN, 1.0, 64, Some(0.5), &mut rng);
    assert!(nan_stats.mean_final.is_finite());
    assert!(
        analyzer
            .try_monte_carlo_forecast(f64::NAN, 1.0, 64, Some(0.5), &mut rng)
            .is_err()
    );
    assert!(
        analyzer
            .try_monte_carlo_forecast(0.6, 1.0, 64, Some(f64::NAN), &mut rng)
            .is_err()
    );
}

#[test]
#[verified_engine::verified]
fn test_ou_variance_reduction_beats_iid_baseline() {
    let params = OuParams::from_values(0.5, 1.0, 0.25).unwrap();
    let dt = TimeStep::new(0.02).unwrap();
    let analyzer = OuAnalyzer::new(params, dt);

    let mut rng_iid = oxidize_core::rng::OxidizeRng::new(0x1234_5678);
    let iid = analyzer
        .try_monte_carlo_forecast(0.5, 1.0, 2_000, Some(0.5), &mut rng_iid)
        .unwrap();

    let mut rng_anti = oxidize_core::rng::OxidizeRng::new(0x1234_5678);
    let anti = analyzer
        .try_monte_carlo_forecast_variance_reduced(
            0.5,
            1.0,
            2_000,
            Some(0.5),
            OuVarianceReduction::Antithetic,
            &mut rng_anti,
        )
        .unwrap();
    assert!(anti.mean_std_error < iid.mean_std_error);

    let mut rng_cv = oxidize_core::rng::OxidizeRng::new(0x1234_5678);
    let cv = analyzer
        .try_monte_carlo_forecast_variance_reduced(
            0.5,
            1.0,
            2_000,
            Some(0.5),
            OuVarianceReduction::ControlVariate,
            &mut rng_cv,
        )
        .unwrap();
    assert!(
        cv.prob_above_threshold_std_error.unwrap() < iid.prob_above_threshold_std_error.unwrap()
    );
}

#[test]
#[verified_engine::verified]
fn test_comeback_probability_and_estimation() {
    let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
    let dt = TimeStep::new(0.01).unwrap();
    let analyzer = OuAnalyzer::new(params, dt);

    let mut rng = oxidize_core::rng::OxidizeRng::default();
    let prob = analyzer.comeback_probability(0.3, 0.5, 1.0, 1000, &mut rng);
    assert!((0.0..=1.0).contains(&prob) && prob > 0.0);

    let true_params = OuParams::from_values(0.5, 1.0, 0.2).unwrap();
    let solver = EulerMaruyama::new(true_params, TimeStep::new(0.02).unwrap());
    let mut rng2 = oxidize_core::rng::OxidizeRng::new(12345);
    let trajectory = solver.simulate(0.5, 2_500, &mut rng2);
    let estimated = estimate_ou_params(&trajectory, 0.02).unwrap();
    assert!((estimated.mu.value() - 0.5).abs() < 0.15);
    assert!(estimated.theta.value() > 0.0);
    assert!((estimated.sigma.value() - 0.2).abs() / 0.2 < 0.12);
    assert!(estimate_ou_params(&[0.5, 0.6], 1.0).is_err());

    let mut ws = math_commons::stochastic::SolverWorkspace::new(0, 8);
    let ws_stats =
        summarize_terminal_values_with_workspace(&mut ws, &[0.2, 0.4, 0.6, 0.8], Some(0.5))
            .unwrap();
    assert!((ws_stats.mean_final - 0.5).abs() < 1e-12);
    assert!(ws_stats.mean_std_error > 0.0);
    assert_eq!(ws_stats.prob_above_threshold, Some(0.5));
    assert!(ws_stats.prob_above_threshold_std_error.unwrap() > 0.0);
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
