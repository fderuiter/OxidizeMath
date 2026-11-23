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

    let solver = EulerMaruyama::new(params, TimeStep::new(0.1).unwrap());
    let mut rng2 = oxidize_core::rng::OxidizeRng::new(12345);
    let trajectory = solver.simulate(0.6, 100, &mut rng2);
    let estimated = estimate_ou_params(&trajectory, 0.1).unwrap();
    assert!((estimated.mu.value() - 0.5).abs() < 0.35);
    assert!(estimate_ou_params(&[0.5, 0.6], 1.0).is_err());
}
