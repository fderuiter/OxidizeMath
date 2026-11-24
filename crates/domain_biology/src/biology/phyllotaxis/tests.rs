use super::*;
use crate::biology::lsystem::TurtleConfig;
use crate::error::BotanicalError;
use scientific_metadata::theory::TheoryDescribable;

#[test]
#[verified_engine::verified]
fn test_vogel_coordinates_match_analytical_formula() {
    let scale = 1.5;
    let engine = VogelPhyllotaxisEngine::golden(64, scale).unwrap();
    assert_eq!(engine.points().len(), 64);

    for (n, pt) in engine.points().iter().enumerate() {
        let expected_r = scale * (n as f64).sqrt();
        let expected_theta = (n as f64) * GOLDEN_ANGLE_RAD;
        assert_eq!(pt.index, n);
        assert!((pt.radius - expected_r).abs() < 1e-12);
        assert!((pt.theta - expected_theta).abs() < 1e-12);
        assert!((pt.x - expected_r * expected_theta.cos()).abs() < 1e-12);
        assert!((pt.y - expected_r * expected_theta.sin()).abs() < 1e-12);
    }
}

#[test]
#[verified_engine::verified]
fn test_golden_angle_exhibits_consecutive_fibonacci_parastichy_transitions() {
    let expected_pairs = [
        (13_usize, (5_usize, 8_usize)),
        (25, (8, 13)),
        (60, (13, 21)),
        (200, (21, 34)),
        (550, (34, 55)),
    ];

    let mut engine = VogelPhyllotaxisEngine::golden(13, 1.0).unwrap();
    for (count, (f_lo, f_hi)) in expected_pairs {
        engine.set_parameters(count, GOLDEN_ANGLE_RAD, 1.0).unwrap();
        let analysis = engine.analysis();
        assert!(
            analysis.is_fibonacci_pair,
            "Expected Fibonacci parastichy pair at N={count}, got ({}, {})",
            analysis.primary_count, analysis.secondary_count
        );
        assert!(!analysis.has_radial_spokes);
        assert_eq!(
            (analysis.primary_count, analysis.secondary_count),
            (f_lo, f_hi),
            "Unexpected parastichy counts at N={count}"
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_rational_divergence_spokes_and_inferior_packing_uniformity() {
    let golden = VogelPhyllotaxisEngine::golden(300, 1.0).unwrap();
    let rational = VogelPhyllotaxisEngine::new(
        300,
        DivergenceAnglePreset::RationalTwoFifths.angle_rad(),
        1.0,
    )
    .unwrap();

    assert!(rational.analysis().has_radial_spokes);
    assert_eq!(rational.analysis().primary_count, 5);
    assert!(
        golden.analysis().packing_uniformity > rational.analysis().packing_uniformity + 0.35,
        "Golden-angle packing ({:.3}) must significantly exceed 2/5 spoke packing ({:.3})",
        golden.analysis().packing_uniformity,
        rational.analysis().packing_uniformity
    );
}

#[test]
#[verified_engine::verified]
fn test_max_primordia_cap_4096_and_parastichy_arm_tracing() {
    let mut engine = VogelPhyllotaxisEngine::golden(MAX_PRIMORDIA, 0.8).unwrap();
    assert_eq!(engine.points().len(), MAX_PRIMORDIA);
    assert!(engine.analysis().is_fibonacci_pair);

    let arm = engine.trace_parastichy_arm(3, 21);
    assert_eq!(arm[0], 3);
    assert_eq!(arm[1], 24);
    assert!(*arm.last().unwrap() < MAX_PRIMORDIA);

    let err = VogelPhyllotaxisEngine::golden(MAX_PRIMORDIA + 1, 1.0).unwrap_err();
    assert_eq!(
        err,
        BotanicalError::InvalidPrimordiaCount {
            count: MAX_PRIMORDIA + 1,
            max: MAX_PRIMORDIA
        }
    );
    assert!(!engine.theory_description().is_empty());
    assert!(!engine.theory_citation().contains("[cite:"));
}

#[test]
#[verified_engine::verified]
fn test_nutrient_coupling_and_plastochron_state_progression() {
    let low = NutrientCoupling::new(NutrientKineticsKind::Hill, 0.25, 2.0, 1.0, 3.0).unwrap();
    let high = NutrientCoupling::new(NutrientKineticsKind::Hill, 4.0, 2.0, 1.0, 3.0).unwrap();
    assert!(high.reaction_velocity() > low.reaction_velocity());
    assert!(high.normalized_rate() > low.normalized_rate());

    let base_turtle = TurtleConfig::default();
    let low_turtle = low.scaled_turtle_config(base_turtle);
    let high_turtle = high.scaled_turtle_config(base_turtle);
    assert!(high_turtle.step_length > low_turtle.step_length);

    let mut state = PhyllotaxisState::new(256, 34, PhyllotaxisConfig::default(), high);
    state.step(0.5);
    assert!(state.active_count() > 34);
    let (m, n) = state.detect_parastichy_pair();
    assert!(is_consecutive_fibonacci_pair(m, n));
}
