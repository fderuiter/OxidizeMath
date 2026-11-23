use super::*;

#[test]
#[verified_engine::verified]
fn test_golden_angle_produces_consecutive_fibonacci_parastichy_pairs() {
    for count in [30usize, 60, 120, 220, 400] {
        let (m, n) = detect_parastichy_spirals(count, GOLDEN_ANGLE_DEG);
        assert!(
            is_consecutive_fibonacci_pair(m, n),
            "Expected consecutive Fibonacci pair for count={count}, got ({m}, {n})"
        );
        assert!(
            m >= 3,
            "Spiral count too small for count={count}: ({m}, {n})"
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_lucas_angle_diverges_from_fibonacci_spirals() {
    let (m, n) = detect_parastichy_spirals(180, LUCAS_ANGLE_DEG);
    assert!(
        !is_consecutive_fibonacci_pair(m, n),
        "Lucas angle should yield Lucas parastichy spirals, got ({m}, {n})"
    );
    // Classic Lucas sequence pairs: (7, 11), (11, 18), (18, 29)
    assert!(
        matches!((m, n), (7, 11) | (11, 18) | (18, 29)),
        "Unexpected Lucas spiral pair ({m}, {n})"
    );
}

#[test]
#[verified_engine::verified]
fn test_nutrient_coupling_modulates_phyllotaxis_initiation_and_lsystem_scale() {
    let low_nutrient =
        NutrientCoupling::new(NutrientKineticsKind::Hill, 0.3, 1.0, 1.0, 3.0).expect("valid hill");
    let high_nutrient =
        NutrientCoupling::new(NutrientKineticsKind::Hill, 4.0, 1.0, 1.0, 3.0).expect("valid hill");

    assert!(high_nutrient.reaction_velocity() > low_nutrient.reaction_velocity() * 5.0);

    let mut state_low = PhyllotaxisState::new(512, 10, PhyllotaxisConfig::default(), low_nutrient);
    let mut state_high =
        PhyllotaxisState::new(512, 10, PhyllotaxisConfig::default(), high_nutrient);

    for _ in 0..20 {
        state_low.step(0.1);
        state_high.step(0.1);
    }
    assert!(
        state_high.active_count() > state_low.active_count(),
        "High substrate must initiate more primordia ({} vs {})",
        state_high.active_count(),
        state_low.active_count()
    );

    let mut lsys = LSystemEngine::default();
    let cfg = LSystemConfig::default();
    lsys.generate(cfg, &low_nutrient);
    let low_max_y = lsys.segments().iter().map(|s| s[3]).fold(0.0_f64, f64::max);

    lsys.generate(cfg, &high_nutrient);
    let high_max_y = lsys.segments().iter().map(|s| s[3]).fold(0.0_f64, f64::max);

    assert!(!lsys.segments().is_empty());
    assert!(
        high_max_y > low_max_y,
        "Higher nutrient saturation should elongate shoot ({high_max_y} vs {low_max_y})"
    );
}
