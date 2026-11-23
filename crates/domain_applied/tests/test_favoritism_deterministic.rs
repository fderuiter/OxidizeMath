//! Integration tests for test favoritism deterministic.
use domain_applied::applied::favoritism::{
    FavoritismInputs, calculate_favoritism_score_full, calculate_favoritism_score_with_rng,
};
use pure_math::pure_math::analysis::integration::{ClenshawCurtis, IntegrationResult, Integrator};

#[test]
#[verified_engine::verified]
fn test_favoritism_deterministic() {
    let inputs = FavoritismInputs::default();

    // Seed 1
    let mut rng1 = oxidize_core::rng::OxidizeRng::default();
    let score1 = calculate_favoritism_score_with_rng(&inputs, &mut rng1);

    // Seed 2 (same seed)
    let mut rng2 = oxidize_core::rng::OxidizeRng::default();
    let score2 = calculate_favoritism_score_with_rng(&inputs, &mut rng2);

    // Seed 3 (different seed)
    let mut rng3 = oxidize_core::rng::OxidizeRng::new(999);
    let score3 = calculate_favoritism_score_with_rng(&inputs, &mut rng3);

    // Check consistency
    assert_eq!(score1, score2, "Scores with same seed must be identical");

    // Check variation (extremely unlikely to be identical with different seeds due to r factor)
    assert_ne!(score1, score3, "Scores with different seeds should vary");
}

struct MockIntegrator;

impl Integrator for MockIntegrator {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, _eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        // Return a bogus value to prove it's being used.
        // For example, return 0.0 regardless of function.
        // But if we return 0.0, we might get division by zero in the score calculation.
        // The score calculation has `denominator = if integral < EPSILON { 1.0 } else { integral }`.

        // Let's return a value that is clearly different from the correct integral (which is usually > 0).
        // The real integral is approx Time * Value.
        // Let's return Time * Value * 2.0.

        let val = f((min + max) / 2.0) * (max - min);
        IntegrationResult {
            value: val * 2.0, // Double the result
            error: 0.0,
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_favoritism_integrator_swap() {
    let inputs = FavoritismInputs::default();
    let rng = oxidize_core::rng::OxidizeRng::default();

    // Use default Clenshaw-Curtis
    let score_cc = calculate_favoritism_score_full(&inputs, &mut rng.clone(), &ClenshawCurtis);

    // Use Mock Integrator
    let score_mock = calculate_favoritism_score_full(&inputs, &mut rng.clone(), &MockIntegrator);

    println!("CC Score: {}, Mock Score: {}", score_cc, score_mock);

    // Since Mock doubles the integrals (numerator and denominator), the ratio might stay similar?
    // Numerator has proximity_integral * emotional_integral (doubled * doubled = 4x).
    // Denominator has sibling_integral (doubled = 2x).
    // Result should be approx 2x.

    // So they should be different.
    assert!(
        (score_cc - score_mock).abs() > 1.0,
        "Mock integrator should yield significantly different result"
    );
}
