//! Cross-platform deterministic favoritism scoring integration tests with pinned SHA-256 digest.

use domain_applied::applied::favoritism::{
    FavoritismInputs, calculate_favoritism_score_full, calculate_favoritism_score_with_rng,
};
use oxidize_core::rng::{OxidizeRng, TrajectoryDigest};
use pure_math::pure_math::analysis::integration::{ClenshawCurtis, IntegrationResult, Integrator};

const PINNED_FAVORITISM_DIGEST: &str =
    "7da3fafbfe85cf925b73caf553320110b19ea7b917fd23b036200c85f1c47eca";

#[test]
#[verified_engine::verified]
fn test_favoritism_deterministic() {
    let inputs = FavoritismInputs::default();

    let run_ensemble = |seed: u64| -> (Vec<f64>, String) {
        let mut rng = OxidizeRng::new(seed);
        let mut digest = TrajectoryDigest::new();
        let mut scores = Vec::with_capacity(16);
        for _ in 0..16 {
            let score = calculate_favoritism_score_with_rng(&inputs, &mut rng);
            digest.update_f64(score);
            scores.push(score);
        }
        (scores, digest.finish_hex())
    };

    let (scores1, hex1) = run_ensemble(42);
    let (scores2, hex2) = run_ensemble(42);
    let (scores3, hex3) = run_ensemble(999);

    assert_eq!(scores1, scores2, "Scores with same seed must be identical");
    assert_eq!(hex1, hex2);
    assert_ne!(scores1, scores3, "Scores with different seeds should vary");
    assert_ne!(hex1, hex3);
    assert_eq!(hex1, PINNED_FAVORITISM_DIGEST);
}

struct MockIntegrator;

impl Integrator for MockIntegrator {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, _eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        let val = f((min + max) / 2.0) * (max - min);
        IntegrationResult {
            value: val * 2.0,
            error: 0.0,
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_favoritism_integrator_swap() {
    let inputs = FavoritismInputs::default();
    let rng = OxidizeRng::new(42);

    let score_cc = calculate_favoritism_score_full(&inputs, &mut rng.clone(), &ClenshawCurtis);
    let score_mock = calculate_favoritism_score_full(&inputs, &mut rng.clone(), &MockIntegrator);

    assert!(
        (score_cc - score_mock).abs() > 1.0,
        "Mock integrator should yield significantly different result"
    );
}
