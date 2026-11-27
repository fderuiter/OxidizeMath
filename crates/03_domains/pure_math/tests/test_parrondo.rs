//! Integration tests verifying zero heap allocations during Parrondo ensemble stepping.

use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::statistics::markov::parrondo::{
    ParrondoEnsemble, ParrondoGame, ParrondoParams, ParrondoSchedule,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_parrondo_ensemble_zero_allocations_under_verified_allocator() {
    let params = ParrondoParams::canonical();
    let schedule = ParrondoSchedule::parse("ABB").unwrap();
    let mut ensemble = ParrondoEnsemble::with_capacity(256, 150).unwrap();
    let mut rng = OxidizeRng::new(2026);

    let ((sched_mean, mix_mean, pure_a_mean), _metrics) = VerifiedEngine::run_verified(|| {
        let s_mean = ensemble.simulate_schedule(&params, &schedule, 150, &mut rng);
        let m_mean = ensemble.simulate_mixture(&params, 150, &mut rng);
        ensemble.reset();
        let mut a_mean = 0.0;
        for _ in 0..150 {
            a_mean = ensemble.step_game(&params, ParrondoGame::GameA, &mut rng);
        }
        (s_mean, m_mean, a_mean)
    });

    assert!(
        sched_mean > 2.0,
        "Expected positive ABB mean wealth, got {sched_mean}"
    );
    assert!(mix_mean.is_finite());
    assert!(pure_a_mean < 2.0);
    assert_eq!(ensemble.mean_trajectory().len(), 151);
}

// [cite:clinical_trials_statistics]
