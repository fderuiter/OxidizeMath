//! Verifies that `LeniaEngine::step` performs zero dynamic heap allocations on a `256x256`
//! periodic grid under `verified_engine::allocator::VerifiedAllocator`.

use domain_biology::biology::lenia::{LeniaEngine, LeniaSpecies};
use domain_biology::biology::morphogenesis::GrayScottKinetics;
use domain_biology::biology::morphogenesis::ReactionKinetics;
use oxidize_core::rng::OxidizeRng;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_lenia_256x256_step_zero_heap_allocations() {
    let mut rng = OxidizeRng::new(2026);
    let mut engine = LeniaEngine::from_species(256, 256, LeniaSpecies::Orbium, 0.02, &mut rng);

    let (telemetry, metrics) = VerifiedEngine::run_verified(|| {
        engine.step();
        engine.step();
        engine.telemetry()
    });

    assert_eq!(telemetry.step_count, 2);
    assert!(telemetry.total_mass > 20.0);
    assert!(metrics.function_calls > 0);
}

#[test]
#[verified_engine::verified]
fn test_gray_scott_kinetics_presets_and_zero_alloc_reaction() {
    let mitosis = GrayScottKinetics::mitosis();
    let coral = GrayScottKinetics::coral();
    assert_ne!(mitosis, coral);

    let (rates, _) = VerifiedEngine::run_verified(|| mitosis.reaction([0.5, 0.25]));
    assert!(rates[0].is_finite() && rates[1].is_finite());
    assert!(rates[0] < 0.0);
    assert!(rates[1] > 0.0);
}
