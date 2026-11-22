//! Global `VerifiedAllocator` and frame-budget verification for `CoupledVortexMorphogenesis`.
#![cfg(all(feature = "physics", feature = "biology"))]

use math_explorer::verified_engine::allocator::VerifiedAllocator;
use math_explorer::verified_engine::engine::VerifiedEngine;
use math_explorer::vortex_morphogenesis::{
    CoupledVortexMorphogenesis, VortexMorphogenesisConfig, VortexRegimePreset,
};
use std::time::Instant;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_coupled_100x50_and_100x100_zero_allocations_with_global_allocator() {
    let mut engine_50 =
        CoupledVortexMorphogenesis::from_preset(VortexRegimePreset::KarmanVortexSpots);

    let mut cfg_100 = VortexMorphogenesisConfig::for_preset(VortexRegimePreset::KarmanVortexSpots);
    cfg_100.width = 100;
    cfg_100.height = 100;
    let mut engine_100 = CoupledVortexMorphogenesis::new(cfg_100);
    engine_100.paint_obstacle_disc(22, 50, 5, true);
    engine_100.paint_emitter_disc(31, 50, 3, true);

    // VerifiedAllocator aborts immediately if any dynamic heap allocation occurs inside run_verified.
    let (_, metrics_50) = VerifiedEngine::run_verified(|| {
        for _ in 0..4 {
            engine_50.step();
        }
    });
    assert!(metrics_50.arithmetic_ops > 0 && metrics_50.memory_loads > 0);
    assert!(engine_50.is_finite());

    let start = Instant::now();
    let (_, metrics_100) = VerifiedEngine::run_verified(|| {
        engine_100.step();
    });
    let elapsed = start.elapsed();
    assert!(metrics_100.arithmetic_ops > 0 && metrics_100.memory_loads > 0);
    assert!(engine_100.is_finite());
    // Debug builds are unoptimized; ensure reasonable bounded runtime even in debug mode
    assert!(
        elapsed.as_millis() < 250,
        "Single coupled 100x100 debug step took {:?}",
        elapsed
    );
}
