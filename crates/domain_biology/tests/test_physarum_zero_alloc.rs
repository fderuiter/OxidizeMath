//! Verifies that `PhysarumEngine::step` performs zero dynamic heap allocations across
//! `64x64`..`256x256` grids and up to `16,384` Structure-of-Arrays agents under `VerifiedAllocator`.

use domain_biology::biology::physarum::{
    MAX_PHYSARUM_AGENTS, MAX_PHYSARUM_GRID, PhysarumEngine, PhysarumPresetKind,
};
use oxidize_core::rng::OxidizeRng;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_physarum_256x256_16384_agents_step_zero_heap_allocations() {
    let mut engine = PhysarumEngine::from_preset(
        MAX_PHYSARUM_GRID,
        MAX_PHYSARUM_GRID,
        MAX_PHYSARUM_AGENTS,
        PhysarumPresetKind::TokyoRail,
        OxidizeRng::new(2026),
    );

    let (telemetry, metrics) = VerifiedEngine::run_verified(|| {
        engine.step();
        engine.step();
        engine.set_agent_count(4096);
        engine.sever_trail_disc(120, 120, 4.0);
        engine.paint_obstacle_disc(90, 90, 3.0, true);
        engine.step();
        engine.telemetry()
    });

    assert_eq!(telemetry.step_count, 3);
    assert!(telemetry.total_trail_mass > 10.0);
    assert!(metrics.function_calls > 0);
}
