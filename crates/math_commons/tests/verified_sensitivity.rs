//! Zero-allocation runtime verification of `math_commons::sensitivity::SensitivityEngine`
//! hot evaluation loops under `VerifiedAllocator`.
#![cfg(feature = "std")]

use math_commons::sensitivity::{
    MAX_SENSITIVITY_BATCH_EVALS, ParameterUncertainty, SensitivityEngine, TornadoBounds,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn sensitivity_tornado_and_monte_carlo_hot_loops_zero_heap_allocations() {
    let params = [
        ("dose", 100.0_f64),
        ("ka", 1.2_f64),
        ("ke", 0.12_f64),
        ("volume", 40.0_f64),
    ];
    let bounds = [
        TornadoBounds::new(50.0, 250.0),
        TornadoBounds::new(0.4, 3.0),
        TornadoBounds::new(0.03, 0.4),
        TornadoBounds::new(20.0, 80.0),
    ];
    let uncertainties = [
        ParameterUncertainty::new(10.0, 50.0, 250.0),
        ParameterUncertainty::new(0.15, 0.4, 3.0),
        ParameterUncertainty::new(0.02, 0.03, 0.4),
        ParameterUncertainty::new(4.0, 20.0, 80.0),
    ];

    let mut engine = SensitivityEngine::new();
    // Initial schema setup allocates backing vectors once outside hot loops.
    engine.prepare_tornado(&params, &bounds);
    engine.prepare_monte_carlo(&params, &uncertainties, 0xCAFE_BEEF, 64);

    let eval_slice = |v: &[f64]| -> f64 { (v[0] / v[3]) * (v[1] / (v[1] + v[2])) };

    VerifiedEngine::run_verified(|| {
        while !engine.is_converged() {
            engine.compute_slice(&params, eval_slice);
        }
        while !engine.is_tornado_converged() {
            engine.step_tornado_slice(&params, &bounds, MAX_SENSITIVITY_BATCH_EVALS, eval_slice);
        }
        while !engine.is_monte_carlo_converged() {
            engine.step_monte_carlo_slice(
                &params,
                &uncertainties,
                0xCAFE_BEEF,
                64,
                MAX_SENSITIVITY_BATCH_EVALS,
                eval_slice,
            );
        }
    });

    assert_eq!(engine.metrics.len(), 4);
    assert_eq!(engine.tornado_intervals().len(), 4);
    assert_eq!(engine.monte_carlo_summary().unwrap().sample_count, 64);
}
