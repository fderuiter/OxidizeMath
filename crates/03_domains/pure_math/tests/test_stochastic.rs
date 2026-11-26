//! Integration tests for stochastic simulation with VerifiedAllocator.

use pure_math::pure_math::analysis::stochastic::{
    GillespieSolver, StochasticError, StochasticSystem,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct TestModel;
struct TestState {
    x: i32,
}

impl StochasticSystem<TestState> for TestModel {
    #[verified_engine::verified]
    fn propensities(&self, state: &TestState, out: &mut Vec<f64>) {
        if state.x > 0 {
            out.push(1.0);
        }
    }

    #[verified_engine::verified]
    fn react(&self, state: &mut TestState, reaction_index: usize) -> Result<(), StochasticError> {
        if reaction_index == 0 {
            state.x -= 1;
            Ok(())
        } else {
            Err(StochasticError::InvalidReactionIndex(reaction_index))
        }
    }
}

#[test]
fn test_stochastic_step_zero_allocations_with_global_allocator() {
    let rng = oxidize_core::rng::OxidizeRng::default();
    let mut solver = GillespieSolver::new(rng);
    let model = TestModel;
    let mut state = TestState { x: 50 };

    let (_res, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..10 {
            solver.step(&model, &mut state).unwrap();
        }
    });

    assert_eq!(state.x, 40);
    assert_eq!(metrics.memory_loads, 0);
}
