//! Verifies that `GraphReplicatorFitness::fitness` performs zero dynamic heap allocations across
//! 1,000 evaluations under `verified_engine::allocator::VerifiedAllocator`.

use domain_applied::applied::game_theory::evolutionary::{
    FitnessStrategy, GraphReplicatorFitness, GraphUpdateRule,
};
use nalgebra::{DMatrix, DVector};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_graph_replicator_fitness_1000_steps_zero_heap_allocations() {
    let payoff = DMatrix::from_row_slice(2, 2, &[3.0, -1.0, 4.0, 0.0]);
    let row_ptr = [0usize, 3, 6, 9, 12];
    let fitness =
        GraphReplicatorFitness::from_csr(payoff, 4, &row_ptr, GraphUpdateRule::DeathBirth)
            .expect("square payoff matrix");
    let mut state = DVector::from_vec(vec![0.35, 0.65]);
    let mut out = DVector::zeros(2);

    let (final_c, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..1_000 {
            fitness.fitness(&state, &mut out);
            let mean_f = state[0] * out[0] + state[1] * out[1];
            state[0] = (state[0] + 0.005 * state[0] * (out[0] - mean_f)).clamp(1e-6, 1.0 - 1e-6);
            state[1] = 1.0 - state[0];
        }
        state[0]
    });

    assert!(
        final_c > 0.9,
        "Ohtsuki-Nowak Death-Birth with b/c = 4 > k = 3 must favor cooperation, got {final_c}"
    );
    assert!(metrics.function_calls > 0);
}
