//! Verifies that `step_csr_hodgkin_huxley_network` performs zero dynamic heap allocations across
//! 1,000 steps on a 32-neuron CSR circuit under `verified_engine::allocator::VerifiedAllocator`.

use domain_biology::biology::neuroscience::{
    HodgkinHuxleyNeuron, MAX_CSR_HH_NEURONS, step_csr_hodgkin_huxley_network,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_csr_hodgkin_huxley_32_neurons_1000_steps_zero_heap_allocations() {
    let n = MAX_CSR_HH_NEURONS;
    let mut neurons: Vec<HodgkinHuxleyNeuron> =
        (0..n).map(|_| HodgkinHuxleyNeuron::new(-65.0)).collect();
    let mut external = vec![0.0_f64; n];
    external[0] = 14.0;
    let mut next_inputs = vec![0.0_f64; n];

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(2 * n);
    let mut weights = Vec::with_capacity(2 * n);
    row_ptr.push(0);
    for u in 0..n {
        col_idx.push((u + 1) % n);
        weights.push(1.8);
        col_idx.push((u + n - 1) % n);
        weights.push(-0.6);
        row_ptr.push(col_idx.len());
    }

    let (v0, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..1_000 {
            step_csr_hodgkin_huxley_network(
                &mut neurons,
                &external,
                &mut next_inputs,
                &row_ptr,
                &col_idx,
                &weights,
                15.0,
                0.0,
                0.05,
            );
        }
        neurons[0].v()
    });

    assert!(v0.is_finite());
    assert!(neurons.iter().all(|cell| cell.v().is_finite()));
    assert!(metrics.function_calls > 0);
}
