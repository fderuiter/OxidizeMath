//! Zero-allocation Compressed Sparse Row (CSR) synaptic coupling for Hodgkin-Huxley circuits.

use super::neuron::HodgkinHuxleyNeuron;

/// Maximum supported neuron count for interactive CSR Hodgkin-Huxley circuits.
pub const MAX_CSR_HH_NEURONS: usize = 32;

/// Steps a coupled Hodgkin-Huxley neural circuit over a weighted CSR graph (`u -> v` synapses)
/// using preallocated `next_inputs` with zero dynamic heap allocations.
///
/// Seeds `next_inputs[i]` from `external_inputs` (defaulting missing entries to `0.0`), then for
/// every presynaptic neuron `u` exceeding `spike_threshold_mv`, adds `synaptic_gain * w` along each
/// outgoing CSR synapse `(u -> v, w)` before advancing every neuron by `dt_ms`.
///
/// # Examples
///
/// ```
/// # use domain_biology::__doctest_facade as math_explorer;
/// use math_explorer::biology::neuroscience::{
///     HodgkinHuxleyNeuron, step_csr_hodgkin_huxley_network,
/// };
///
/// // Directed 2-neuron synapse 0 -> 1 with weight 1.0:
/// let mut neurons = vec![HodgkinHuxleyNeuron::new(20.0), HodgkinHuxleyNeuron::new(-65.0)];
/// let external = [10.0_f64, 0.0];
/// let mut next_inputs = vec![0.0; 2];
/// let row_ptr = [0usize, 1, 1];
/// let col_idx = [1usize];
/// let weights = [1.0_f64];
/// step_csr_hodgkin_huxley_network(
///     &mut neurons,
///     &external,
///     &mut next_inputs,
///     &row_ptr,
///     &col_idx,
///     &weights,
///     15.0,
///     0.0,
///     0.05,
/// );
/// assert!(neurons[1].v() > -65.0);
/// ```
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn step_csr_hodgkin_huxley_network(
    neurons: &mut [HodgkinHuxleyNeuron],
    external_inputs: &[f64],
    next_inputs: &mut [f64],
    row_ptr: &[usize],
    col_idx: &[usize],
    weights: &[f64],
    synaptic_gain: f64,
    spike_threshold_mv: f64,
    dt_ms: f64,
) {
    let n = neurons.len().min(next_inputs.len());
    if n == 0 || row_ptr.len() < n + 1 || col_idx.len() != weights.len() || !dt_ms.is_finite() {
        return;
    }
    for (i, slot) in next_inputs.iter_mut().enumerate().take(n) {
        let ext = external_inputs.get(i).copied().unwrap_or(0.0);
        *slot = if ext.is_finite() { ext } else { 0.0 };
    }

    for u in 0..n {
        if neurons[u].v() > spike_threshold_mv {
            let start = row_ptr[u].min(col_idx.len());
            let end = row_ptr[u + 1].min(col_idx.len());
            for idx in start..end {
                let v = col_idx[idx];
                if v < n {
                    let w = weights[idx];
                    if w.is_finite() {
                        next_inputs[v] += synaptic_gain * w;
                    }
                }
            }
        }
    }

    for i in 0..n {
        neurons[i].update(dt_ms, next_inputs[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_csr_hodgkin_huxley_network_zero_alloc_1000_steps() {
        let mut neurons = vec![
            HodgkinHuxleyNeuron::new(25.0),
            HodgkinHuxleyNeuron::new(-65.0),
            HodgkinHuxleyNeuron::new(-65.0),
        ];
        let external = [10.0_f64, 0.0, 0.0];
        let mut next_inputs = vec![0.0; 3];
        let row_ptr = [0usize, 1, 2, 3];
        let col_idx = [1usize, 2, 0];
        let weights = [1.0_f64, 1.0, 1.0];

        let (v1, metrics) = VerifiedEngine::run_verified(|| {
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
            neurons[1].v()
        });
        assert!(v1.is_finite());
        assert!(metrics.function_calls >= 1_000);
    }
}
