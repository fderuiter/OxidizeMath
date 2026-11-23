//! Specialized zero-allocation 1D Markov chain forward-backward message passing and Potts MRF
//! local factor belief updates.

use crate::error::GraphEngineError;
use engine_prob::discrete_posterior_into;

#[inline]
const fn idx2d(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

/// Borrowed parameter view of a discrete hidden Markov chain factor graph.
#[derive(Debug, Clone, Copy)]
pub struct ChainModelView<'a> {
    /// Number of hidden states $N$.
    pub num_states: usize,
    /// Number of discrete emission symbols $M$.
    pub num_obs: usize,
    /// Initial state distribution $\pi$ (`len >= num_states`).
    pub initial: &'a [f64],
    /// Row-major transition matrix $A$ (`len >= num_states * num_states`).
    pub transitions: &'a [f64],
    /// Row-major emission matrix $B$ (`len >= num_states * num_obs`).
    pub emissions: &'a [f64],
}

/// Caller-supplied time-major output and scratch slices for [`chain_forward_backward_into`].
pub struct ChainFbWorkspace<'a> {
    /// Scaled forward messages (`len >= T * num_states`).
    pub alpha: &'a mut [f64],
    /// Scaled backward messages (`len >= T * num_states`).
    pub beta: &'a mut [f64],
    /// Inverse conditional scaling factors $c_t = 1 / \sum_i \tilde{\alpha}_t(i)$ (`len >= T`).
    pub scales: &'a mut [f64],
    /// Normalized single-step state posteriors $\gamma_t(i) = P(X_t = i \mid Y_{1:T})$ (`len >= T * num_states`).
    pub gamma: &'a mut [f64],
}

/// Evaluates local Potts MRF posterior beliefs
/// $P(z_v = k \mid y_v, z_{\mathcal{N}(v)}) \propto L_k(y_v) \pi_k \exp(\beta c_k)$
/// into `posteriors_out` with zero heap allocations.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn potts_neighborhood_posteriors_into(
    likelihoods: &[f64],
    priors: &[f64],
    neighbor_counts: &[usize],
    beta: f64,
    scratch_potentials: &mut [f64],
    posteriors_out: &mut [f64],
) -> Result<(), GraphEngineError> {
    let k = likelihoods.len();
    if k == 0 {
        return Err(GraphEngineError::EmptyGraph {
            context: "potts_neighborhood_posteriors_into",
        });
    }
    if priors.len() != k || neighbor_counts.len() != k {
        return Err(GraphEngineError::DimensionMismatch {
            expected: k,
            actual: priors.len().min(neighbor_counts.len()),
        });
    }
    if scratch_potentials.len() < k || posteriors_out.len() < k {
        return Err(GraphEngineError::BufferTooSmall {
            required: k,
            actual: scratch_potentials.len().min(posteriors_out.len()),
        });
    }
    if !beta.is_finite() {
        return Err(GraphEngineError::InvalidPotential {
            reason: "Potts coupling beta must be finite",
        });
    }
    let max_count = neighbor_counts.iter().copied().max().unwrap_or(0) as f64;
    for i in 0..k {
        let shifted = beta * ((neighbor_counts[i] as f64) - max_count);
        scratch_potentials[i] = priors[i].max(0.0) * libm::exp(shifted);
    }
    discrete_posterior_into(likelihoods, &scratch_potentials[..k], posteriors_out).map_err(
        |_| GraphEngineError::InvalidPotential {
            reason: "non-finite likelihood or prior in Potts update",
        },
    )?;
    debug_assert!(posteriors_out[..k].iter().all(|&p| p >= 0.0));
    Ok(())
}

fn forward_chain_pass(
    model: &ChainModelView<'_>,
    observations: &[usize],
    alpha_out: &mut [f64],
    scales_out: &mut [f64],
) -> Result<f64, GraphEngineError> {
    let n = model.num_states;
    let m = model.num_obs;
    let num_steps = observations.len();
    let y0 = observations[0];
    if y0 >= m {
        return Err(GraphEngineError::InvalidVariable { var: 0, value: y0 });
    }
    let mut s0 = 0.0;
    for i in 0..n {
        let v = model.initial[i] * model.emissions[idx2d(i, y0, m)];
        alpha_out[idx2d(0, i, n)] = v;
        s0 += v;
    }
    if s0 <= 0.0 {
        scales_out.fill(0.0);
        return Ok(f64::NEG_INFINITY);
    }
    let inv_s0 = 1.0 / s0;
    for i in 0..n {
        alpha_out[idx2d(0, i, n)] *= inv_s0;
    }
    scales_out[0] = inv_s0;
    let mut log_likelihood = libm::log(s0);

    for t in 1..num_steps {
        let yt = observations[t];
        if yt >= m {
            return Err(GraphEngineError::InvalidVariable { var: t, value: yt });
        }
        let mut st = 0.0;
        for j in 0..n {
            let mut acc = 0.0;
            for i in 0..n {
                acc += alpha_out[idx2d(t - 1, i, n)] * model.transitions[idx2d(i, j, n)];
            }
            let val = acc * model.emissions[idx2d(j, yt, m)];
            alpha_out[idx2d(t, j, n)] = val;
            st += val;
        }
        if st <= 0.0 {
            scales_out[t..num_steps].fill(0.0);
            return Ok(f64::NEG_INFINITY);
        }
        let inv_st = 1.0 / st;
        for j in 0..n {
            alpha_out[idx2d(t, j, n)] *= inv_st;
        }
        scales_out[t] = inv_st;
        log_likelihood += libm::log(st);
    }
    Ok(log_likelihood)
}

fn backward_and_posteriors_pass(
    model: &ChainModelView<'_>,
    observations: &[usize],
    ws: &mut ChainFbWorkspace<'_>,
) {
    let n = model.num_states;
    let m = model.num_obs;
    let num_steps = observations.len();
    let last_t = num_steps - 1;
    let c_last = ws.scales[last_t];
    for i in 0..n {
        ws.beta[idx2d(last_t, i, n)] = c_last;
    }
    for t in (0..last_t).rev() {
        let y_next = observations[t + 1];
        let ct = ws.scales[t];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += model.transitions[idx2d(i, j, n)]
                    * model.emissions[idx2d(j, y_next, m)]
                    * ws.beta[idx2d(t + 1, j, n)];
            }
            ws.beta[idx2d(t, i, n)] = sum * ct;
        }
    }
    for t in 0..num_steps {
        let mut norm = 0.0;
        for i in 0..n {
            let flat = idx2d(t, i, n);
            let g = ws.alpha[flat] * ws.beta[flat];
            ws.gamma[flat] = g;
            norm += g;
        }
        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            for i in 0..n {
                ws.gamma[idx2d(t, i, n)] *= inv_norm;
            }
        }
    }
}

/// Runs scaled sum-product forward-backward inference on a discrete 1D Markov chain factor graph
/// over caller-supplied time-major buffers (`num_steps * num_states`) with zero heap allocations.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn chain_forward_backward_into(
    model: &ChainModelView<'_>,
    observations: &[usize],
    ws: &mut ChainFbWorkspace<'_>,
) -> Result<f64, GraphEngineError> {
    if model.num_states == 0 || model.num_obs == 0 || observations.is_empty() {
        return Err(GraphEngineError::EmptyGraph {
            context: "chain_forward_backward_into",
        });
    }
    let t_len = observations.len();
    let mat_len = t_len * model.num_states;
    if model.initial.len() < model.num_states
        || model.transitions.len() < model.num_states * model.num_states
        || model.emissions.len() < model.num_states * model.num_obs
    {
        return Err(GraphEngineError::DimensionMismatch {
            expected: model.num_states,
            actual: model.initial.len(),
        });
    }
    if ws.alpha.len() < mat_len
        || ws.beta.len() < mat_len
        || ws.gamma.len() < mat_len
        || ws.scales.len() < t_len
    {
        return Err(GraphEngineError::BufferTooSmall {
            required: mat_len,
            actual: ws.alpha.len().min(ws.beta.len()).min(ws.gamma.len()),
        });
    }
    let ll = forward_chain_pass(model, observations, ws.alpha, ws.scales)?;
    if ll.is_finite() {
        backward_and_posteriors_pass(model, observations, ws);
    } else {
        ws.beta[..mat_len].fill(0.0);
        ws.gamma[..mat_len].fill(0.0);
    }
    Ok(ll)
}

/// Accumulates expected state transition counts $\sum_{t=0}^{T-2} \xi_t(i, j)$ from a completed
/// [`chain_forward_backward_into`] pass into row-major `xi_sum_out` (`len >= num_states * num_states`)
/// with zero heap allocations.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn chain_accumulate_xi_into(
    model: &ChainModelView<'_>,
    observations: &[usize],
    ws: &ChainFbWorkspace<'_>,
    xi_sum_out: &mut [f64],
) -> Result<(), GraphEngineError> {
    let n = model.num_states;
    let m = model.num_obs;
    let nn = n * n;
    if xi_sum_out.len() < nn {
        return Err(GraphEngineError::BufferTooSmall {
            required: nn,
            actual: xi_sum_out.len(),
        });
    }
    xi_sum_out[..nn].fill(0.0);
    if observations.len() <= 1 {
        return Ok(());
    }
    for t in 0..(observations.len() - 1) {
        let y_next = observations[t + 1];
        let mut step_norm = 0.0;
        for i in 0..n {
            let alpha_i = ws.alpha[idx2d(t, i, n)];
            for j in 0..n {
                step_norm += alpha_i
                    * model.transitions[idx2d(i, j, n)]
                    * model.emissions[idx2d(j, y_next, m)]
                    * ws.beta[idx2d(t + 1, j, n)];
            }
        }
        if step_norm <= 0.0 {
            continue;
        }
        let inv_norm = 1.0 / step_norm;
        for i in 0..n {
            let alpha_i = ws.alpha[idx2d(t, i, n)] * inv_norm;
            for j in 0..n {
                let val = alpha_i
                    * model.transitions[idx2d(i, j, n)]
                    * model.emissions[idx2d(j, y_next, m)]
                    * ws.beta[idx2d(t + 1, j, n)];
                xi_sum_out[idx2d(i, j, n)] += val;
            }
        }
    }
    debug_assert!(xi_sum_out[..nn].iter().all(|&v| v >= 0.0));
    Ok(())
}
