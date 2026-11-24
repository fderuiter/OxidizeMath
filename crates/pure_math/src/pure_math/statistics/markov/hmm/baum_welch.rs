//! Zero-allocation Baum-Welch Expectation-Maximization (EM) parameter estimation for HMMs.
// [cite:clinical_trials_statistics]

use super::baum_welch_types::{BaumWelchConfig, BaumWelchReport, BaumWelchWorkspace};
use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use nalgebra::RealField;
use num_traits::ToPrimitive;

fn validate_bw_inputs<T: RealField + Copy + ToPrimitive>(
    hmm: &HiddenMarkovModel<T>,
    observations: &[usize],
    config: &BaumWelchConfig<T>,
    workspace: &BaumWelchWorkspace<T>,
    history: &[T],
) -> Result<(), MarkovError> {
    BaumWelchConfig::new(
        config.max_iterations,
        config.tolerance,
        config.dirichlet_alpha,
    )?;
    if observations.is_empty() {
        return Err(MarkovError::InvalidObservation {
            reason: "Observation sequence is empty".to_string(),
        });
    }
    if history.len() < config.max_iterations {
        return Err(MarkovError::Math(MathError::DimensionMismatch {
            expected: Dimension(config.max_iterations),
            actual: Dimension(history.len()),
        }));
    }
    if workspace.num_states < hmm.num_states
        || workspace.num_observations < hmm.num_observations
        || workspace.max_len < observations.len()
    {
        return Err(MarkovError::Math(MathError::DimensionMismatch {
            expected: Dimension(observations.len()),
            actual: Dimension(workspace.max_len),
        }));
    }
    for &obs in observations {
        if obs >= hmm.num_observations {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation index out of bounds".to_string(),
            });
        }
    }
    Ok(())
}

fn run_forward_into<T: RealField + Copy + ToPrimitive>(
    hmm: &HiddenMarkovModel<T>,
    observations: &[usize],
    ws: &mut BaumWelchWorkspace<T>,
) -> T {
    let n = hmm.num_states;
    let zero = T::zero();
    let neg_inf = T::from_f64(f64::NEG_INFINITY)
        .or_else(T::min_value)
        .unwrap_or_else(|| zero.ln());
    let y0 = observations[0];
    let mut scale0 = zero;
    for i in 0..n {
        let v = hmm.initial[i] * hmm.emissions[(i, y0)];
        ws.alpha[i] = v;
        scale0 += v;
    }
    if scale0 <= zero {
        return neg_inf;
    }
    for i in 0..n {
        ws.alpha[i] /= scale0;
    }
    ws.scales[0] = scale0;
    let mut ll = scale0.ln();

    let mut prev_off = 0;
    for (step_idx, &y_t) in observations.iter().enumerate().skip(1) {
        let curr_off = prev_off + n;
        let mut scale_t = zero;
        for j in 0..n {
            let mut sum = zero;
            for i in 0..n {
                sum += ws.alpha[prev_off + i] * hmm.transitions[(i, j)];
            }
            let v = sum * hmm.emissions[(j, y_t)];
            ws.alpha[curr_off + j] = v;
            scale_t += v;
        }
        if scale_t <= zero {
            return neg_inf;
        }
        for j in 0..n {
            ws.alpha[curr_off + j] /= scale_t;
        }
        ws.scales[step_idx] = scale_t;
        ll += scale_t.ln();
        prev_off = curr_off;
    }
    ll
}

fn init_backward_accumulators<T: RealField + Copy + ToPrimitive>(
    hmm: &HiddenMarkovModel<T>,
    observations: &[usize],
    ws: &mut BaumWelchWorkspace<T>,
) -> usize {
    let n = hmm.num_states;
    let m = hmm.num_observations;
    let t_max = observations.len();
    let zero = T::zero();
    let one = T::one();

    ws.init_gamma[..n].fill(zero);
    ws.gamma_trans_sum[..n].fill(zero);
    ws.gamma_all_sum[..n].fill(zero);
    ws.xi_sum[..n * n].fill(zero);
    ws.emission_sum[..n * m].fill(zero);
    ws.beta_next[..n].fill(one);

    let last_off = (t_max - 1) * n;
    let y_last = observations[t_max - 1];
    let mut emit_off = 0;
    for i in 0..n {
        let g = ws.alpha[last_off + i];
        ws.gamma_all_sum[i] += g;
        ws.emission_sum[emit_off + y_last] += g;
        if t_max == 1 {
            ws.init_gamma[i] = g;
        }
        emit_off += m;
    }
    last_off
}

fn accumulate_backward_stats<T: RealField + Copy + ToPrimitive>(
    hmm: &HiddenMarkovModel<T>,
    observations: &[usize],
    ws: &mut BaumWelchWorkspace<T>,
) {
    let n = hmm.num_states;
    let m = hmm.num_observations;
    let t_max = observations.len();
    let zero = T::zero();
    let one = T::one();
    let mut step_off = init_backward_accumulators(hmm, observations, ws);

    for t in (0..t_max - 1).rev() {
        step_off -= n;
        let y_t = observations[t];
        let y_next = observations[t + 1];
        let inv_s_next = one / ws.scales[t + 1];
        for j in 0..n {
            ws.obs_beta[j] = hmm.emissions[(j, y_next)] * ws.beta_next[j];
        }
        let mut norm = zero;
        for i in 0..n {
            let mut sum = zero;
            for j in 0..n {
                sum += hmm.transitions[(i, j)] * ws.obs_beta[j];
            }
            ws.beta_curr[i] = sum * inv_s_next;
            norm += ws.alpha[step_off + i] * ws.beta_curr[i];
        }
        let inv_norm = if norm > zero { one / norm } else { one };
        let mut xi_row_off = 0;
        let mut em_row_off = 0;
        for i in 0..n {
            let alpha_i = ws.alpha[step_off + i];
            let gamma_i = alpha_i * ws.beta_curr[i] * inv_norm;
            ws.gamma_trans_sum[i] += gamma_i;
            ws.gamma_all_sum[i] += gamma_i;
            ws.emission_sum[em_row_off + y_t] += gamma_i;
            if t == 0 {
                ws.init_gamma[i] = gamma_i;
            }
            let scale_xi = alpha_i * inv_s_next * inv_norm;
            for j in 0..n {
                ws.xi_sum[xi_row_off + j] += scale_xi * hmm.transitions[(i, j)] * ws.obs_beta[j];
            }
            xi_row_off += n;
            em_row_off += m;
        }
        ws.beta_next[..n].copy_from_slice(&ws.beta_curr[..n]);
    }
}

fn snapshot_parameters<T: RealField + Copy + ToPrimitive>(
    hmm: &HiddenMarkovModel<T>,
    ws: &mut BaumWelchWorkspace<T>,
) {
    for (dst, src) in ws.prev_initial.iter_mut().zip(hmm.initial.iter()) {
        *dst = *src;
    }
    for (i, row) in ws
        .prev_transitions
        .chunks_mut(hmm.num_states)
        .take(hmm.num_states)
        .enumerate()
    {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = hmm.transitions[(i, j)];
        }
    }
    for (i, row) in ws
        .prev_emissions
        .chunks_mut(hmm.num_observations)
        .take(hmm.num_states)
        .enumerate()
    {
        for (k, slot) in row.iter_mut().enumerate() {
            *slot = hmm.emissions[(i, k)];
        }
    }
}

fn restore_parameters<T: RealField + Copy + ToPrimitive>(
    hmm: &mut HiddenMarkovModel<T>,
    ws: &BaumWelchWorkspace<T>,
) {
    for (dst, &src) in hmm.initial.iter_mut().zip(ws.prev_initial.iter()) {
        *dst = src;
    }
    for (i, row) in ws
        .prev_transitions
        .chunks(hmm.num_states)
        .take(hmm.num_states)
        .enumerate()
    {
        for (j, &val) in row.iter().enumerate() {
            hmm.transitions[(i, j)] = val;
        }
    }
    for (i, row) in ws
        .prev_emissions
        .chunks(hmm.num_observations)
        .take(hmm.num_states)
        .enumerate()
    {
        for (k, &val) in row.iter().enumerate() {
            hmm.emissions[(i, k)] = val;
        }
    }
}

fn apply_m_step<T: RealField + Copy + ToPrimitive>(
    hmm: &mut HiddenMarkovModel<T>,
    ws: &BaumWelchWorkspace<T>,
    prior: T,
    has_transitions: bool,
) {
    let n = hmm.num_states;
    let m = hmm.num_observations;
    let zero = T::zero();

    let mut pi_sum = zero;
    for i in 0..n {
        let v = ws.init_gamma[i] + prior;
        hmm.initial[i] = v;
        pi_sum += v;
    }
    if pi_sum > zero {
        for i in 0..n {
            hmm.initial[i] /= pi_sum;
        }
    }

    if has_transitions {
        let mut xi_off = 0;
        for i in 0..n {
            let mut row_sum = zero;
            for j in 0..n {
                row_sum += ws.xi_sum[xi_off + j] + prior;
            }
            if row_sum > zero {
                for j in 0..n {
                    hmm.transitions[(i, j)] = (ws.xi_sum[xi_off + j] + prior) / row_sum;
                }
            }
            xi_off += n;
        }
    }

    let mut em_off = 0;
    for i in 0..n {
        let mut row_sum = zero;
        for k in 0..m {
            row_sum += ws.emission_sum[em_off + k] + prior;
        }
        if row_sum > zero {
            for k in 0..m {
                hmm.emissions[(i, k)] = (ws.emission_sum[em_off + k] + prior) / row_sum;
            }
        }
        em_off += m;
    }
}

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Fits `(initial, transitions, emissions)` via Baum-Welch EM in-place using preallocated
    /// [`BaumWelchWorkspace`] buffers with zero heap allocations.
    #[verified_engine::verified]
    pub fn fit_baum_welch_into(
        &mut self,
        observations: &[usize],
        config: &BaumWelchConfig<T>,
        workspace: &mut BaumWelchWorkspace<T>,
        log_likelihood_history: &mut [T],
    ) -> Result<BaumWelchReport<T>, MarkovError> {
        validate_bw_inputs(
            self,
            observations,
            config,
            workspace,
            log_likelihood_history,
        )?;
        let initial_ll = run_forward_into(self, observations, workspace);
        if !initial_ll.is_finite() {
            return Ok(BaumWelchReport {
                iterations: 0,
                converged: true,
                initial_log_likelihood: initial_ll,
                final_log_likelihood: initial_ll,
            });
        }
        accumulate_backward_stats(self, observations, workspace);

        let has_transitions = observations.len() > 1;
        let mut prev_ll = initial_ll;
        let mut iterations = 0;
        let mut converged = false;

        for (iter, hist_slot) in log_likelihood_history
            .iter_mut()
            .enumerate()
            .take(config.max_iterations)
        {
            snapshot_parameters(self, workspace);
            apply_m_step(self, workspace, config.dirichlet_alpha, has_transitions);
            let new_ll = run_forward_into(self, observations, workspace);
            if !new_ll.is_finite() || new_ll < prev_ll {
                restore_parameters(self, workspace);
                *hist_slot = prev_ll;
                iterations = iter + 1;
                converged = true;
                break;
            }
            *hist_slot = new_ll;
            let delta = new_ll - prev_ll;
            prev_ll = new_ll;
            iterations = iter + 1;
            if delta <= config.tolerance {
                converged = true;
                break;
            }
            accumulate_backward_stats(self, observations, workspace);
        }

        Ok(BaumWelchReport {
            iterations,
            converged,
            initial_log_likelihood: initial_ll,
            final_log_likelihood: prev_ll,
        })
    }

    /// Convenience wrapper around [`HiddenMarkovModel::fit_baum_welch_into`] that allocates a
    /// [`BaumWelchWorkspace`] and returns the convergence report plus per-iteration log-likelihoods.
    #[verified_engine::verified(alloc_ok = "allocates owned output buffer")]
    pub fn fit_baum_welch(
        &mut self,
        observations: &[usize],
        config: &BaumWelchConfig<T>,
    ) -> Result<(BaumWelchReport<T>, Vec<T>), MarkovError> {
        if observations.is_empty() {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation sequence is empty".to_string(),
            });
        }
        BaumWelchConfig::new(
            config.max_iterations,
            config.tolerance,
            config.dirichlet_alpha,
        )?;
        let mut workspace =
            BaumWelchWorkspace::new(self.num_states, self.num_observations, observations.len());
        let mut history = vec![T::zero(); config.max_iterations];
        let report =
            self.fit_baum_welch_into(observations, config, &mut workspace, &mut history)?;
        history.truncate(report.iterations);
        Ok((report, history))
    }
}
