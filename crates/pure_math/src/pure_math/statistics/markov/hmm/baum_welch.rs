//! Zero-allocation Baum-Welch Expectation-Maximization (EM) parameter estimation for HMMs,
//! delegating chain forward-backward message passing to [`engine_graph`] and row normalization
//! to [`engine_prob`].
// [cite:clinical_trials_statistics]

pub use super::baum_welch_types::BaumWelchResult;
use super::baum_welch_types::{BaumWelchConfig, BaumWelchReport, BaumWelchWorkspace};
use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
use engine_graph::{
    ChainFbWorkspace, ChainModelView, chain_accumulate_xi_into, chain_forward_backward_into,
};
use engine_prob::discrete_posterior_into;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use nalgebra::RealField;
use num_traits::ToPrimitive;

#[inline]
const fn idx2d(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

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

fn sync_f64_model<T: RealField + Copy + ToPrimitive>(
    hmm: &HiddenMarkovModel<T>,
    ws: &mut BaumWelchWorkspace<T>,
) {
    let n = hmm.num_states;
    let m = hmm.num_observations;
    for i in 0..n {
        ws.f64_initial[i] = hmm.initial[i].to_f64().unwrap_or(0.0);
        for j in 0..n {
            ws.f64_transitions[idx2d(i, j, n)] = hmm.transitions[(i, j)].to_f64().unwrap_or(0.0);
        }
        for k in 0..m {
            ws.f64_emissions[idx2d(i, k, m)] = hmm.emissions[(i, k)].to_f64().unwrap_or(0.0);
        }
    }
}

pub(super) fn run_e_step_into<T: RealField + Copy + ToPrimitive>(
    hmm: &HiddenMarkovModel<T>,
    observations: &[usize],
    ws: &mut BaumWelchWorkspace<T>,
) -> f64 {
    let n = hmm.num_states;
    let m = hmm.num_observations;
    let t_len = observations.len();
    sync_f64_model(hmm, ws);

    let ll = {
        let model_view = ChainModelView {
            num_states: n,
            num_obs: m,
            initial: &ws.f64_initial[..n],
            transitions: &ws.f64_transitions[..n * n],
            emissions: &ws.f64_emissions[..n * m],
        };
        let mut fb_ws = ChainFbWorkspace {
            alpha: &mut ws.f64_alpha[..t_len * n],
            beta: &mut ws.f64_beta[..t_len * n],
            scales: &mut ws.f64_scales[..t_len],
            gamma: &mut ws.f64_gamma[..t_len * n],
        };
        let ll = chain_forward_backward_into(&model_view, observations, &mut fb_ws)
            .unwrap_or(f64::NEG_INFINITY);
        if ll.is_finite() {
            let _ = chain_accumulate_xi_into(
                &model_view,
                observations,
                &fb_ws,
                &mut ws.f64_xi_sum[..n * n],
            );
        }
        ll
    };
    if !ll.is_finite() {
        return ll;
    }

    ws.f64_emission_sum[..n * m].fill(0.0);
    for (t, &obs) in observations.iter().enumerate() {
        for i in 0..n {
            ws.f64_emission_sum[idx2d(i, obs, m)] += ws.f64_gamma[idx2d(t, i, n)];
        }
    }
    ll
}

pub(super) fn snapshot_parameters<T: RealField + Copy + ToPrimitive>(
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

pub(super) fn restore_parameters<T: RealField + Copy + ToPrimitive>(
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

pub(super) fn apply_m_step<T: RealField + Copy + ToPrimitive>(
    hmm: &mut HiddenMarkovModel<T>,
    ws: &mut BaumWelchWorkspace<T>,
    prior_f64: f64,
    has_transitions: bool,
) {
    let n = hmm.num_states;
    let m = hmm.num_observations;

    for i in 0..n {
        ws.f64_row_in[i] = ws.f64_gamma[idx2d(0, i, n)] + prior_f64;
    }
    if let Ok(ev) = discrete_posterior_into(
        &ws.f64_row_in[..n],
        &ws.f64_row_ones[..n],
        &mut ws.f64_row_out[..n],
    ) && ev > 0.0
    {
        for i in 0..n {
            hmm.initial[i] = T::from_f64(ws.f64_row_out[i]).unwrap_or_else(T::zero);
        }
    }

    if has_transitions {
        for i in 0..n {
            for j in 0..n {
                ws.f64_row_in[j] = ws.f64_xi_sum[idx2d(i, j, n)] + prior_f64;
            }
            if let Ok(ev) = discrete_posterior_into(
                &ws.f64_row_in[..n],
                &ws.f64_row_ones[..n],
                &mut ws.f64_row_out[..n],
            ) && ev > 0.0
            {
                for j in 0..n {
                    hmm.transitions[(i, j)] =
                        T::from_f64(ws.f64_row_out[j]).unwrap_or_else(T::zero);
                }
            }
        }
    }

    for i in 0..n {
        for k in 0..m {
            ws.f64_row_in[k] = ws.f64_emission_sum[idx2d(i, k, m)] + prior_f64;
        }
        if let Ok(ev) = discrete_posterior_into(
            &ws.f64_row_in[..m],
            &ws.f64_row_ones[..m],
            &mut ws.f64_row_out[..m],
        ) && ev > 0.0
        {
            for k in 0..m {
                hmm.emissions[(i, k)] = T::from_f64(ws.f64_row_out[k]).unwrap_or_else(T::zero);
            }
        }
    }
}

pub(super) fn scalar_from_ll<T: RealField + Copy>(ll: f64) -> T {
    if ll.is_finite() {
        T::from_f64(ll).unwrap_or_else(T::zero)
    } else {
        let zero = T::zero();
        T::from_f64(f64::NEG_INFINITY)
            .or_else(T::min_value)
            .unwrap_or_else(|| zero.ln())
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
        let initial_ll_f64 = run_e_step_into(self, observations, workspace);
        let initial_ll = scalar_from_ll::<T>(initial_ll_f64);
        if !initial_ll_f64.is_finite() {
            return Ok(BaumWelchReport {
                iterations: 0,
                converged: true,
                initial_log_likelihood: initial_ll,
                final_log_likelihood: initial_ll,
            });
        }

        let has_transitions = observations.len() > 1;
        let prior_f64 = config.dirichlet_alpha.to_f64().unwrap_or(0.0);
        let tol_f64 = config.tolerance.to_f64().unwrap_or(0.0);
        let mut prev_ll_f64 = initial_ll_f64;
        let mut prev_ll = initial_ll;
        let mut iterations = 0;
        let mut converged = false;

        for (iter, hist_slot) in log_likelihood_history
            .iter_mut()
            .enumerate()
            .take(config.max_iterations)
        {
            snapshot_parameters(self, workspace);
            apply_m_step(self, workspace, prior_f64, has_transitions);
            let new_ll_f64 = run_e_step_into(self, observations, workspace);
            if !new_ll_f64.is_finite() || new_ll_f64 < prev_ll_f64 {
                restore_parameters(self, workspace);
                *hist_slot = prev_ll;
                iterations = iter + 1;
                converged = true;
                break;
            }
            let new_ll = scalar_from_ll::<T>(new_ll_f64);
            *hist_slot = new_ll;
            let delta = new_ll_f64 - prev_ll_f64;
            prev_ll_f64 = new_ll_f64;
            prev_ll = new_ll;
            iterations = iter + 1;
            if delta <= tol_f64 {
                converged = true;
                break;
            }
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
    #[verified_engine::verified]
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
