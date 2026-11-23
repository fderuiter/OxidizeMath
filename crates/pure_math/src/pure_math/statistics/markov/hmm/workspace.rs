//! Preallocated zero-allocation scratch workspace for Hidden Markov Model evaluation.

use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::RealField;
use num_traits::ToPrimitive;

/// Summary metrics from a zero-allocation HMM forward-backward + Viterbi pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HmmEvaluation<T> {
    /// Unscaled observation sequence likelihood $P(Y_{0:T-1})$.
    pub likelihood: T,
    /// Log observation likelihood $\ln P(Y_{0:T-1})$.
    pub log_likelihood: T,
    /// Exact joint probability $P(S^*_{0:T-1}, Y_{0:T-1})$ of the Viterbi path.
    pub viterbi_joint_prob: T,
    /// Exact joint probability $P(\hat{S}^{\text{MAP}}_{0:T-1}, Y_{0:T-1})$ of the pointwise marginal path.
    pub marginal_joint_prob: T,
    /// Whether the pointwise marginal argmax path traverses a forbidden $A_{i,j} = 0$ transition.
    pub has_forbidden_marginal_step: bool,
    /// First transition index $t$ (from $t \to t+1$) where $A_{\hat{s}_t, \hat{s}_{t+1}} = 0$.
    pub first_forbidden_step: Option<usize>,
}

/// Preallocated flat buffers for zero-allocation HMM forward, backward, smoothing, filtering, and Viterbi passes.
#[derive(Debug, Clone)]
pub struct HmmScratch<T> {
    max_states: usize,
    max_steps: usize,
    alpha: Vec<T>,
    beta: Vec<T>,
    gamma: Vec<T>,
    delta: Vec<T>,
    psi: Vec<usize>,
    scaling_factors: Vec<T>,
    filter_belief: Vec<T>,
    viterbi_path: Vec<usize>,
    marginal_path: Vec<usize>,
}

impl<T: RealField + Copy> HmmScratch<T> {
    /// Preallocates scratch buffers for up to `max_states` hidden states and `max_steps` observations.
    #[verified_engine::verified]
    pub fn new(max_states: usize, max_steps: usize) -> Self {
        let cap = max_states.saturating_mul(max_steps);
        let zero = T::zero();
        Self {
            max_states,
            max_steps,
            alpha: vec![zero; cap],
            beta: vec![zero; cap],
            gamma: vec![zero; cap],
            delta: vec![zero; cap],
            psi: vec![0; cap],
            scaling_factors: vec![zero; max_steps],
            filter_belief: vec![zero; max_states],
            viterbi_path: vec![0; max_steps],
            marginal_path: vec![0; max_steps],
        }
    }

    /// Maximum supported hidden states without reallocation.
    #[verified_engine::verified]
    pub fn max_states(&self) -> usize {
        self.max_states
    }

    /// Maximum supported sequence length without reallocation.
    #[verified_engine::verified]
    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Reads posterior probability $\gamma_t(i)$ after [`HiddenMarkovModel::evaluate_with_scratch`].
    #[verified_engine::verified]
    pub fn gamma_at(&self, state: usize, step: usize) -> T {
        let idx = flatten_2d_index(step, state, self.max_steps);
        self.gamma[idx]
    }

    /// Reads scaled forward probability $\hat{\alpha}_t(i)$ after evaluation.
    #[verified_engine::verified]
    pub fn alpha_at(&self, state: usize, step: usize) -> T {
        let idx = flatten_2d_index(step, state, self.max_steps);
        self.alpha[idx]
    }

    /// Returns the filtered belief vector $P(X_{T-1} = i \mid Y_{0:T-1})$ of length `num_states`.
    #[verified_engine::verified]
    pub fn filter_slice(&self, num_states: usize) -> &[T] {
        &self.filter_belief[..num_states.min(self.max_states)]
    }

    /// Returns the decoded Viterbi joint state sequence of length `len`.
    #[verified_engine::verified]
    pub fn viterbi_slice(&self, len: usize) -> &[usize] {
        &self.viterbi_path[..len.min(self.max_steps)]
    }

    /// Returns the pointwise marginal MAP state sequence $\arg\max_i \gamma_t(i)$ of length `len`.
    #[verified_engine::verified]
    pub fn marginal_slice(&self, len: usize) -> &[usize] {
        &self.marginal_path[..len.min(self.max_steps)]
    }

    /// Returns the per-step forward scaling factors $c_t = P(Y_t \mid Y_{0:t-1})$ of length `len`.
    #[verified_engine::verified]
    pub fn scaling_slice(&self, len: usize) -> &[T] {
        &self.scaling_factors[..len.min(self.max_steps)]
    }
}

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Computes exact joint probability $P(S_{0:T-1} = \text{path}, Y_{0:T-1} = \text{observations})$ with zero allocations.
    #[verified_engine::verified]
    pub fn path_joint_probability(
        &self,
        path: &[usize],
        observations: &[usize],
    ) -> std::result::Result<T, MarkovError> {
        self.path_joint_prob_inner(path, observations)
    }

    /// Executes forward scaling, backward smoothing, filtering, marginal MAP, and Viterbi decoding in place.
    #[verified_engine::verified]
    pub fn evaluate_with_scratch(
        &self,
        observations: &[usize],
        scratch: &mut HmmScratch<T>,
    ) -> std::result::Result<HmmEvaluation<T>, MarkovError> {
        self.validate_scratch_inputs(observations, scratch)?;
        let (likelihood, log_likelihood) = self.run_forward_into(observations, scratch);
        self.run_backward_and_gamma_into(observations, scratch);
        self.run_viterbi_into(observations, scratch);

        let t_len = observations.len();
        let viterbi_joint_prob =
            self.path_joint_prob_inner(&scratch.viterbi_path[..t_len], observations)?;
        let marginal_joint_prob =
            self.path_joint_prob_inner(&scratch.marginal_path[..t_len], observations)?;
        let first_forbidden_step = self.find_forbidden_transition(&scratch.marginal_path[..t_len]);

        Ok(HmmEvaluation {
            likelihood,
            log_likelihood,
            viterbi_joint_prob,
            marginal_joint_prob,
            has_forbidden_marginal_step: first_forbidden_step.is_some(),
            first_forbidden_step,
        })
    }

    fn validate_scratch_inputs(
        &self,
        observations: &[usize],
        scratch: &HmmScratch<T>,
    ) -> std::result::Result<(), MarkovError> {
        if observations.is_empty() {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation sequence cannot be empty".to_string(),
            });
        }
        if self.num_states > scratch.max_states || observations.len() > scratch.max_steps {
            return Err(MarkovError::InvalidObservation {
                reason: "Scratch buffer capacity exceeded".to_string(),
            });
        }
        for &obs in observations {
            if obs >= self.num_observations {
                return Err(MarkovError::InvalidObservation {
                    reason: "Observation index exceeds num_observations".to_string(),
                });
            }
        }
        Ok(())
    }

    fn path_joint_prob_inner(
        &self,
        path: &[usize],
        observations: &[usize],
    ) -> std::result::Result<T, MarkovError> {
        if observations.is_empty() {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation sequence cannot be empty".to_string(),
            });
        }
        if path.len() != observations.len() {
            return Err(MarkovError::InvalidState {
                reason: "State path length must match observations".to_string(),
            });
        }
        let s0 = path[0];
        let y0 = observations[0];
        if s0 >= self.num_states {
            return Err(MarkovError::InvalidState {
                reason: "Initial state index out of bounds".to_string(),
            });
        }
        if y0 >= self.num_observations {
            return Err(MarkovError::InvalidObservation {
                reason: "Initial observation index out of bounds".to_string(),
            });
        }
        let mut prob = self.initial[s0] * self.emissions[(s0, y0)];
        for t in 1..observations.len() {
            let prev = path[t - 1];
            let curr = path[t];
            let y_t = observations[t];
            if curr >= self.num_states {
                return Err(MarkovError::InvalidState {
                    reason: "State index out of bounds".to_string(),
                });
            }
            if y_t >= self.num_observations {
                return Err(MarkovError::InvalidObservation {
                    reason: "Observation index out of bounds".to_string(),
                });
            }
            prob *= self.transitions[(prev, curr)] * self.emissions[(curr, y_t)];
        }
        Ok(prob)
    }

    fn find_forbidden_transition(&self, path: &[usize]) -> Option<usize> {
        let zero = T::zero();
        (0..path.len().saturating_sub(1))
            .find(|&t| self.transitions[(path[t], path[t + 1])] <= zero)
    }

    fn run_forward_into(&self, observations: &[usize], scratch: &mut HmmScratch<T>) -> (T, T) {
        let n_states = self.num_states;
        let stride = scratch.max_steps;
        let t_len = observations.len();
        let zero = T::zero();
        let one = T::one();

        let y0 = observations[0];
        let mut scale0 = zero;
        for i in 0..n_states {
            let val = self.initial[i] * self.emissions[(i, y0)];
            scratch.alpha[flatten_2d_index(0, i, stride)] = val;
            scale0 += val;
        }
        if scale0 > zero {
            for i in 0..n_states {
                scratch.alpha[flatten_2d_index(0, i, stride)] /= scale0;
            }
        }
        scratch.scaling_factors[0] = scale0;

        for t in 1..t_len {
            let y_t = observations[t];
            let mut scale_t = zero;
            for j in 0..n_states {
                let mut sum = zero;
                for i in 0..n_states {
                    sum += scratch.alpha[flatten_2d_index(t - 1, i, stride)]
                        * self.transitions[(i, j)];
                }
                let val = sum * self.emissions[(j, y_t)];
                scratch.alpha[flatten_2d_index(t, j, stride)] = val;
                scale_t += val;
            }
            if scale_t > zero {
                for j in 0..n_states {
                    scratch.alpha[flatten_2d_index(t, j, stride)] /= scale_t;
                }
            }
            scratch.scaling_factors[t] = scale_t;
        }

        let mut likelihood = one;
        let mut log_likelihood = zero;
        for &c in scratch.scaling_factors.iter().take(t_len) {
            likelihood *= c;
            if c <= zero {
                log_likelihood = zero.ln();
            } else if log_likelihood.is_finite() {
                log_likelihood += c.ln();
            }
        }
        for i in 0..n_states {
            scratch.filter_belief[i] = scratch.alpha[flatten_2d_index(t_len - 1, i, stride)];
        }
        (likelihood, log_likelihood)
    }

    fn run_backward_and_gamma_into(&self, observations: &[usize], scratch: &mut HmmScratch<T>) {
        let n_states = self.num_states;
        let stride = scratch.max_steps;
        let t_len = observations.len();
        let zero = T::zero();
        let one = T::one();

        let last_scale = scratch.scaling_factors[t_len - 1];
        let last_beta = if last_scale > zero {
            one / last_scale
        } else {
            zero
        };
        for i in 0..n_states {
            scratch.beta[flatten_2d_index(t_len - 1, i, stride)] = last_beta;
        }

        for t in (0..t_len - 1).rev() {
            let y_next = observations[t + 1];
            let scale_t = scratch.scaling_factors[t];
            for i in 0..n_states {
                let mut sum = zero;
                for j in 0..n_states {
                    sum += self.transitions[(i, j)]
                        * self.emissions[(j, y_next)]
                        * scratch.beta[flatten_2d_index(t + 1, j, stride)];
                }
                scratch.beta[flatten_2d_index(t, i, stride)] =
                    if scale_t > zero { sum / scale_t } else { zero };
            }
        }

        for t in 0..t_len {
            let mut sum = zero;
            for i in 0..n_states {
                let idx = flatten_2d_index(t, i, stride);
                let g = scratch.alpha[idx] * scratch.beta[idx];
                scratch.gamma[idx] = g;
                sum += g;
            }
            let mut best_state = 0;
            let mut best_val = zero;
            for i in 0..n_states {
                let idx = flatten_2d_index(t, i, stride);
                if sum > zero {
                    scratch.gamma[idx] /= sum;
                }
                let val = scratch.gamma[idx];
                if i == 0 || val > best_val {
                    best_val = val;
                    best_state = i;
                }
            }
            scratch.marginal_path[t] = best_state;
        }
    }

    fn run_viterbi_into(&self, observations: &[usize], scratch: &mut HmmScratch<T>) {
        let n_states = self.num_states;
        let stride = scratch.max_steps;
        let t_len = observations.len();
        let zero = T::zero();
        let neg_inf = zero.ln();

        let y0 = observations[0];
        for i in 0..n_states {
            let p = self.initial[i] * self.emissions[(i, y0)];
            let idx = flatten_2d_index(0, i, stride);
            scratch.delta[idx] = if p > zero { p.ln() } else { neg_inf };
            scratch.psi[idx] = 0;
        }

        for t in 1..t_len {
            let y_t = observations[t];
            for j in 0..n_states {
                let emit_p = self.emissions[(j, y_t)];
                let emit_ln = if emit_p > zero { emit_p.ln() } else { neg_inf };
                let mut best_prev = 0;
                let mut best_score = neg_inf;
                for i in 0..n_states {
                    let trans_p = self.transitions[(i, j)];
                    let prev_delta = scratch.delta[flatten_2d_index(t - 1, i, stride)];
                    let score = if trans_p > zero && prev_delta > neg_inf {
                        prev_delta + trans_p.ln()
                    } else {
                        neg_inf
                    };
                    if i == 0 || score > best_score {
                        best_score = score;
                        best_prev = i;
                    }
                }
                let idx = flatten_2d_index(t, j, stride);
                scratch.delta[idx] = if best_score > neg_inf && emit_ln > neg_inf {
                    best_score + emit_ln
                } else {
                    neg_inf
                };
                scratch.psi[idx] = best_prev;
            }
        }

        let mut last_best = 0;
        let mut last_score = scratch.delta[flatten_2d_index(t_len - 1, 0, stride)];
        for i in 1..n_states {
            let score = scratch.delta[flatten_2d_index(t_len - 1, i, stride)];
            if score > last_score {
                last_score = score;
                last_best = i;
            }
        }
        scratch.viterbi_path[t_len - 1] = last_best;
        for t in (0..t_len - 1).rev() {
            let next_state = scratch.viterbi_path[t + 1];
            scratch.viterbi_path[t] = scratch.psi[flatten_2d_index(t + 1, next_state, stride)];
        }
    }
}

// [cite:clinical_trials_statistics]
