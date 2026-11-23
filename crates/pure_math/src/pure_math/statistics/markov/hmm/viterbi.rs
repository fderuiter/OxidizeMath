use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use nalgebra::RealField;
use num_traits::ToPrimitive;

#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;

/// Preallocated scratch buffers for zero-allocation log-domain Viterbi decoding.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ViterbiWorkspace<T> {
    num_states: usize,
    num_observations: usize,
    max_len: usize,
    prev_log_delta: Vec<T>,
    curr_log_delta: Vec<T>,
    log_initial: Vec<T>,
    log_transitions: Vec<T>,
    log_emissions: Vec<T>,
    backpointers: Vec<usize>,
}

impl<T: RealField + Copy> ViterbiWorkspace<T> {
    /// Allocates a workspace sized for up to `num_states`, `num_observations`, and `max_len` steps.
    #[verified_engine::verified]
    pub fn new(num_states: usize, num_observations: usize, max_len: usize) -> Self {
        let z = T::zero();
        Self {
            num_states,
            num_observations,
            max_len,
            prev_log_delta: vec![z; num_states],
            curr_log_delta: vec![z; num_states],
            log_initial: vec![z; num_states],
            log_transitions: vec![z; num_states.saturating_mul(num_states)],
            log_emissions: vec![z; num_states.saturating_mul(num_observations)],
            backpointers: vec![0; num_states.saturating_mul(max_len)],
        }
    }

    /// Grows internal buffers if needed to accommodate the requested capacities.
    #[verified_engine::verified]
    pub fn ensure_capacity(&mut self, num_states: usize, num_observations: usize, max_len: usize) {
        let z = T::zero();
        if num_states > self.num_states {
            self.prev_log_delta.resize(num_states, z);
            self.curr_log_delta.resize(num_states, z);
            self.log_initial.resize(num_states, z);
        }
        if num_states > self.num_states || num_observations > self.num_observations {
            self.log_transitions
                .resize(num_states.saturating_mul(num_states), z);
            self.log_emissions
                .resize(num_states.saturating_mul(num_observations), z);
        }
        let needed_bp = num_states
            .max(self.num_states)
            .saturating_mul(max_len.max(self.max_len));
        if needed_bp > self.backpointers.len() {
            self.backpointers.resize(needed_bp, 0);
        }
        self.num_states = self.num_states.max(num_states);
        self.num_observations = self.num_observations.max(num_observations);
        self.max_len = self.max_len.max(max_len);
    }

    /// Returns the maximum number of hidden states supported without reallocation.
    #[verified_engine::verified]
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Returns the maximum alphabet size supported without reallocation.
    #[verified_engine::verified]
    pub fn num_observations(&self) -> usize {
        self.num_observations
    }

    /// Returns the maximum observation sequence length supported without reallocation.
    #[verified_engine::verified]
    pub fn max_len(&self) -> usize {
        self.max_len
    }
}

#[inline]
fn neg_infinity<T: RealField + Copy>() -> T {
    T::from_f64(f64::NEG_INFINITY)
        .or_else(T::min_value)
        .unwrap_or_else(|| T::zero().ln())
}

#[inline]
fn log_prob<T: RealField + Copy>(prob: T, neg_inf: T) -> T {
    if prob > T::zero() { prob.ln() } else { neg_inf }
}

fn validate_viterbi_inputs<T: RealField + Copy + ToPrimitive>(
    hmm: &HiddenMarkovModel<T>,
    observations: &[usize],
    workspace: &ViterbiWorkspace<T>,
    out_path: &[usize],
) -> Result<()> {
    if observations.is_empty() {
        return Err(MarkovError::InvalidObservation {
            reason: "Observation sequence is empty".to_string(),
        });
    }
    if out_path.len() != observations.len() {
        return Err(MarkovError::Math(MathError::DimensionMismatch {
            expected: Dimension(observations.len()),
            actual: Dimension(out_path.len()),
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

fn populate_log_tables<T: RealField + Copy + ToPrimitive>(
    hmm: &HiddenMarkovModel<T>,
    workspace: &mut ViterbiWorkspace<T>,
    neg_inf: T,
) {
    let n = hmm.num_states;
    let m = hmm.num_observations;
    for i in 0..n {
        workspace.log_initial[i] = log_prob(hmm.initial[i], neg_inf);
    }
    let mut trans_off = 0;
    for i in 0..n {
        for j in 0..n {
            workspace.log_transitions[trans_off + j] = log_prob(hmm.transitions[(i, j)], neg_inf);
        }
        trans_off += n;
    }
    let mut emit_off = 0;
    for i in 0..n {
        for k in 0..m {
            workspace.log_emissions[emit_off + k] = log_prob(hmm.emissions[(i, k)], neg_inf);
        }
        emit_off += m;
    }
}

#[allow(clippy::too_many_arguments)]
fn viterbi_step<T: RealField + Copy>(
    n: usize,
    m: usize,
    y_t: usize,
    neg_inf: T,
    prev_delta: &[T],
    log_transitions: &[T],
    log_emissions: &[T],
    curr_delta: &mut [T],
    psi_step: &mut [usize],
) {
    let mut emit_off = 0;
    for j in 0..n {
        let mut best_val = neg_inf;
        let mut best_idx = 0;
        let mut found = false;
        let mut trans_off = 0;
        for (i, &prev_lp) in prev_delta.iter().take(n).enumerate() {
            let log_a = log_transitions[trans_off + j];
            if prev_lp > neg_inf && log_a > neg_inf {
                let candidate = prev_lp + log_a;
                if !found || candidate > best_val {
                    found = true;
                    best_val = candidate;
                    best_idx = i;
                }
            }
            trans_off += n;
        }
        let log_b = log_emissions[emit_off + y_t];
        curr_delta[j] = if found && log_b > neg_inf {
            best_val + log_b
        } else {
            neg_inf
        };
        psi_step[j] = best_idx;
        emit_off += m;
    }
}

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Decodes the most likely hidden state sequence into `out_path` using exact log-domain
    /// dynamic programming and preallocated [`ViterbiWorkspace`] buffers with zero heap allocations.
    #[verified_engine::verified]
    pub fn viterbi_into(
        &self,
        observations: &[usize],
        workspace: &mut ViterbiWorkspace<T>,
        out_path: &mut [usize],
    ) -> Result<()> {
        validate_viterbi_inputs(self, observations, workspace, out_path)?;
        let n = self.num_states;
        let m = self.num_observations;
        let t_max = observations.len();
        let neg_inf = neg_infinity::<T>();
        populate_log_tables(self, workspace, neg_inf);

        let y0 = observations[0];
        let mut emit_off = 0;
        for i in 0..n {
            let log_pi = workspace.log_initial[i];
            let log_b = workspace.log_emissions[emit_off + y0];
            workspace.prev_log_delta[i] = if log_pi > neg_inf && log_b > neg_inf {
                log_pi + log_b
            } else {
                neg_inf
            };
            workspace.backpointers[i] = 0;
            emit_off += m;
        }

        let mut bp_off = n;
        for &y_t in &observations[1..t_max] {
            viterbi_step(
                n,
                m,
                y_t,
                neg_inf,
                &workspace.prev_log_delta,
                &workspace.log_transitions,
                &workspace.log_emissions,
                &mut workspace.curr_log_delta,
                &mut workspace.backpointers[bp_off..bp_off + n],
            );
            std::mem::swap(&mut workspace.prev_log_delta, &mut workspace.curr_log_delta);
            bp_off += n;
        }

        let mut best_final = neg_inf;
        let mut last_state = 0;
        let mut found = false;
        for i in 0..n {
            let val = workspace.prev_log_delta[i];
            if val > neg_inf && (!found || val > best_final) {
                found = true;
                best_final = val;
                last_state = i;
            }
        }
        out_path[t_max - 1] = last_state;

        let mut step_bp_off = bp_off;
        for t in (0..t_max - 1).rev() {
            step_bp_off -= n;
            out_path[t] = workspace.backpointers[step_bp_off + out_path[t + 1]];
        }
        Ok(())
    }

    /// Viterbi algorithm: finds the most likely hidden state sequence via exact log-domain
    /// dynamic programming.
    ///
    /// # Examples
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::markov::hmm::HiddenMarkovModel;
    /// use math_explorer::nalgebra::{DMatrix, DVector};
    ///
    /// let initial = DVector::from_vec(vec![0.5, 0.5]);
    /// let transitions = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
    /// let emissions = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.2, 0.8]);
    /// let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();
    /// assert_eq!(hmm.viterbi(&[1, 0]).unwrap(), vec![1, 0]);
    /// ```
    #[verified_engine::verified]
    pub fn viterbi(&self, observations: &[usize]) -> Result<Vec<usize>> {
        if observations.is_empty() {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation sequence is empty".to_string(),
            });
        }
        let mut workspace =
            ViterbiWorkspace::new(self.num_states, self.num_observations, observations.len());
        let mut out_path = vec![0; observations.len()];
        self.viterbi_into(observations, &mut workspace, &mut out_path)?;
        Ok(out_path)
    }
}

// [cite:clinical_trials_statistics]
