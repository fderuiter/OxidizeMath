use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::{DMatrix, RealField};
use num_traits::ToPrimitive;

#[verified_engine::verified]
fn log_prob_or_neg_inf<T: RealField + Copy>(p: T, neg_inf: T) -> T {
    if p > T::zero() { p.ln() } else { neg_inf }
}

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Log-space Viterbi algorithm: finds the most likely hidden state sequence without underflow.
    ///
    /// Accumulates log probabilities with $-\infty$ zero guards:
    /// - $\delta_0(i) = \ln \pi_i + \ln B_{i, y_0}$
    /// - $\delta_t(j) = \max_i [\delta_{t-1}(i) + \ln A_{ij}] + \ln B_{j, y_t}$
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidObservation`] if `observations` is empty or contains an
    /// out-of-bounds symbol index.
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
    /// let path = hmm.viterbi(&[1, 0]).unwrap();
    /// assert_eq!(path, vec![1, 0]);
    /// ```
    #[verified_engine::verified]
    pub fn viterbi(&self, observations: &[usize]) -> Result<Vec<usize>> {
        self.validate_observations(observations)?;

        let t_max = observations.len();
        let neg_inf = T::from_f64(f64::NEG_INFINITY).unwrap();
        let mut delta = DMatrix::from_element(self.num_states, t_max, neg_inf);
        let mut psi = vec![0_usize; self.num_states * t_max];

        let y0 = observations[0];
        for i in 0..self.num_states {
            let log_pi = log_prob_or_neg_inf(self.initial[i], neg_inf);
            let log_b = log_prob_or_neg_inf(self.emissions[(i, y0)], neg_inf);
            delta[(i, 0)] = if log_pi > neg_inf && log_b > neg_inf {
                log_pi + log_b
            } else {
                neg_inf
            };
        }

        for (t, &y_t) in observations.iter().enumerate().skip(1) {
            self.step_viterbi_column(t, y_t, neg_inf, &mut delta, &mut psi);
        }

        Ok(self.backtrack_viterbi(t_max, &delta, &psi))
    }

    #[verified_engine::verified]
    fn step_viterbi_column(
        &self,
        t: usize,
        y_t: usize,
        neg_inf: T,
        delta: &mut DMatrix<T>,
        psi: &mut [usize],
    ) {
        let t_max = delta.ncols();
        for j in 0..self.num_states {
            let mut best_prev = neg_inf;
            let mut best_idx = 0_usize;

            for i in 0..self.num_states {
                let prev = delta[(i, t - 1)];
                let log_a = log_prob_or_neg_inf(self.transitions[(i, j)], neg_inf);
                if prev > neg_inf && log_a > neg_inf {
                    let cand = prev + log_a;
                    if cand > best_prev {
                        best_prev = cand;
                        best_idx = i;
                    }
                }
            }

            let log_b = log_prob_or_neg_inf(self.emissions[(j, y_t)], neg_inf);
            delta[(j, t)] = if best_prev > neg_inf && log_b > neg_inf {
                best_prev + log_b
            } else {
                neg_inf
            };
            psi[flatten_2d_index(t, j, t_max)] = best_idx;
        }
    }

    #[verified_engine::verified]
    fn backtrack_viterbi(&self, t_max: usize, delta: &DMatrix<T>, psi: &[usize]) -> Vec<usize> {
        let mut path = vec![0_usize; t_max];
        let mut best_val = delta[(0, t_max - 1)];
        let mut best_idx = 0_usize;
        for i in 1..self.num_states {
            if delta[(i, t_max - 1)] > best_val {
                best_val = delta[(i, t_max - 1)];
                best_idx = i;
            }
        }
        path[t_max - 1] = best_idx;

        for t in (0..t_max - 1).rev() {
            let next_state = path[t + 1];
            path[t] = psi[flatten_2d_index(t + 1, next_state, t_max)];
        }
        path
    }
}

// [cite:clinical_trials_statistics]
