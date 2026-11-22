use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, RealField};
use num_traits::ToPrimitive;

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Viterbi algorithm: finds the most likely state sequence.
    ///
    /// # Mathematical Background
    ///
    /// Define δ(t, i) = max P(X₁, ..., Xₜ₋₁, Xₜ = i, Y₁, ..., Yₜ),
    /// the maximum probability of any state sequence ending in state i at time t.
    ///
    /// Recursion:
    /// - δ(1, i) = πᵢ · Bᵢ,y₁
    /// - δ(t+1, j) = max_i [δ(t, i) · Aᵢⱼ] · Bⱼ,yₜ₊₁
    ///
    /// Backtracking gives the most likely path.
    ///
    /// # Arguments
    ///
    /// * `observations` - Sequence of observation indices
    ///
    /// # Returns
    ///
    /// The most likely state sequence.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidObservation`] if the `observations` sequence is empty or
    /// contains an index out of bounds.
    ///
    /// # Panics
    ///
    /// Panics if the state index cannot be converted to or from the generic type `T`.
    /// This is highly unlikely for standard floating-point types.
    ///
    /// # Examples
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::markov::hmm::HiddenMarkovModel;
    /// use math_explorer::nalgebra::{DMatrix, DVector};
    ///
    /// let initial = DVector::from_vec(vec![0.5, 0.5]);
    /// let transitions = DMatrix::from_row_slice(2, 2, &[
    ///     0.7, 0.3,
    ///     0.4, 0.6,
    /// ]);
    /// let emissions = DMatrix::from_row_slice(2, 2, &[
    ///     0.7, 0.3,
    ///     0.2, 0.8,
    /// ]);
    /// let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();
    /// let obs = vec![1, 0];
    /// let path = hmm.viterbi(&obs).unwrap();
    /// assert_eq!(path, vec![1, 0]);
    /// ```
    #[verified_engine::verified]
    pub fn viterbi(&self, observations: &[usize]) -> Result<Vec<usize>> {
        if observations.is_empty() {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation sequence is empty".to_string(),
            });
        }

        let t_max = observations.len();
        let mut delta = DMatrix::zeros(self.num_states, t_max);
        let mut psi = DMatrix::zeros(self.num_states, t_max);
        let zero = T::zero();

        // Initialize: δ(1, i) = πᵢ · Bᵢ,y₁
        let y0 = observations[0];
        if y0 >= self.num_observations {
            return Err(MarkovError::InvalidObservation {
                reason: format!("Observation {} out of bounds", y0),
            });
        }

        for i in 0..self.num_states {
            delta[(i, 0)] = self.initial[i] * self.emissions[(i, y0)];
        }

        // Recursion
        for t in 1..t_max {
            let y_t = observations[t];
            if y_t >= self.num_observations {
                return Err(MarkovError::InvalidObservation {
                    reason: format!("Observation {} out of bounds", y_t),
                });
            }

            for j in 0..self.num_states {
                let mut max_val = zero;
                let mut max_idx = 0;

                for i in 0..self.num_states {
                    let val = delta[(i, t - 1)] * self.transitions[(i, j)];
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }

                delta[(j, t)] = max_val * self.emissions[(j, y_t)];
                psi[(j, t)] = T::from_usize(max_idx).unwrap();
            }
        }

        // Backtracking
        let mut path = vec![0; t_max];

        // Find best final state
        let mut max_val = zero;
        let mut max_idx = 0;
        for i in 0..self.num_states {
            if delta[(i, t_max - 1)] > max_val {
                max_val = delta[(i, t_max - 1)];
                max_idx = i;
            }
        }
        path[t_max - 1] = max_idx;

        // Backtrack
        for t in (0..t_max - 1).rev() {
            path[t] = psi[(path[t + 1], t + 1)].to_usize().unwrap();
        }

        Ok(path)
    }
}
