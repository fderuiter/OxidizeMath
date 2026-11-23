use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Computes P(observations) using the forward algorithm.
    ///
    /// # Mathematical Background
    ///
    /// Define α(t, i) = P(Y₁, ..., Yₜ, Xₜ = i), the probability of observing
    /// the first t observations and being in state i at time t.
    ///
    /// Recursion:
    /// - α(1, i) = πᵢ · Bᵢ,y₁
    /// - α(t+1, j) = Σᵢ α(t, i) · Aᵢⱼ · Bⱼ,yₜ₊₁
    ///
    /// Then P(Y₁, ..., Yₜ) = Σᵢ α(T, i).
    ///
    /// # Arguments
    ///
    /// * `observations` - Sequence of observation indices
    ///
    /// # Returns
    ///
    /// The probability of the observation sequence.
    #[verified_engine::verified]
    pub fn forward(&self, observations: &[usize]) -> Result<T> {
        let (_, scaling_factors) = self.forward_probabilities(observations)?;
        let one = T::one();
        Ok(scaling_factors.iter().fold(one, |acc, &c| acc * c))
    }

    /// Computes log P(observations) from the forward scaling factors to avoid underflow.
    ///
    /// Returns `-infinity` (`T::zero().ln()`) when the observation sequence has zero likelihood.
    #[verified_engine::verified]
    pub fn forward_log_likelihood(&self, observations: &[usize]) -> Result<T> {
        let (_, scaling_factors) = self.forward_probabilities(observations)?;
        let zero = T::zero();
        let mut log_likelihood = zero;
        for &c in &scaling_factors {
            if c <= zero {
                return Ok(zero.ln());
            }
            log_likelihood += c.ln();
        }
        Ok(log_likelihood)
    }

    /// Computes forward probabilities α(t, i) for all t and i.
    ///
    /// # Returns
    ///
    /// A matrix where column t contains α(t, ·).
    pub(crate) fn forward_probabilities(
        &self,
        observations: &[usize],
    ) -> Result<(DMatrix<T>, Vec<T>)> {
        if observations.is_empty() {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation sequence is empty".to_string(),
            });
        }

        let t_max = observations.len();
        let mut alpha = DMatrix::zeros(self.num_states, t_max);
        let mut scaling_factors = vec![T::zero(); t_max];
        let zero = T::zero();

        // Initialize: α(1, i) = πᵢ · Bᵢ,y₁
        let y0 = observations[0];
        if y0 >= self.num_observations {
            return Err(MarkovError::InvalidObservation {
                reason: format!("Observation {} out of bounds", y0),
            });
        }

        for i in 0..self.num_states {
            alpha[(i, 0)] = self.initial[i] * self.emissions[(i, y0)];
        }

        // Scale to prevent underflow
        let scale0: T = alpha.column(0).iter().fold(zero, |acc, &x| acc + x);
        if scale0 > zero {
            for i in 0..self.num_states {
                alpha[(i, 0)] /= scale0;
            }
        }
        scaling_factors[0] = scale0;

        // Recursion: α(t+1, j) = Σᵢ α(t, i) · Aᵢⱼ · Bⱼ,yₜ₊₁
        for t in 1..t_max {
            let y_t = observations[t];
            if y_t >= self.num_observations {
                return Err(MarkovError::InvalidObservation {
                    reason: format!("Observation {} out of bounds", y_t),
                });
            }

            for j in 0..self.num_states {
                let mut sum = zero;
                for i in 0..self.num_states {
                    sum += alpha[(i, t - 1)] * self.transitions[(i, j)];
                }
                alpha[(j, t)] = sum * self.emissions[(j, y_t)];
            }

            // Scale
            let scale_t: T = alpha.column(t).iter().fold(zero, |acc, &x| acc + x);
            if scale_t > zero {
                for i in 0..self.num_states {
                    alpha[(i, t)] /= scale_t;
                }
            }
            scaling_factors[t] = scale_t;
        }

        Ok((alpha, scaling_factors))
    }

    /// Computes backward probabilities β(t, i).
    ///
    /// # Mathematical Background
    ///
    /// Define β(t, i) = P(Yₜ₊₁, ..., Yₜ | Xₜ = i), the probability of
    /// observing the remaining observations given state i at time t.
    ///
    /// Recursion:
    /// - β(T, i) = 1
    /// - β(t, i) = Σⱼ Aᵢⱼ · Bⱼ,yₜ₊₁ · β(t+1, j)
    ///
    /// # Arguments
    ///
    /// * `observations` - Sequence of observation indices
    /// * `scaling_factors` - Scaling factors from forward pass
    ///
    /// # Returns
    ///
    /// A matrix where column t contains β(t, ·).
    pub(crate) fn backward_probabilities(
        &self,
        observations: &[usize],
        scaling_factors: &[T],
    ) -> Result<DMatrix<T>> {
        if observations.is_empty() {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation sequence is empty".to_string(),
            });
        }

        let t_max = observations.len();
        let mut beta = DMatrix::zeros(self.num_states, t_max);
        let one = T::one();
        let zero = T::zero();

        // Initialize: β(T, i) = 1
        let last_scale = scaling_factors[t_max - 1];
        let init_beta = if last_scale > zero {
            one / last_scale
        } else {
            zero
        };
        for i in 0..self.num_states {
            beta[(i, t_max - 1)] = init_beta;
        }

        // Recursion (backward)
        for t in (0..t_max - 1).rev() {
            let y_next = observations[t + 1];
            let scale_t = scaling_factors[t];

            for i in 0..self.num_states {
                let mut sum = zero;
                for j in 0..self.num_states {
                    sum +=
                        self.transitions[(i, j)] * self.emissions[(j, y_next)] * beta[(j, t + 1)];
                }
                beta[(i, t)] = if scale_t > zero { sum / scale_t } else { zero };
            }
        }

        Ok(beta)
    }

    /// Computes the posterior state probabilities γ(t, i) = P(Xₜ = i | Y₁, ..., Yₜ).
    ///
    /// # Mathematical Background
    ///
    /// Using forward and backward probabilities:
    /// γ(t, i) = α(t, i) · β(t, i) / P(Y₁, ..., Yₜ)
    ///
    /// # Arguments
    ///
    /// * `observations` - Sequence of observation indices
    ///
    /// # Returns
    ///
    /// A matrix where column t contains γ(t, ·).
    #[verified_engine::verified]
    pub fn posterior_probabilities(&self, observations: &[usize]) -> Result<DMatrix<T>> {
        let (alpha, scaling_factors) = self.forward_probabilities(observations)?;
        let beta = self.backward_probabilities(observations, &scaling_factors)?;

        let t_max = observations.len();
        let mut gamma = DMatrix::zeros(self.num_states, t_max);
        let zero = T::zero();

        for t in 0..t_max {
            for i in 0..self.num_states {
                gamma[(i, t)] = alpha[(i, t)] * beta[(i, t)];
            }

            // Normalize
            let sum: T = gamma.column(t).iter().fold(zero, |acc, &x| acc + x);
            if sum > zero {
                for i in 0..self.num_states {
                    gamma[(i, t)] /= sum;
                }
            }
        }

        Ok(gamma)
    }

    /// Computes the posterior state probabilities for the most recent observation (filtering).
    ///
    /// # Arguments
    ///
    /// * `observations` - Sequence of observation indices
    ///
    /// # Returns
    ///
    /// The posterior probabilities P(Xₜ = i | Y₁, ..., Yₜ) at the final time T.
    #[verified_engine::verified]
    pub fn filter(&self, observations: &[usize]) -> Result<DVector<T>> {
        let (alpha, _) = self.forward_probabilities(observations)?;
        let t_max = observations.len();

        let mut posterior = alpha.column(t_max - 1).into_owned();
        let zero = T::zero();
        let sum: T = posterior.iter().fold(zero, |acc, &x| acc + x);
        if sum > zero {
            posterior /= sum;
        }

        Ok(posterior)
    }
}
