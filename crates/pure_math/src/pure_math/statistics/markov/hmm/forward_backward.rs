use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Computes the exact log-likelihood $\ln P(Y_1, \dots, Y_T)$ from forward scaling factors.
    ///
    /// Returns $-\infty$ when the observation sequence has zero probability under the model.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidObservation`] if `observations` is empty or out of bounds.
    #[verified_engine::verified]
    pub fn forward_log_likelihood(&self, observations: &[usize]) -> Result<T> {
        let (_, scaling_factors) = self.forward_probabilities(observations)?;
        let zero = T::zero();
        let neg_inf = T::from_f64(f64::NEG_INFINITY).unwrap();

        let mut log_lik = zero;
        for &c_t in &scaling_factors {
            if c_t <= zero {
                return Ok(neg_inf);
            }
            log_lik += c_t.ln();
        }
        Ok(log_lik)
    }

    /// Computes $P(Y_1, \dots, Y_T) = \exp(\text{forward\_log\_likelihood})$ using the scaled forward pass.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidObservation`] if `observations` is empty or out of bounds.
    #[verified_engine::verified]
    pub fn forward(&self, observations: &[usize]) -> Result<T> {
        let log_lik = self.forward_log_likelihood(observations)?;
        if !log_lik.is_finite() {
            return Ok(T::zero());
        }
        Ok(log_lik.exp())
    }

    /// Computes scaled forward probabilities $\hat{\alpha}(t, i)$ and step scaling factors $c_t$.
    #[verified_engine::verified]
    pub(crate) fn forward_probabilities(
        &self,
        observations: &[usize],
    ) -> Result<(DMatrix<T>, Vec<T>)> {
        self.validate_observations(observations)?;

        let t_max = observations.len();
        let mut alpha = DMatrix::zeros(self.num_states, t_max);
        let mut scaling_factors = vec![T::zero(); t_max];
        let zero = T::zero();

        let y0 = observations[0];
        for i in 0..self.num_states {
            alpha[(i, 0)] = self.initial[i] * self.emissions[(i, y0)];
        }

        let scale0: T = alpha.column(0).iter().fold(zero, |acc, &x| acc + x);
        scaling_factors[0] = scale0;
        if scale0 <= zero {
            return Ok((alpha, scaling_factors));
        }
        for i in 0..self.num_states {
            alpha[(i, 0)] /= scale0;
        }

        for (t, &y_t) in observations.iter().enumerate().skip(1) {
            for j in 0..self.num_states {
                let mut sum = zero;
                for i in 0..self.num_states {
                    sum += alpha[(i, t - 1)] * self.transitions[(i, j)];
                }
                alpha[(j, t)] = sum * self.emissions[(j, y_t)];
            }

            let scale_t: T = alpha.column(t).iter().fold(zero, |acc, &x| acc + x);
            scaling_factors[t] = scale_t;
            if scale_t <= zero {
                return Ok((alpha, scaling_factors));
            }
            for i in 0..self.num_states {
                alpha[(i, t)] /= scale_t;
            }
        }

        Ok((alpha, scaling_factors))
    }

    #[verified_engine::verified]
    pub(crate) fn validate_observations(&self, observations: &[usize]) -> Result<()> {
        if observations.is_empty() {
            return Err(MarkovError::InvalidObservation {
                reason: "Observation sequence is empty".to_string(),
            });
        }
        for &obs in observations {
            if obs >= self.num_observations {
                return Err(MarkovError::InvalidObservation {
                    reason: format!("Observation {obs} out of bounds"),
                });
            }
        }
        Ok(())
    }

    /// Computes scaled backward probabilities $\hat{\beta}(t, i)$.
    #[verified_engine::verified]
    pub(crate) fn backward_probabilities(
        &self,
        observations: &[usize],
        scaling_factors: &[T],
    ) -> Result<DMatrix<T>> {
        self.validate_observations(observations)?;

        let t_max = observations.len();
        let mut beta = DMatrix::zeros(self.num_states, t_max);
        let one = T::one();
        let zero = T::zero();

        let last_scale = scaling_factors[t_max - 1];
        if last_scale <= zero {
            return Ok(beta);
        }
        for i in 0..self.num_states {
            beta[(i, t_max - 1)] = one / last_scale;
        }

        for t in (0..t_max - 1).rev() {
            let scale_t = scaling_factors[t];
            if scale_t <= zero {
                break;
            }
            let y_next = observations[t + 1];
            for i in 0..self.num_states {
                let mut sum = zero;
                for j in 0..self.num_states {
                    sum +=
                        self.transitions[(i, j)] * self.emissions[(j, y_next)] * beta[(j, t + 1)];
                }
                beta[(i, t)] = sum / scale_t;
            }
        }

        Ok(beta)
    }

    /// Computes the posterior state probabilities $\gamma(t, i) = P(X_t = i \mid Y_1, \dots, Y_T)$.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidObservation`] if `observations` is empty or out of bounds.
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
            let sum: T = gamma.column(t).iter().fold(zero, |acc, &x| acc + x);
            if sum > zero {
                for i in 0..self.num_states {
                    gamma[(i, t)] /= sum;
                }
            }
        }

        Ok(gamma)
    }

    /// Computes filtering probabilities $P(X_T = i \mid Y_1, \dots, Y_T)$ at the final step $T$.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidObservation`] if `observations` is empty or out of bounds.
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

// [cite:clinical_trials_statistics]
