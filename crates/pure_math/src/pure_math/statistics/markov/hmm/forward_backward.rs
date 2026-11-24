use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use engine_graph::{ChainFbWorkspace, ChainModelView, chain_forward_backward_into};
use engine_prob::discrete_posterior_into;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

#[inline]
const fn idx2d(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

struct EngineFbBuffers {
    log_likelihood: f64,
    alpha: Vec<f64>,
    gamma: Vec<f64>,
}

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
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

    fn export_f64_parameters(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.num_states;
        let m = self.num_observations;
        let initial: Vec<f64> = self
            .initial
            .iter()
            .map(|&v| v.to_f64().unwrap_or(0.0))
            .collect();
        let mut transitions = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                transitions[idx2d(i, j, n)] = self.transitions[(i, j)].to_f64().unwrap_or(0.0);
            }
        }
        let mut emissions = vec![0.0; n * m];
        for i in 0..n {
            for k in 0..m {
                emissions[idx2d(i, k, m)] = self.emissions[(i, k)].to_f64().unwrap_or(0.0);
            }
        }
        (initial, transitions, emissions)
    }

    fn run_engine_forward_backward(&self, observations: &[usize]) -> Result<EngineFbBuffers> {
        self.validate_observations(observations)?;
        let n = self.num_states;
        let t_len = observations.len();
        let mat_len = t_len * n;
        let (initial, transitions, emissions) = self.export_f64_parameters();
        let mut alpha = vec![0.0; mat_len];
        let mut beta = vec![0.0; mat_len];
        let mut scales = vec![0.0; t_len];
        let mut gamma = vec![0.0; mat_len];
        let view = ChainModelView {
            num_states: n,
            num_obs: self.num_observations,
            initial: &initial,
            transitions: &transitions,
            emissions: &emissions,
        };
        let mut fb_ws = ChainFbWorkspace {
            alpha: &mut alpha,
            beta: &mut beta,
            scales: &mut scales,
            gamma: &mut gamma,
        };
        let log_likelihood =
            chain_forward_backward_into(&view, observations, &mut fb_ws).map_err(|err| {
                MarkovError::InvalidObservation {
                    reason: err.to_string(),
                }
            })?;
        Ok(EngineFbBuffers {
            log_likelihood,
            alpha,
            gamma,
        })
    }

    /// Computes $\ln P(Y_1, \dots, Y_T)$ via [`engine_graph::chain_forward_backward_into`].
    #[verified_engine::verified]
    pub fn log_likelihood(&self, observations: &[usize]) -> Result<T> {
        let fb = self.run_engine_forward_backward(observations)?;
        if fb.log_likelihood.is_finite() {
            Ok(T::from_f64(fb.log_likelihood).unwrap_or_else(T::zero))
        } else {
            let zero = T::zero();
            Ok(T::from_f64(f64::NEG_INFINITY)
                .or_else(T::min_value)
                .unwrap_or_else(|| zero.ln()))
        }
    }

    /// Computes the exact log-likelihood $\ln P(Y_1, \dots, Y_T)$ from forward scaling factors.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidObservation`] if `observations` is empty or out of bounds.
    #[verified_engine::verified]
    pub fn forward_log_likelihood(&self, observations: &[usize]) -> Result<T> {
        self.log_likelihood(observations)
    }

    /// Computes $P(Y_1, \dots, Y_T)$ using the scaled forward algorithm in [`engine_graph`].
    ///
    /// With conditional scaling factors $s_t = P(Y_t \mid Y_{1:t-1})$, the exact observation
    /// likelihood is $P(Y_{1:T}) = \prod_{t=0}^{T-1} s_t = \exp\!\left(\sum_{t=0}^{T-1} \ln s_t\right)$.
    #[verified_engine::verified]
    pub fn forward(&self, observations: &[usize]) -> Result<T> {
        let ll = self.log_likelihood(observations)?;
        if ll.is_finite() {
            Ok(ll.exp())
        } else {
            Ok(T::zero())
        }
    }

    /// Computes posterior state probabilities $\gamma(t, i) = P(X_t = i \mid Y_{1:T})$
    /// via [`engine_graph::chain_forward_backward_into`].
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidObservation`] if `observations` is empty or out of bounds.
    #[verified_engine::verified]
    pub fn posterior_probabilities(&self, observations: &[usize]) -> Result<DMatrix<T>> {
        let fb = self.run_engine_forward_backward(observations)?;
        let n = self.num_states;
        let t_max = observations.len();
        let mut gamma = DMatrix::zeros(n, t_max);
        for t in 0..t_max {
            for i in 0..n {
                gamma[(i, t)] = T::from_f64(fb.gamma[idx2d(t, i, n)]).unwrap_or_else(T::zero);
            }
        }
        Ok(gamma)
    }

    /// Computes filtered posterior state probabilities $P(X_T = i \mid Y_{1:T})$ at the final step
    /// via [`engine_graph::chain_forward_backward_into`] and [`engine_prob::discrete_posterior_into`].
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidObservation`] if `observations` is empty or out of bounds.
    #[verified_engine::verified]
    pub fn filter(&self, observations: &[usize]) -> Result<DVector<T>> {
        let fb = self.run_engine_forward_backward(observations)?;
        let n = self.num_states;
        let last_t = observations.len() - 1;
        let start = idx2d(last_t, 0, n);
        let uniform = vec![1.0; n];
        let mut post_f64 = vec![0.0; n];
        discrete_posterior_into(&fb.alpha[start..start + n], &uniform, &mut post_f64).map_err(
            |err| {
                MarkovError::Math(math_commons::error::MathError::NumericalError {
                    reason: err.to_string(),
                })
            },
        )?;
        Ok(DVector::from_iterator(
            n,
            post_f64
                .into_iter()
                .map(|p| T::from_f64(p).unwrap_or_else(T::zero)),
        ))
    }
}

// [cite:clinical_trials_statistics]
