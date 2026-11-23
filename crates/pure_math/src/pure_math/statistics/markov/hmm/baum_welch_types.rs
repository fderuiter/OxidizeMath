//! Configuration, convergence report, and preallocated scratch workspace for Baum-Welch EM.
// [cite:clinical_trials_statistics]

use crate::error::MarkovError;
use nalgebra::RealField;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

/// Hyperparameters controlling Baum-Welch EM training.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaumWelchConfig<T> {
    /// Maximum number of EM iterations (`>= 1`).
    pub max_iterations: usize,
    /// Non-negative log-likelihood improvement threshold for early stopping.
    pub tolerance: T,
    /// Non-negative symmetric Dirichlet prior pseudocount (`0` = pure MLE).
    pub dirichlet_alpha: T,
}

impl Default for BaumWelchConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            dirichlet_alpha: 0.0,
        }
    }
}

impl<T: RealField + Copy + ToPrimitive> BaumWelchConfig<T> {
    /// Validates and constructs a [`BaumWelchConfig`].
    #[verified_engine::verified]
    pub fn new(
        max_iterations: usize,
        tolerance: T,
        dirichlet_alpha: T,
    ) -> Result<Self, MarkovError> {
        if max_iterations == 0 {
            return Err(MarkovError::InvalidState {
                reason: "max_iterations must be at least 1".to_string(),
            });
        }
        let zero = T::zero();
        if !tolerance.is_finite() || tolerance < zero {
            return Err(MarkovError::InvalidState {
                reason: "tolerance must be finite and non-negative".to_string(),
            });
        }
        if !dirichlet_alpha.is_finite() || dirichlet_alpha < zero {
            return Err(MarkovError::InvalidState {
                reason: "dirichlet_alpha must be finite and non-negative".to_string(),
            });
        }
        Ok(Self {
            max_iterations,
            tolerance,
            dirichlet_alpha,
        })
    }
}

/// Convergence and log-likelihood summary produced by Baum-Welch EM fitting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaumWelchReport<T> {
    /// Number of EM iterations executed.
    pub iterations: usize,
    /// Whether convergence was achieved within `max_iterations`.
    pub converged: bool,
    /// Log-likelihood prior to the first parameter update.
    pub initial_log_likelihood: T,
    /// Log-likelihood of the final fitted model.
    pub final_log_likelihood: T,
}

/// Preallocated workspace buffers for zero-allocation Baum-Welch EM fitting.
#[derive(Debug, Clone)]
pub struct BaumWelchWorkspace<T> {
    pub(super) num_states: usize,
    pub(super) num_observations: usize,
    pub(super) max_len: usize,
    pub(super) alpha: Vec<T>,
    pub(super) scales: Vec<T>,
    pub(super) beta_curr: Vec<T>,
    pub(super) beta_next: Vec<T>,
    pub(super) obs_beta: Vec<T>,
    pub(super) init_gamma: Vec<T>,
    pub(super) gamma_trans_sum: Vec<T>,
    pub(super) gamma_all_sum: Vec<T>,
    pub(super) xi_sum: Vec<T>,
    pub(super) emission_sum: Vec<T>,
    pub(super) prev_initial: Vec<T>,
    pub(super) prev_transitions: Vec<T>,
    pub(super) prev_emissions: Vec<T>,
}

impl<T: RealField + Copy> BaumWelchWorkspace<T> {
    /// Allocates scratch buffers sized for `(num_states, num_observations, max_len)`.
    #[verified_engine::verified]
    pub fn new(num_states: usize, num_observations: usize, max_len: usize) -> Self {
        let z = T::zero();
        let nn = num_states.saturating_mul(num_states);
        let nm = num_states.saturating_mul(num_observations);
        let tn = max_len.saturating_mul(num_states);
        Self {
            num_states,
            num_observations,
            max_len,
            alpha: vec![z; tn],
            scales: vec![z; max_len],
            beta_curr: vec![z; num_states],
            beta_next: vec![z; num_states],
            obs_beta: vec![z; num_states],
            init_gamma: vec![z; num_states],
            gamma_trans_sum: vec![z; num_states],
            gamma_all_sum: vec![z; num_states],
            xi_sum: vec![z; nn],
            emission_sum: vec![z; nm],
            prev_initial: vec![z; num_states],
            prev_transitions: vec![z; nn],
            prev_emissions: vec![z; nm],
        }
    }

    /// Grows internal scratch buffers if needed to support `(num_states, num_observations, max_len)`.
    #[verified_engine::verified]
    pub fn ensure_capacity(&mut self, num_states: usize, num_observations: usize, max_len: usize) {
        let n = self.num_states.max(num_states);
        let m = self.num_observations.max(num_observations);
        let t = self.max_len.max(max_len);
        if n > self.num_states || m > self.num_observations || t > self.max_len {
            *self = Self::new(n, m, t);
        }
    }
}
