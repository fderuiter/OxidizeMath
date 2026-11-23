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
    pub(super) prev_initial: Vec<T>,
    pub(super) prev_transitions: Vec<T>,
    pub(super) prev_emissions: Vec<T>,
    pub(super) f64_initial: Vec<f64>,
    pub(super) f64_transitions: Vec<f64>,
    pub(super) f64_emissions: Vec<f64>,
    pub(super) f64_alpha: Vec<f64>,
    pub(super) f64_beta: Vec<f64>,
    pub(super) f64_scales: Vec<f64>,
    pub(super) f64_gamma: Vec<f64>,
    pub(super) f64_xi_sum: Vec<f64>,
    pub(super) f64_emission_sum: Vec<f64>,
    pub(super) f64_row_in: Vec<f64>,
    pub(super) f64_row_ones: Vec<f64>,
    pub(super) f64_row_out: Vec<f64>,
}

impl<T: RealField + Copy> BaumWelchWorkspace<T> {
    /// Allocates scratch buffers sized for `(num_states, num_observations, max_len)`.
    #[verified_engine::verified]
    pub fn new(num_states: usize, num_observations: usize, max_len: usize) -> Self {
        let z = T::zero();
        let nn = num_states.saturating_mul(num_states);
        let nm = num_states.saturating_mul(num_observations);
        let tn = max_len.saturating_mul(num_states);
        let row_cap = num_states.max(num_observations).max(1);
        Self {
            num_states,
            num_observations,
            max_len,
            prev_initial: vec![z; num_states],
            prev_transitions: vec![z; nn],
            prev_emissions: vec![z; nm],
            f64_initial: vec![0.0; num_states],
            f64_transitions: vec![0.0; nn],
            f64_emissions: vec![0.0; nm],
            f64_alpha: vec![0.0; tn],
            f64_beta: vec![0.0; tn],
            f64_scales: vec![0.0; max_len],
            f64_gamma: vec![0.0; tn],
            f64_xi_sum: vec![0.0; nn],
            f64_emission_sum: vec![0.0; nm],
            f64_row_in: vec![0.0; row_cap],
            f64_row_ones: vec![1.0; row_cap],
            f64_row_out: vec![0.0; row_cap],
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
