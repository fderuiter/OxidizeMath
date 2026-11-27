//! Configuration, convergence report, and preallocated scratch workspace for Baum-Welch EM.
// [cite:clinical_trials_statistics]

use crate::error::MarkovError;
use bytemuck::{Pod, Zeroable};
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};
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

/// Summary returned by [`super::HiddenMarkovModel::baum_welch`].
#[derive(Debug, Clone, PartialEq)]
pub struct BaumWelchResult<T> {
    /// Log-likelihood trace $\ln P(Y \mid \theta^{(k)})$ at each accepted EM state.
    pub log_likelihoods: Vec<T>,
    /// Number of parameter update iterations completed.
    pub iterations: usize,
    /// True if consecutive log-likelihood improvement fell within `tolerance`.
    pub converged: bool,
}

/// Preallocated workspace buffers for zero-allocation Baum-Welch EM fitting.
#[derive(Debug, Clone)]
pub struct BaumWelchWorkspace<T> {
    pub(super) num_states: usize,
    pub(super) num_observations: usize,
    pub(super) max_len: usize,
    pub(super) arena: OwnedByteArena,
    pub(super) alpha: ArenaRegion<T>,
    pub(super) scales: ArenaRegion<T>,
    pub(super) beta_curr: ArenaRegion<T>,
    pub(super) beta_next: ArenaRegion<T>,
    pub(super) obs_beta: ArenaRegion<T>,
    pub(super) init_gamma: ArenaRegion<T>,
    pub(super) gamma_trans_sum: ArenaRegion<T>,
    pub(super) gamma_all_sum: ArenaRegion<T>,
    pub(super) xi_sum: ArenaRegion<T>,
    pub(super) emission_sum: ArenaRegion<T>,
    pub(super) prev_initial: ArenaRegion<T>,
    pub(super) prev_transitions: ArenaRegion<T>,
    pub(super) prev_emissions: ArenaRegion<T>,
}

impl<T: RealField + Copy + Pod + Zeroable> BaumWelchWorkspace<T> {
    /// Fallibly allocates scratch buffers in a single [`OwnedByteArena`] sized for `(num_states, num_observations, max_len)`.
    #[verified_engine::verified]
    pub fn try_new(
        num_states: usize,
        num_observations: usize,
        max_len: usize,
    ) -> Result<Self, math_memory::ArenaError> {
        let nn = num_states
            .checked_mul(num_states)
            .ok_or(math_memory::ArenaError::ArithmeticOverflow)?;
        let nm = num_states
            .checked_mul(num_observations)
            .ok_or(math_memory::ArenaError::ArithmeticOverflow)?;
        let tn = max_len
            .checked_mul(num_states)
            .ok_or(math_memory::ArenaError::ArithmeticOverflow)?;
        let mut layout = ArenaLayout::new();
        let alpha = layout.reserve::<T>(tn)?;
        let scales = layout.reserve::<T>(max_len)?;
        let beta_curr = layout.reserve::<T>(num_states)?;
        let beta_next = layout.reserve::<T>(num_states)?;
        let obs_beta = layout.reserve::<T>(num_states)?;
        let init_gamma = layout.reserve::<T>(num_states)?;
        let gamma_trans_sum = layout.reserve::<T>(num_states)?;
        let gamma_all_sum = layout.reserve::<T>(num_states)?;
        let xi_sum = layout.reserve::<T>(nn)?;
        let emission_sum = layout.reserve::<T>(nm)?;
        let prev_initial = layout.reserve::<T>(num_states)?;
        let prev_transitions = layout.reserve::<T>(nn)?;
        let prev_emissions = layout.reserve::<T>(nm)?;
        Ok(Self {
            num_states,
            num_observations,
            max_len,
            arena: OwnedByteArena::try_from_layout(&layout)?,
            alpha,
            scales,
            beta_curr,
            beta_next,
            obs_beta,
            init_gamma,
            gamma_trans_sum,
            gamma_all_sum,
            xi_sum,
            emission_sum,
            prev_initial,
            prev_transitions,
            prev_emissions,
        })
    }

    /// Allocates scratch buffers in a single [`OwnedByteArena`] sized for `(num_states, num_observations, max_len)`.
    #[verified_engine::verified]
    pub fn new(num_states: usize, num_observations: usize, max_len: usize) -> Self {
        Self::try_new(num_states, num_observations, max_len)
            .expect("BaumWelchWorkspace::new failed")
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
