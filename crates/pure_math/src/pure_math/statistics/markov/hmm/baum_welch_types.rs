//! Configuration, convergence report, and preallocated scratch workspace for Baum-Welch EM.
// [cite:clinical_trials_statistics]

use crate::error::MarkovError;
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena, Pod, Zeroable};
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

pub(super) struct BwSlices<'a, T> {
    pub(super) alpha: &'a mut [T],
    pub(super) scales: &'a mut [T],
    pub(super) beta_curr: &'a mut [T],
    pub(super) beta_next: &'a mut [T],
    pub(super) obs_beta: &'a mut [T],
    pub(super) init_gamma: &'a mut [T],
    pub(super) gamma_trans_sum: &'a mut [T],
    pub(super) gamma_all_sum: &'a mut [T],
    pub(super) xi_sum: &'a mut [T],
    pub(super) emission_sum: &'a mut [T],
    pub(super) prev_initial: &'a mut [T],
    pub(super) prev_transitions: &'a mut [T],
    pub(super) prev_emissions: &'a mut [T],
}

/// Preallocated workspace buffers for zero-allocation Baum-Welch EM fitting.
#[derive(Debug, Clone)]
pub struct BaumWelchWorkspace<T> {
    pub(super) num_states: usize,
    pub(super) num_observations: usize,
    pub(super) max_len: usize,
    pub(super) arena: OwnedByteArena,
    pub(super) init_gamma_reg: ArenaRegion<T>,
    pub(super) xi_sum_reg: ArenaRegion<T>,
    pub(super) emission_sum_reg: ArenaRegion<T>,
    pub(super) prev_initial_reg: ArenaRegion<T>,
    pub(super) prev_transitions_reg: ArenaRegion<T>,
    pub(super) prev_emissions_reg: ArenaRegion<T>,
}

impl<T: RealField + Copy + Pod + Zeroable> BaumWelchWorkspace<T> {
    /// Allocates scratch buffers sized for `(num_states, num_observations, max_len)`.
    #[verified_engine::verified]
    pub fn new(num_states: usize, num_observations: usize, max_len: usize) -> Self {
        let nn = num_states.saturating_mul(num_states);
        let nm = num_states.saturating_mul(num_observations);
        let tn = max_len.saturating_mul(num_states);
        let mut layout = ArenaLayout::new();
        let _ = layout.add_slice::<T>(tn).expect("bw alpha");
        let _ = layout.add_slice::<T>(max_len).expect("bw scales");
        let _ = layout.add_slice::<T>(num_states).expect("bw beta_curr");
        let _ = layout.add_slice::<T>(num_states).expect("bw beta_next");
        let _ = layout.add_slice::<T>(num_states).expect("bw obs_beta");
        let init_gamma_reg = layout.add_slice::<T>(num_states).expect("bw init_gamma");
        let _ = layout
            .add_slice::<T>(num_states)
            .expect("bw gamma_trans_sum");
        let _ = layout.add_slice::<T>(num_states).expect("bw gamma_all_sum");
        let xi_sum_reg = layout.add_slice::<T>(nn).expect("bw xi_sum");
        let emission_sum_reg = layout.add_slice::<T>(nm).expect("bw emission_sum");
        let prev_initial_reg = layout.add_slice::<T>(num_states).expect("bw prev_initial");
        let prev_transitions_reg = layout.add_slice::<T>(nn).expect("bw prev_transitions");
        let prev_emissions_reg = layout.add_slice::<T>(nm).expect("bw prev_emissions");
        Self {
            num_states,
            num_observations,
            max_len,
            arena: OwnedByteArena::with_capacity(layout.total_bytes()),
            init_gamma_reg,
            xi_sum_reg,
            emission_sum_reg,
            prev_initial_reg,
            prev_transitions_reg,
            prev_emissions_reg,
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

    pub(super) fn slices_mut(&mut self) -> BwSlices<'_, T> {
        let n = self.num_states;
        let nn = n.saturating_mul(n);
        let nm = n.saturating_mul(self.num_observations);
        let tn = self.max_len.saturating_mul(n);
        let mut scope = self.arena.reborrow();
        BwSlices {
            alpha: scope.retain_slice::<T>(tn).expect("alpha"),
            scales: scope.retain_slice::<T>(self.max_len).expect("scales"),
            beta_curr: scope.retain_slice::<T>(n).expect("beta_curr"),
            beta_next: scope.retain_slice::<T>(n).expect("beta_next"),
            obs_beta: scope.retain_slice::<T>(n).expect("obs_beta"),
            init_gamma: scope.retain_slice::<T>(n).expect("init_gamma"),
            gamma_trans_sum: scope.retain_slice::<T>(n).expect("gamma_trans_sum"),
            gamma_all_sum: scope.retain_slice::<T>(n).expect("gamma_all_sum"),
            xi_sum: scope.retain_slice::<T>(nn).expect("xi_sum"),
            emission_sum: scope.retain_slice::<T>(nm).expect("emission_sum"),
            prev_initial: scope.retain_slice::<T>(n).expect("prev_initial"),
            prev_transitions: scope.retain_slice::<T>(nn).expect("prev_transitions"),
            prev_emissions: scope.retain_slice::<T>(nm).expect("prev_emissions"),
        }
    }
}
