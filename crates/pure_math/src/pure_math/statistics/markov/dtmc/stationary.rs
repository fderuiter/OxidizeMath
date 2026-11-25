//! Stationary and limiting distribution solvers for [`MarkovChain`].

use super::{MarkovChain, Result};
use crate::error::MarkovError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, flatten_2d_index};
use math_commons::stochastic::SolverWorkspace;
use nalgebra::{DVector, RealField};
use num_traits::ToPrimitive;

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
    /// Fallible companion to [`stationary_distribution`](Self::stationary_distribution).
    #[verified_engine::verified]
    pub fn try_stationary_distribution(
        &self,
    ) -> core::result::Result<Option<DVector<T>>, math_commons::error::MathError> {
        #[allow(unused_braces)]
        Ok({
            let n = self.num_states();
            __ensure_math(
                n > 0,
                "stationary_distribution: precondition or invariant violated",
            )?;
            let mut ws = SolverWorkspace::new(n, 0);
            let mut out = DVector::zeros(n);
            self.stationary_distribution_with_workspace(&mut ws, &mut out)
                .then_some(out)
        })
    }

    /// Computes the unique stationary distribution $\pi$ satisfying $\pi^T P = \pi^T$ and
    /// $\sum_i \pi_i = 1$ via Tarjan SCC sink isolation and Householder QR.
    ///
    /// Returns `None` when multiple closed sink classes exist (e.g. multi-absorbing chains where
    /// stationary weights are non-unique). Handles periodic irreducible chains (such as bipartite
    /// cycles) directly without power-iteration oscillation.
    ///
    /// # Examples
    /// ```rust
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
    /// use math_explorer::nalgebra::DMatrix;
    ///
    /// let transition_matrix = DMatrix::from_row_slice(2, 2, &[
    ///     0.8, 0.2,
    ///     0.4, 0.6,
    /// ]);
    /// let state_types = vec![StateType::Transient, StateType::Transient];
    /// let chain = MarkovChain::<f64>::new(transition_matrix, state_types).unwrap();
    ///
    /// let pi = chain.stationary_distribution().unwrap();
    /// assert!((pi[0] - 2.0 / 3.0).abs() < math_explorer::registry::TOLERANCE_HIGH);
    /// assert!((pi[1] - 1.0 / 3.0).abs() < math_explorer::registry::TOLERANCE_HIGH);
    /// ```
    #[must_use]
    #[verified_engine::verified]
    pub fn stationary_distribution(&self) -> Option<DVector<T>> {
        self.try_stationary_distribution()
            .expect("stationary_distribution preconditions must hold")
    }

    /// Computes the unique stationary distribution into `out` using preallocated [`SolverWorkspace`]
    /// buffers with zero heap allocations when `ws.state_capacity() >= self.num_states()`.
    #[verified_engine::verified]
    pub fn stationary_distribution_with_workspace(
        &self,
        ws: &mut SolverWorkspace,
        out: &mut DVector<T>,
    ) -> bool {
        let n = self.num_states();
        debug_assert_eq!(out.len(), n);
        if n == 0 || out.len() != n {
            return false;
        }
        {
            let staged = ws.operator_staging_mut(n);
            for r in 0..n {
                for c in 0..n {
                    let Some(v) = self.transition_matrix[(r, c)].to_f64() else {
                        return false;
                    };
                    staged[flatten_2d_index(c, r, n)] = v;
                }
            }
        }
        let Some(pi_f64) = ws.solve_staged_dtmc_stationary_slice(n) else {
            return false;
        };
        for (i, &val) in pi_f64.iter().enumerate() {
            let Some(cast) = T::from_f64(val) else {
                return false;
            };
            out[i] = cast;
        }
        true
    }

    /// Maps an initial state probability vector $\pi_0$ to its asymptotic limiting distribution
    /// $\lim_{n \to \infty} \pi_0^T P^n$.
    ///
    /// For absorbing chains, transient mass is mapped through [`Self::absorption_probabilities`]
    /// ($B = N \cdot R$) onto absorbing states ($\pi_{\infty}(a_j) = \pi_0(a_j) + \sum_i \pi_0(t_i) B_{ij}$,
    /// $\pi_{\infty}(t_i) = 0$). For non-absorbing chains, returns the unique stationary distribution
    /// when the unique closed recurrent class is aperiodic, or `Err(MarkovError::InvalidState)`
    /// when periodicity or multiple non-absorbing closed classes prevent convergence.
    ///
    /// # Errors
    ///
    /// Returns an error if `initial` is not a valid probability vector on `self.num_states()`,
    /// $(I - Q)$ is singular, or the chain has no limiting distribution due to periodicity or
    /// multiple non-absorbing closed classes.
    #[verified_engine::verified]
    pub fn limiting_distribution(&self, initial: &DVector<T>) -> Result<DVector<T>> {
        let n = self.num_states();
        debug_assert!(n > 0);
        self.validate_initial_distribution(initial)?;

        if !self.absorbing_indices.is_empty() {
            return self.limiting_via_absorption(initial);
        }

        let Some(pi) = self.stationary_distribution() else {
            return Err(MarkovError::InvalidState {
                reason: "Chain has multiple closed recurrent classes and no absorbing states"
                    .to_string(),
            });
        };
        if self.period() != Some(1) {
            return Err(MarkovError::InvalidState {
                reason: "Periodic Markov chain has no limiting distribution".to_string(),
            });
        }
        Ok(pi)
    }

    /// Convenience wrapper computing [`Self::limiting_distribution`] starting deterministically
    /// in `initial_state`, returning `None` when `initial_state` is out of bounds or no limit exists.
    #[must_use]
    #[verified_engine::verified]
    pub fn limiting_distribution_from_state(&self, initial_state: usize) -> Option<DVector<T>> {
        let n = self.num_states();
        if initial_state >= n {
            return None;
        }
        let mut e = DVector::zeros(n);
        e[initial_state] = T::one();
        self.limiting_distribution(&e).ok()
    }

    #[verified_engine::verified]
    fn validate_initial_distribution(&self, initial: &DVector<T>) -> Result<()> {
        let n = self.num_states();
        debug_assert!(n > 0);
        if initial.len() != n {
            return Err(MarkovError::Math(MathError::DimensionMismatch {
                expected: Dimension(n),
                actual: Dimension(initial.len()),
            }));
        }
        let mut sum = 0.0_f64;
        let tol_hi = math_commons::registry::TOLERANCE_HIGH;
        for i in 0..n {
            let val = initial[i].to_f64().unwrap_or(f64::NAN);
            if !val.is_finite() || val < -tol_hi || val > 1.0 + tol_hi {
                return Err(MarkovError::InvalidProbability { value: val });
            }
            sum += val;
        }
        if (sum - 1.0).abs() > math_commons::registry::TOLERANCE_FAST {
            return Err(MarkovError::NotStochastic {
                reason: format!("Initial distribution sums to {sum}, expected 1.0"),
            });
        }
        Ok(())
    }

    #[verified_engine::verified]
    fn limiting_via_absorption(&self, initial: &DVector<T>) -> Result<DVector<T>> {
        let n = self.num_states();
        debug_assert!(!self.absorbing_indices.is_empty());
        if self.transient_indices.is_empty() {
            return Ok(initial.clone());
        }
        let b = self.absorption_probabilities()?;
        let mut limiting = DVector::zeros(n);
        for (j, &abs_idx) in self.absorbing_indices.iter().enumerate() {
            let mut mass = initial[abs_idx];
            for (i, &tr_idx) in self.transient_indices.iter().enumerate() {
                mass += initial[tr_idx] * b[(i, j)];
            }
            limiting[abs_idx] = mass;
        }
        Ok(limiting)
    }
}

// [cite:clinical_trials_statistics]

#[inline]
fn __ensure_math(
    cond: bool,
    reason: &'static str,
) -> core::result::Result<(), math_commons::error::MathError> {
    if cond {
        Ok(())
    } else {
        Err(math_commons::error::MathError::NumericalError { reason })
    }
}
