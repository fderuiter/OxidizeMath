//! Stationary and limiting distribution solvers for [`MarkovChain`].
// [cite:clinical_trials_statistics]

use super::{MarkovChain, Result};
use crate::error::MarkovError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use math_commons::stochastic::SolverWorkspace;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
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
    #[verified_engine::verified(alloc_ok = "allocates workspace and output vector")]
    pub fn stationary_distribution(&self) -> Option<DVector<T>> {
        let n = self.num_states();
        assert!(n > 0);
        let mut ws = SolverWorkspace::new(n, 0);
        let mut out = DVector::zeros(n);
        self.stationary_distribution_with_workspace(&mut ws, &mut out)
            .then_some(out)
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
                    staged[math_commons::math_kernel::types::flatten_2d_index(c, r, n)] = v;
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
    #[verified_engine::verified(alloc_ok = "formats error message on invalid input")]
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
        if !self.has_aperiodic_unique_sink() {
            return Err(MarkovError::InvalidState {
                reason: "Periodic Markov chain has no limiting distribution".to_string(),
            });
        }
        Ok(pi)
    }

    #[verified_engine::verified(alloc_ok = "formats error message on invalid input")]
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

    #[verified_engine::verified(alloc_ok = "allocates owned output vector")]
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

    fn has_aperiodic_unique_sink(&self) -> bool {
        let n = self.num_states();
        let mut flat = vec![0.0_f64; n * n];
        for r in 0..n {
            for c in 0..n {
                let Some(v) = self.transition_matrix[(r, c)].to_f64() else {
                    return false;
                };
                flat[math_commons::math_kernel::types::flatten_2d_index(c, r, n)] = v;
            }
        }
        let mut ws = SolverWorkspace::new(n, 0);
        let Some(scc) = ws.analyze_scc(&flat, n) else {
            return false;
        };
        if scc.num_sink_components != 1 || scc.unique_sink_size == 0 {
            return false;
        }
        let sink = ws.unique_sink_members(scc.unique_sink_size);
        is_sink_primitive(&flat, n, sink)
    }
}

/// Checks primitivity (aperiodicity) of an irreducible $m \times m$ sink submatrix via
/// Wielandt's bound $k = m^2 - 2m + 2$ using repeated squaring.
fn is_sink_primitive(flat: &[f64], n: usize, sink: &[usize]) -> bool {
    let m = sink.len();
    if m <= 1 {
        return m == 1;
    }
    let mut base = DMatrix::<f64>::zeros(m, m);
    for (r, &sr) in sink.iter().enumerate() {
        for (c, &sc) in sink.iter().enumerate() {
            let prob = flat[math_commons::math_kernel::types::flatten_2d_index(sc, sr, n)];
            base[(r, c)] = if prob > 1e-14 { 1.0 } else { 0.0 };
        }
    }
    let mut exp = m * m - 2 * m + 2;
    let mut acc = DMatrix::<f64>::identity(m, m);
    while exp > 0 {
        if exp & 1 == 1 {
            acc = boolean_matmul(&acc, &base);
        }
        exp >>= 1;
        if exp > 0 {
            base = boolean_matmul(&base, &base);
        }
    }
    acc.iter().all(|&v| v > 0.0)
}

fn boolean_matmul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let p = a * b;
    p.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
}
