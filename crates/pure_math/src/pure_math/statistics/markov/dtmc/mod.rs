//! Discrete-Time Markov Chains (DTMC).
//!
//! This module implements discrete-time Markov chains with support for:
//! - Transient and absorbing states
//! - Tarjan strongly connected communicating classes (`scc`)
//! - Period detection (`period`) and mean hitting times (`hitting`)
//! - Canonical form decomposition & fundamental matrix computation
//! - Expected Possession Value (EPV) calculation
//! - QR linear nullspace stationary and limiting distributions (`stationary`)
//! - Zero-allocation Monte Carlo stepping (`simulation`)

use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

/// Mean first-passage / hitting times solver.
pub mod hitting;
/// State and chain periodicity detection via BFS chord GCD.
pub mod period;
/// Iterative Tarjan strongly connected components and communicating classes.
pub mod scc;
/// Zero-allocation Monte Carlo trajectory and visit-count simulation.
pub mod simulation;
/// QR nullspace stationary and absorbing/aperiodic limiting distributions.
pub mod stationary;
mod types;

pub use scc::CommunicatingClass;
pub use types::StateType;

/// A discrete-time Markov chain.
///
/// # Mathematical Background
///
/// A discrete-time Markov chain is a stochastic process {Xₙ} with the Markov property:
/// P(Xₙ₊₁ = j | X₀, X₁, ..., Xₙ) = P(Xₙ₊₁ = j | Xₙ)
///
/// The chain is characterized by a transition matrix P where `P[i,j]` = P(Xₙ₊₁ = j | Xₙ = i).
///
/// For chains with absorbing states, the canonical form is:
/// ```text
/// P = [ Q  R ]
///     [ 0  I ]
/// ```
///
/// # Example
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
/// use math_explorer::nalgebra::DMatrix;
///
/// let transition_matrix = DMatrix::from_row_slice(3, 3, &[
///     0.7, 0.2, 0.1,
///     0.3, 0.5, 0.2,
///     0.0, 0.0, 1.0,
/// ]);
/// let state_types = vec![
///     StateType::Transient,
///     StateType::Transient,
///     StateType::Absorbing,
/// ];
/// let chain = MarkovChain::<f64>::new(transition_matrix, state_types).unwrap();
/// let fundamental = chain.fundamental_matrix().unwrap();
/// let absorption_probs = chain.absorption_probabilities().unwrap();
/// assert_eq!(fundamental.nrows(), 2);
/// assert_eq!(absorption_probs.nrows(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct MarkovChain<T: RealField + Copy + ToPrimitive> {
    /// The transition matrix P.
    pub(crate) transition_matrix: DMatrix<T>,
    /// Type of each state (transient or absorbing).
    pub(crate) state_types: Vec<StateType>,
    /// Indices of transient states.
    pub(crate) transient_indices: Vec<usize>,
    /// Indices of absorbing states.
    pub(crate) absorbing_indices: Vec<usize>,
}

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
    /// Creates a new Markov chain.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If matrix size doesn't match `state_types` length
    /// - `NotStochastic`: If matrix rows don't sum to 1
    /// - `InvalidState`: If a state marked `Absorbing` does not have $P_{ii} = 1$
    #[verified_engine::verified]
    pub fn new(transition_matrix: DMatrix<T>, state_types: Vec<StateType>) -> Result<Self> {
        Self::validate_dimensions(&transition_matrix, &state_types)?;
        crate::pure_math::statistics::markov::validation::validate_stochastic_matrix(
            &transition_matrix,
        )?;
        Self::validate_absorbing_rows(&transition_matrix, &state_types)?;
        let (transient_indices, absorbing_indices) = Self::partition_state_indices(&state_types);

        Ok(MarkovChain {
            transition_matrix,
            state_types,
            transient_indices,
            absorbing_indices,
        })
    }

    #[verified_engine::verified]
    fn validate_dimensions(
        transition_matrix: &DMatrix<T>,
        state_types: &[StateType],
    ) -> Result<()> {
        let n = transition_matrix.nrows();
        if transition_matrix.ncols() != n {
            return Err(MarkovError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(n),
                    actual: math_commons::math_kernel::types::Dimension(transition_matrix.ncols()),
                },
            ));
        }
        if state_types.len() != n {
            return Err(MarkovError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(n),
                    actual: math_commons::math_kernel::types::Dimension(state_types.len()),
                },
            ));
        }
        Ok(())
    }

    #[verified_engine::verified]
    fn validate_absorbing_rows(
        transition_matrix: &DMatrix<T>,
        state_types: &[StateType],
    ) -> Result<()> {
        let n = transition_matrix.nrows();
        let tolerance = T::from_f64(math_commons::registry::TOLERANCE_HIGH).unwrap();
        for (i, &state_type) in state_types.iter().enumerate() {
            if state_type == StateType::Absorbing {
                for j in 0..n {
                    let expected = if i == j { T::one() } else { T::zero() };
                    let actual = transition_matrix[(i, j)];
                    if (actual - expected).abs() > tolerance {
                        return Err(MarkovError::InvalidState {
                            reason: format!(
                                "State {} marked as absorbing but P[{},{}] = {} != {}",
                                i,
                                i,
                                j,
                                actual.to_f64().unwrap_or(f64::NAN),
                                expected.to_f64().unwrap_or(f64::NAN)
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    #[verified_engine::verified]
    fn partition_state_indices(state_types: &[StateType]) -> (Vec<usize>, Vec<usize>) {
        let mut transient_indices = Vec::new();
        let mut absorbing_indices = Vec::new();
        for (i, &state_type) in state_types.iter().enumerate() {
            match state_type {
                StateType::Transient => transient_indices.push(i),
                StateType::Absorbing => absorbing_indices.push(i),
            }
        }
        (transient_indices, absorbing_indices)
    }

    /// Returns the transition matrix.
    #[verified_engine::verified]
    pub fn transition_matrix(&self) -> &DMatrix<T> {
        &self.transition_matrix
    }

    /// Returns the state types.
    #[verified_engine::verified]
    pub fn state_types(&self) -> &[StateType] {
        &self.state_types
    }

    /// Returns the number of states.
    #[verified_engine::verified]
    pub fn num_states(&self) -> usize {
        self.state_types.len()
    }

    /// Returns the number of transient states.
    #[verified_engine::verified]
    pub fn num_transient(&self) -> usize {
        self.transient_indices.len()
    }

    /// Returns the number of absorbing states.
    #[verified_engine::verified]
    pub fn num_absorbing(&self) -> usize {
        self.absorbing_indices.len()
    }

    /// Extracts the Q submatrix (transient → transient transitions).
    #[verified_engine::verified]
    pub fn q_matrix(&self) -> DMatrix<T> {
        let n_t = self.transient_indices.len();
        let mut q = DMatrix::zeros(n_t, n_t);
        for (i, &idx_i) in self.transient_indices.iter().enumerate() {
            for (j, &idx_j) in self.transient_indices.iter().enumerate() {
                q[(i, j)] = self.transition_matrix[(idx_i, idx_j)];
            }
        }
        q
    }

    /// Extracts the R submatrix (transient → absorbing transitions).
    #[verified_engine::verified]
    pub fn r_matrix(&self) -> DMatrix<T> {
        let n_t = self.transient_indices.len();
        let n_a = self.absorbing_indices.len();
        let mut r = DMatrix::zeros(n_t, n_a);
        for (i, &idx_i) in self.transient_indices.iter().enumerate() {
            for (j, &idx_j) in self.absorbing_indices.iter().enumerate() {
                r[(i, j)] = self.transition_matrix[(idx_i, idx_j)];
            }
        }
        r
    }

    /// Computes the fundamental matrix $N = (I - Q)^{-1}$.
    ///
    /// # Errors
    ///
    /// Returns `SingularMatrix` if $(I - Q)$ is not invertible.
    #[verified_engine::verified]
    pub fn fundamental_matrix(&self) -> Result<DMatrix<T>> {
        let q = self.q_matrix();
        let n_t = q.nrows();
        let i_minus_q = DMatrix::identity(n_t, n_t) - q;

        i_minus_q
            .try_inverse()
            .ok_or_else(|| MarkovError::SingularMatrix {
                context: "Cannot invert (I - Q) for fundamental matrix".to_string(),
            })
    }

    /// Computes absorption probabilities $B = N \cdot R$.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no absorbing states or the fundamental matrix is singular.
    #[verified_engine::verified]
    pub fn absorption_probabilities(&self) -> Result<DMatrix<T>> {
        if self.absorbing_indices.is_empty() {
            return Err(MarkovError::InvalidState {
                reason: "No absorbing states in chain".to_string(),
            });
        }
        let n = self.fundamental_matrix()?;
        let r = self.r_matrix();
        Ok(n * r)
    }

    /// Computes expected absorption times $t_i = \sum_j N_{ij}$.
    ///
    /// # Errors
    ///
    /// Returns an error if the fundamental matrix cannot be computed.
    #[verified_engine::verified]
    pub fn expected_absorption_times(&self) -> Result<DVector<T>> {
        let n = self.fundamental_matrix()?;
        let n_t = n.nrows();
        let mut times = DVector::zeros(n_t);
        for i in 0..n_t {
            times[i] = n.row(i).iter().fold(T::zero(), |acc, &x| acc + x);
        }
        Ok(times)
    }

    /// Computes the n-step transition matrix $P^n$ via repeated squaring.
    #[verified_engine::verified]
    pub fn n_step_transition(&self, n: usize) -> DMatrix<T> {
        if n == 0 {
            return DMatrix::identity(self.num_states(), self.num_states());
        }
        if n == 1 {
            return self.transition_matrix.clone();
        }

        let mut result = DMatrix::identity(self.num_states(), self.num_states());
        let mut base = self.transition_matrix.clone();
        let mut power = n;

        while power > 0 {
            if power % 2 == 1 {
                result *= &base;
            }
            base = &base * &base;
            power /= 2;
        }
        result
    }

    /// Computes the Expected Possession Value (EPV) $N \cdot r$ for each transient state.
    ///
    /// # Errors
    ///
    /// Returns an error if `rewards.len() != self.num_absorbing()` or $N$ is singular.
    #[verified_engine::verified]
    pub fn expected_possession_value(&self, rewards: &DVector<T>) -> Result<DVector<T>> {
        if rewards.len() != self.num_absorbing() {
            return Err(MarkovError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(self.num_absorbing()),
                    actual: math_commons::math_kernel::types::Dimension(rewards.len()),
                },
            ));
        }

        let b = self.absorption_probabilities()?;
        Ok(b * rewards)
    }
}

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
