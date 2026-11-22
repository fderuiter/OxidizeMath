//! Discrete-Time Markov Chains (DTMC).
//!
//! This module implements discrete-time Markov chains with support for:
//! - Transient and absorbing states
//! - Canonical form decomposition
//! - Fundamental matrix computation
//! - Expected Possession Value (EPV) calculation
//! - Stationary distribution computation
//! - Absorption probabilities

use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

mod types;
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
/// where:
/// - Q: transient → transient transitions
/// - R: transient → absorbing transitions
/// - 0: absorbing → transient (impossible)
/// - I: absorbing → absorbing (identity)
///
/// The fundamental matrix is N = (I - Q)⁻¹, where `N[i,j]` is the expected number
/// of times the chain visits transient state j starting from transient state i.
///
/// # Example
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
/// use math_explorer::nalgebra::DMatrix;
///
/// // Simple 3-state chain: two transient states, one absorbing
/// let transition_matrix = DMatrix::from_row_slice(3, 3, &[
///     0.7, 0.2, 0.1,  // State 0 (transient)
///     0.3, 0.5, 0.2,  // State 1 (transient)
///     0.0, 0.0, 1.0,  // State 2 (absorbing)
/// ]);
///
/// let state_types = vec![
///     StateType::Transient,
///     StateType::Transient,
///     StateType::Absorbing,
/// ];
///
/// let chain = MarkovChain::<f64>::new(transition_matrix, state_types).unwrap();
///
/// // Compute expected number of visits
/// let fundamental = chain.fundamental_matrix().unwrap();
/// println!("Expected visits: {:?}", fundamental);
///
/// // Compute absorption probabilities
/// let absorption_probs = chain.absorption_probabilities().unwrap();
/// println!("Absorption probabilities: {:?}", absorption_probs);
/// ```
#[derive(Debug, Clone)]
pub struct MarkovChain<T: RealField + Copy + ToPrimitive> {
    /// The transition matrix P.
    transition_matrix: DMatrix<T>,
    /// Type of each state (transient or absorbing).
    state_types: Vec<StateType>,
    /// Indices of transient states.
    transient_indices: Vec<usize>,
    /// Indices of absorbing states.
    absorbing_indices: Vec<usize>,
}

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
    /// Creates a new Markov chain.
    ///
    /// # Arguments
    ///
    /// * `transition_matrix` - The transition matrix P where `P[i,j]` = P(next=j | current=i)
    /// * `state_types` - Classification of each state
    ///
    /// # Returns
    ///
    /// A new `MarkovChain` or an error if validation fails.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If matrix size doesn't match state_types length
    /// - `NotStochastic`: If matrix rows don't sum to 1
    ///
    /// # Panics
    ///
    /// Panics if the generic real field `T` fails to instantiate from the `f64` value `math_commons::registry::TOLERANCE_HIGH`.
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    #[verified_engine::verified]
    pub fn new(transition_matrix: DMatrix<T>, state_types: Vec<StateType>) -> Result<Self> {
        let n = transition_matrix.nrows();

        // Validate dimensions
        if transition_matrix.ncols() != n {
            return Err(crate::error::MarkovError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(n),
                    actual: math_commons::math_kernel::types::Dimension(transition_matrix.ncols()),
                },
            ));
        }

        if state_types.len() != n {
            return Err(crate::error::MarkovError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(n),
                    actual: math_commons::math_kernel::types::Dimension(state_types.len()),
                },
            ));
        }

        // Validate stochasticity
        crate::pure_math::statistics::markov::validation::validate_stochastic_matrix(
            &transition_matrix,
        )?;

        // Validate absorbing states
        for (i, state_type) in state_types.iter().enumerate() {
            if *state_type == StateType::Absorbing {
                // Check that P[i,i] = 1 and all other entries in row i are 0
                for j in 0..n {
                    let expected = if i == j { T::one() } else { T::zero() };
                    let actual = transition_matrix[(i, j)];
                    let tolerance = T::from_f64(math_commons::registry::TOLERANCE_HIGH).unwrap();
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

        // Separate transient and absorbing indices
        let transient_indices: Vec<usize> = state_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                if *t == StateType::Transient {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        let absorbing_indices: Vec<usize> = state_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                if *t == StateType::Absorbing {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        Ok(MarkovChain {
            transition_matrix,
            state_types,
            transient_indices,
            absorbing_indices,
        })
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
    ///
    /// # Returns
    ///
    /// The Q submatrix of size (n_transient × n_transient).
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
    ///
    /// # Returns
    ///
    /// The R submatrix of size (n_transient × n_absorbing).
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

    /// Computes the fundamental matrix N = (I - Q)⁻¹.
    ///
    /// # Mathematical Background
    ///
    /// The fundamental matrix N has the property that `N[i,j]` is the expected
    /// number of times the chain visits transient state j before absorption,
    /// given that it starts in transient state i.
    ///
    /// # Returns
    ///
    /// The fundamental matrix N of size (n_transient × n_transient).
    ///
    /// # Errors
    ///
    /// Returns `SingularMatrix` if (I - Q) is not invertible.
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

    /// Computes absorption probabilities B = N·R.
    ///
    /// # Mathematical Background
    ///
    /// `B[i,j]` is the probability that the chain, starting from transient state i,
    /// will eventually be absorbed into absorbing state j.
    ///
    /// # Returns
    ///
    /// The absorption probability matrix B of size (n_transient × n_absorbing).
    ///
    /// # Errors
    ///
    /// Returns an error if the fundamental matrix cannot be computed.
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

    /// Computes expected absorption times (expected number of steps before absorption).
    ///
    /// # Mathematical Background
    ///
    /// The expected absorption time from transient state i is `t[i]` = Σⱼ `N[i,j]`.
    ///
    /// # Returns
    ///
    /// A vector of expected absorption times, one for each transient state.
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

    /// Computes the n-step transition matrix P^n.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of steps
    ///
    /// # Returns
    ///
    /// The n-step transition matrix P^n.
    #[verified_engine::verified]
    pub fn n_step_transition(&self, n: usize) -> DMatrix<T> {
        if n == 0 {
            return DMatrix::identity(self.num_states(), self.num_states());
        }
        if n == 1 {
            return self.transition_matrix.clone();
        }

        // Use repeated squaring for efficiency
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

    /// Computes the stationary distribution (if it exists).
    ///
    /// # Mathematical Background
    ///
    /// A stationary distribution π satisfies π·P = π and Σᵢ πᵢ = 1.
    /// For irreducible, aperiodic chains, the stationary distribution exists
    /// and is unique. It can be found as the limit of P^n as n → ∞.
    ///
    /// # Returns
    ///
    /// The stationary distribution as a probability vector, or None if it
    /// doesn't exist or convergence fails.
    ///
    /// # Panics
    ///
    /// Panics if the generic real field `T` fails to instantiate from a `usize` (number of absorbing states or number of total states) or an `f64` (`1e-12`).
    ///
    /// # Examples
    /// ```rust
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
    /// use math_explorer::nalgebra::DMatrix;
    ///
    /// // A simple 2-state ergodic chain
    /// let transition_matrix = DMatrix::from_row_slice(2, 2, &[
    ///     0.8, 0.2,
    ///     0.4, 0.6,
    /// ]);
    /// let state_types = vec![StateType::Transient, StateType::Transient];
    /// let chain = MarkovChain::<f64>::new(transition_matrix, state_types).unwrap();
    ///
    /// let pi = chain.stationary_distribution().unwrap();
    /// // The stationary distribution solves πP = π.
    /// // Solving the system: 0.8*π0 + 0.4*π1 = π0 => 0.4*π1 = 0.2*π0 => π0 = 2*π1.
    /// // Since π0 + π1 = 1, we get π0 ≈ 0.666... and π1 ≈ 0.333...
    /// assert!((pi[0] - 2.0 / 3.0).abs() < math_explorer::registry::TOLERANCE_HIGH);
    /// assert!((pi[1] - 1.0 / 3.0).abs() < math_explorer::registry::TOLERANCE_HIGH);
    /// ```
    #[verified_engine::verified]
    pub fn stationary_distribution(&self) -> Option<DVector<T>> {
        // For chains with absorbing states, only absorbing states have non-zero
        // stationary probability
        if !self.absorbing_indices.is_empty() {
            let mut pi = DVector::zeros(self.num_states());
            let n_absorbing = self.absorbing_indices.len();

            // Each absorbing state gets equal probability
            for &idx in &self.absorbing_indices {
                pi[idx] = T::one() / T::from_usize(n_absorbing).unwrap();
            }

            return Some(pi);
        }

        // For ergodic chains, use power method
        let max_iters: usize = math_commons::registry::MAX_ITERATIONS;
        let tolerance = T::from_f64(1e-12).unwrap();

        let mut pi = DVector::from_element(
            self.num_states(),
            T::one() / T::from_usize(self.num_states()).unwrap(),
        );

        // OPTIMIZATION: Cache transpose outside the loop to avoid O(N^2) allocations per iteration
        let transition_matrix_t = self.transition_matrix.transpose();

        for _ in 0..max_iters {
            let pi_next = &transition_matrix_t * &pi;

            // Check convergence
            let diff = (&pi_next - &pi).norm();
            if diff < tolerance {
                return Some(pi_next);
            }

            pi = pi_next;
        }

        None
    }

    /// Computes the Expected Possession Value (EPV) for each transient state.
    ///
    /// # Mathematical Background
    ///
    /// EPV is computed as:
    /// EPV = N · r
    ///
    /// where:
    /// - N is the fundamental matrix
    /// - r is a reward vector for absorbing states
    ///
    /// # Arguments
    ///
    /// * `rewards` - Reward for each absorbing state
    ///
    /// # Returns
    ///
    /// A vector of EPV values, one for each transient state.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The fundamental matrix cannot be computed
    /// - The reward vector size doesn't match the number of absorbing states
    #[verified_engine::verified]
    pub fn expected_possession_value(&self, rewards: &DVector<T>) -> Result<DVector<T>> {
        if rewards.len() != self.num_absorbing() {
            return Err(crate::error::MarkovError::Math(
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
