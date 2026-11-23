//! Continuous-Time Markov Chains (CTMC).
//!
//! This module implements continuous-time Markov chains, where transitions
//! can occur at any continuous time point rather than discrete steps:
//! - Generator validation and matrix exponential $P(t) = \exp(Gt)$
//! - Iterative Tarjan communicating classes and rank-checked QR steady-state (`steady_state`)
//! - Expected absorption times
//! - Gillespie trajectory simulation including zero-allocation buffers (`simulation`)

use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

/// Gillespie trajectory simulation (allocating and zero-allocation variants).
pub mod simulation;
/// Communicating classes and rank-checked QR linear steady-state solver.
pub mod steady_state;

/// A continuous-time Markov chain characterized by its generator matrix.
///
/// # Mathematical Background
///
/// A continuous-time Markov chain (CTMC) is characterized by a generator matrix G
/// (also called rate matrix or infinitesimal generator) where:
/// - `G[i,j]` for $i \ne j$: the rate of transition from state $i$ to state $j$
/// - `G[i,i]` = $-\sum_{j \ne i} G_{i,j}$: chosen so each row sums to $0$
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::markov::ctmc::ContinuousMarkovChain;
/// use math_explorer::nalgebra::DMatrix;
///
/// let generator = DMatrix::from_row_slice(2, 2, &[
///     -2.0,  2.0,
///      3.0, -3.0,
/// ]);
/// let chain = ContinuousMarkovChain::<f64>::new(generator).unwrap();
/// let pi = chain.steady_state().unwrap();
/// assert!((pi[0] - 0.6).abs() < 1e-12);
/// assert!((pi[1] - 0.4).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct ContinuousMarkovChain<T: RealField + Copy + ToPrimitive> {
    /// The generator matrix G.
    pub(crate) generator: DMatrix<T>,
    /// Number of states.
    pub(crate) num_states: usize,
}

impl<T: RealField + Copy + ToPrimitive> ContinuousMarkovChain<T> {
    /// Creates a new continuous-time Markov chain.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If generator is not square
    /// - `InvalidGenerator`: If off-diagonal entries are negative or rows do not sum to 0
    #[verified_engine::verified]
    pub fn new(generator: DMatrix<T>) -> Result<Self> {
        let n = generator.nrows();
        if generator.ncols() != n {
            return Err(MarkovError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(n),
                    actual: math_commons::math_kernel::types::Dimension(generator.ncols()),
                },
            ));
        }

        crate::pure_math::statistics::markov::validation::validate_generator_matrix(&generator)?;

        Ok(ContinuousMarkovChain {
            generator,
            num_states: n,
        })
    }

    /// Returns the generator matrix.
    #[verified_engine::verified]
    pub fn generator(&self) -> &DMatrix<T> {
        &self.generator
    }

    /// Returns the number of states.
    #[verified_engine::verified]
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Computes the transition probability matrix $P(t) = \exp(Gt)$.
    ///
    /// # Errors
    ///
    /// - `InvalidState`: If $t$ is negative or non-finite
    /// - `NumericalError`: If Padé matrix exponential inversion fails
    #[verified_engine::verified]
    pub fn transition_probabilities(&self, t: T) -> Result<DMatrix<T>> {
        if !t.is_finite() || t < T::zero() {
            return Err(MarkovError::InvalidState {
                reason: format!(
                    "Time t must be non-negative and finite, got {}",
                    t.to_f64().unwrap_or(f64::NAN)
                ),
            });
        }

        if t == T::zero() {
            return Ok(DMatrix::identity(self.num_states, self.num_states));
        }

        let gt = &self.generator * t;
        self.matrix_exponential(&gt)
    }

    /// Computes the matrix exponential using Padé(6,6) approximation with scaling and squaring.
    #[verified_engine::verified]
    fn matrix_exponential(&self, a: &DMatrix<T>) -> Result<DMatrix<T>> {
        let n = a.nrows();
        let norm = self.matrix_norm_1(a);
        let s = if norm > 1.0 {
            (norm.log2().ceil() as i32).max(0)
        } else {
            0
        };

        let scale = T::from_f64(2.0).unwrap().powi(s);
        let a_scaled = a / scale;

        let a2 = &a_scaled * &a_scaled;
        let a4 = &a2 * &a2;
        let a6 = &a2 * &a4;

        let c0 = T::one();
        let c1 = T::from_f64(0.5).unwrap();
        let c2 = T::from_f64(1.0 / 9.0).unwrap();
        let c3 = T::from_f64(1.0 / 72.0).unwrap();
        let c4 = T::from_f64(1.0 / 1008.0).unwrap();
        let c5 = T::from_f64(1.0 / 30240.0).unwrap();
        let c6 = T::from_f64(1.0 / 1814400.0).unwrap();

        let id = DMatrix::identity(n, n);
        let u = &a_scaled * (&a6 * c6 + &a4 * c4 + &a2 * c2 + &id * c0);
        let v = &a6 * c5 + &a4 * c3 + &a2 * c1 + &id * c0;

        let lhs = v.clone() - &u;
        let rhs = v + u;

        let r = lhs.try_inverse().ok_or_else(|| {
            MarkovError::Math(math_commons::error::MathError::NumericalError {
                reason: "Failed to invert (V-U) in Padé approximation".to_string(),
            })
        })? * rhs;

        let mut result = r;
        for _ in 0..s {
            result = &result * &result;
        }
        Ok(result)
    }

    #[verified_engine::verified]
    fn matrix_norm_1(&self, a: &DMatrix<T>) -> f64 {
        let mut max_sum: f64 = 0.0;
        for j in 0..a.ncols() {
            let col_sum: f64 = a
                .column(j)
                .iter()
                .fold(0.0, |acc, &x| acc + x.to_f64().unwrap_or(0.0).abs());
            max_sum = max_sum.max(col_sum);
        }
        max_sum
    }

    /// Computes the expected time to absorption from each transient state ($Q \cdot t = -\mathbf{1}$).
    ///
    /// # Errors
    ///
    /// Returns an error if `transient_states` is empty or $Q$ is singular.
    #[verified_engine::verified]
    pub fn expected_absorption_times(&self, transient_states: &[usize]) -> Result<DVector<T>> {
        let n_t = transient_states.len();
        if n_t == 0 {
            return Err(MarkovError::InvalidState {
                reason: "No transient states specified".to_string(),
            });
        }

        let mut q = DMatrix::zeros(n_t, n_t);
        for (i, &idx_i) in transient_states.iter().enumerate() {
            for (j, &idx_j) in transient_states.iter().enumerate() {
                q[(i, j)] = self.generator[(idx_i, idx_j)];
            }
        }

        let neg_ones = DVector::from_element(n_t, -T::one());
        q.try_inverse()
            .ok_or_else(|| MarkovError::SingularMatrix {
                context: "Cannot invert Q matrix for absorption times".to_string(),
            })
            .map(|q_inv| q_inv * neg_ones)
    }
}

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
