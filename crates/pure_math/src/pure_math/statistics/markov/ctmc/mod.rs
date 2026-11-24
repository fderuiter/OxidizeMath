//! Continuous-Time Markov Chains (CTMC).
//!
//! This module implements continuous-time Markov chains, where transitions
//! can occur at any continuous time point rather than discrete steps.

use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

mod serde_impl;

/// A continuous-time Markov chain characterized by its generator matrix.
///
/// # Mathematical Background
///
/// A continuous-time Markov chain (CTMC) is characterized by a generator matrix G
/// (also called rate matrix or infinitesimal generator) where:
/// - G\[i,j\] for i ≠ j: the rate of transition from state i to state j
/// - G\[i,i\] = -Σⱼ≠ᵢ G\[i,j\]: chosen so each row sums to 0
///
/// The transition probability matrix over time t is:
/// P(t) = exp(Gt) = Σₖ₌₀^∞ (Gt)^k / k!
///
/// The steady-state distribution π satisfies:
/// π·G = 0 and Σᵢ πᵢ = 1
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::markov::ctmc::ContinuousMarkovChain;
/// use math_explorer::nalgebra::DMatrix;
///
/// // Two-state birth-death process
/// // State 0 → State 1 at rate 2.0
/// // State 1 → State 0 at rate 3.0
/// let generator = DMatrix::from_row_slice(2, 2, &[
///     -2.0,  2.0,
///      3.0, -3.0,
/// ]);
///
/// let chain = ContinuousMarkovChain::<f64>::new(generator).unwrap();
///
/// // Compute transition probabilities at t=1.0
/// let p_t = chain.transition_probabilities(1.0).unwrap();
/// println!("P(1.0) = {:?}", p_t);
///
/// // Compute steady-state distribution
/// if let Some(pi) = chain.steady_state() {
///     println!("Steady state: {:?}", pi);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ContinuousMarkovChain<T: RealField + Copy + ToPrimitive> {
    /// The generator matrix G.
    generator: DMatrix<T>,
    /// Number of states.
    num_states: usize,
}

impl<T: RealField + Copy + ToPrimitive> ContinuousMarkovChain<T> {
    /// Creates a new continuous-time Markov chain.
    ///
    /// # Arguments
    ///
    /// * `generator` - The generator matrix G
    ///
    /// # Returns
    ///
    /// A new `ContinuousMarkovChain` or an error if validation fails.
    ///
    /// # Errors
    ///
    /// - `InvalidGenerator`: If the matrix is not a valid generator
    #[verified_engine::verified]
    pub fn new(generator: DMatrix<T>) -> Result<Self> {
        let n = generator.nrows();
        if n == 0 {
            return Err(MarkovError::InvalidGenerator {
                reason: "Generator matrix must have at least one state".to_string(),
            });
        }

        if generator.ncols() != n {
            return Err(crate::error::MarkovError::Math(
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

    /// Computes the transition probability matrix P(t) = exp(Gt).
    ///
    /// # Arguments
    ///
    /// * `t` - Time value (must be non-negative)
    ///
    /// # Returns
    ///
    /// The transition probability matrix at time t.
    ///
    /// # Errors
    ///
    /// - `InvalidState`: If t is negative or not finite
    /// - `NumericalError`: If matrix exponential computation fails
    ///
    /// # Implementation
    ///
    /// Uses the scaling and squaring method with Padé approximation.
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

        // Compute exp(Gt) using Padé approximation with scaling and squaring
        let gt = &self.generator * t;
        self.matrix_exponential(&gt)
    }

    /// Computes the matrix exponential using Padé approximation with scaling and squaring.
    ///
    /// This is a simplified implementation. For production use, consider using
    /// a specialized library like `nalgebra::linalg::Exp`.
    #[verified_engine::verified]
    fn matrix_exponential(&self, a: &DMatrix<T>) -> Result<DMatrix<T>> {
        let n = a.nrows();

        // Scaling: choose s such that ||A/2^s|| < 1
        let norm = self.matrix_norm_1(a);
        let s = if norm > 1.0 {
            (math_commons::math_kernel::soft_float::log2(norm).ceil() as i32).max(0)
        } else {
            0
        };

        let scale = T::from_f64(2.0).unwrap().powi(s);
        let a_scaled = a / scale;

        // Padé approximation of order 6
        let a2 = &a_scaled * &a_scaled;
        let a4 = &a2 * &a2;
        let a6 = &a2 * &a4;

        // Coefficients for Padé(6,6)
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

        // Solve (V - U)R = V + U for R
        let lhs = v.clone() - &u;
        let rhs = v + u;

        let r = lhs.try_inverse().ok_or_else(|| {
            crate::error::MarkovError::Math(math_commons::error::MathError::NumericalError {
                reason: "Failed to invert (V-U) in Padé approximation".to_string(),
            })
        })? * rhs;

        // Squaring: compute R^(2^s)
        let mut result = r;
        for _ in 0..s {
            result = &result * &result;
        }

        Ok(result)
    }

    /// Computes the 1-norm of a matrix (maximum absolute column sum).
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

    /// Computes the unique steady-state distribution $\pi$ satisfying $\pi^T G = 0$ and
    /// $\sum_i \pi_i = 1$ via Tarjan SCC sink isolation and scale-normalized Householder QR.
    ///
    /// Returns `None` when multiple closed sink classes exist (non-unique steady state).
    #[verified_engine::verified]
    pub fn steady_state(&self) -> Option<DVector<T>> {
        assert!(self.num_states > 0);
        let mut ws = math_commons::stochastic::SolverWorkspace::new(self.num_states, 0);
        let mut out = DVector::zeros(self.num_states);
        self.steady_state_with_workspace(&mut ws, &mut out)
            .then_some(out)
    }

    /// Computes the unique steady-state distribution into `out` using preallocated
    /// [`math_commons::stochastic::SolverWorkspace`] buffers with zero heap allocations when
    /// `ws.state_capacity() >= self.num_states`.
    #[verified_engine::verified]
    pub fn steady_state_with_workspace(
        &self,
        ws: &mut math_commons::stochastic::SolverWorkspace,
        out: &mut DVector<T>,
    ) -> bool {
        let n = self.num_states;
        debug_assert_eq!(out.len(), n);
        if n == 0 || out.len() != n {
            return false;
        }
        {
            let staged = ws.operator_staging_mut(n);
            for r in 0..n {
                for c in 0..n {
                    let Some(v) = self.generator[(r, c)].to_f64() else {
                        return false;
                    };
                    staged[math_commons::math_kernel::types::flatten_2d_index(c, r, n)] = v;
                }
            }
        }
        let Some(pi_f64) = ws.solve_staged_ctmc_steady_state_slice(n) else {
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

    /// Computes the expected time to absorption from a transient state.
    ///
    /// For CTMCs with absorbing states, this computes the mean time until
    /// absorption from each transient state.
    ///
    /// # Arguments
    ///
    /// * `transient_states` - Indices of transient states
    ///
    /// # Returns
    ///
    /// A vector of expected absorption times for each transient state.
    ///
    /// # Mathematical Background
    ///
    /// The expected time satisfies: Q·t = -1, where Q is the transient
    /// submatrix of the generator.
    #[verified_engine::verified]
    pub fn expected_absorption_times(&self, transient_states: &[usize]) -> Result<DVector<T>> {
        let n_t = transient_states.len();
        if n_t == 0 {
            return Err(MarkovError::InvalidState {
                reason: "No transient states specified".to_string(),
            });
        }

        // Extract Q submatrix
        let mut q = DMatrix::zeros(n_t, n_t);
        for (i, &idx_i) in transient_states.iter().enumerate() {
            for (j, &idx_j) in transient_states.iter().enumerate() {
                q[(i, j)] = self.generator[(idx_i, idx_j)];
            }
        }

        // Solve Q·t = -1
        let ones = DVector::from_element(n_t, -T::one());

        q.try_inverse()
            .ok_or_else(|| MarkovError::SingularMatrix {
                context: "Cannot invert Q matrix for absorption times".to_string(),
            })
            .map(|q_inv| q_inv * ones)
    }

    /// Generates a sample trajectory from the continuous-time Markov chain.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state
    /// * `max_time` - Maximum simulation time
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// A vector of (time, state) pairs representing the trajectory.
    ///
    /// # Algorithm
    ///
    /// Uses the Gillespie algorithm:
    /// 1. In state i, the holding time is Exponential(-G\[i,i\])
    /// 2. Next state is chosen with probabilities G\[i,j\] / (-G\[i,i\])
    ///
    /// # Panics
    ///
    /// This function will panic if `T::from_f64` fails when casting time values
    /// or small numerical tolerances into the generic real field `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::markov::ContinuousMarkovChain;
    /// use math_explorer::nalgebra::DMatrix;
    /// use math_explorer::rng::OxidizeRng;
    ///
    /// // A simple 2-state Birth-Death process
    /// let generator = DMatrix::from_row_slice(2, 2, &[
    ///     -1.0,  1.0,
    ///      2.0, -2.0
    /// ]);
    /// let chain = ContinuousMarkovChain::new(generator).unwrap();
    /// let mut rng = OxidizeRng::new(42);
    ///
    /// let trajectory = chain.simulate_trajectory(0, 10.0, &mut rng).unwrap();
    /// assert!(!trajectory.is_empty());
    /// assert_eq!(trajectory[0], (0.0, 0));
    /// ```
    #[verified_engine::verified]
    pub fn simulate_trajectory<R: rand::Rng>(
        &self,
        initial_state: usize,
        max_time: T,
        rng: &mut R,
    ) -> Result<Vec<(T, usize)>> {
        use crate::pure_math::analysis::stochastic::GillespieSolver;

        debug_assert!(self.num_states > 0);
        if initial_state >= self.num_states {
            return Err(MarkovError::InvalidState {
                reason: format!("Initial state {} out of bounds", initial_state),
            });
        }

        let mut trajectory = vec![(T::zero(), initial_state)];
        let mut current_state = initial_state;
        let mut current_time = T::zero();
        let mut solver = GillespieSolver::with_capacity(rng, self.num_states);

        while current_time < max_time {
            let holding_time = solver.step(self, &mut current_state).map_err(|err| {
                MarkovError::Math(math_commons::error::MathError::NumericalError {
                    reason: err.to_string(),
                })
            })?;
            if !holding_time.is_finite() {
                break;
            }
            let Some(dt_val) = T::from_f64(holding_time) else {
                return Err(MarkovError::Math(
                    math_commons::error::MathError::ConversionError {
                        reason: format!("Failed to convert holding time {holding_time}"),
                    },
                ));
            };
            current_time += dt_val;
            if current_time >= max_time {
                break;
            }
            trajectory.push((current_time, current_state));
        }

        Ok(trajectory)
    }
}

impl<T: RealField + Copy + ToPrimitive>
    crate::pure_math::analysis::stochastic::StochasticSystem<usize> for ContinuousMarkovChain<T>
{
    #[verified_engine::verified]
    fn propensities(&self, state: &usize, out: &mut Vec<f64>) {
        debug_assert!(*state < self.num_states);
        let i = *state;
        let tol = math_commons::registry::TOLERANCE_HIGH;
        let total_exit = (-self.generator[(i, i)]).to_f64().unwrap_or(0.0);
        if total_exit <= 0.0 {
            return;
        }
        let min_edge = (total_exit * 1e-15).min(tol);
        for j in 0..self.num_states {
            if j == i {
                out.push(0.0);
            } else {
                let rate = self.generator[(i, j)].to_f64().unwrap_or(0.0);
                out.push(if rate > min_edge { rate } else { 0.0 });
            }
        }
    }

    #[verified_engine::verified]
    fn react(
        &self,
        state: &mut usize,
        reaction_index: usize,
    ) -> std::result::Result<(), crate::pure_math::analysis::stochastic::StochasticError> {
        debug_assert!(*state < self.num_states);
        if reaction_index >= self.num_states {
            return Err(
                crate::pure_math::analysis::stochastic::StochasticError::InvalidReactionIndex(
                    reaction_index,
                ),
            );
        }
        *state = reaction_index;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
