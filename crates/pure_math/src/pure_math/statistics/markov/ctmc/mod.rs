//! Continuous-Time Markov Chains (CTMC).
//!
//! This module implements continuous-time Markov chains, where transitions
//! can occur at any continuous time point rather than discrete steps.

use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;

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
            (norm.log2().ceil() as i32).max(0)
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

    /// Computes the steady-state distribution.
    ///
    /// # Mathematical Background
    ///
    /// The steady-state distribution π satisfies:
    /// π·G = 0 and Σᵢ πᵢ = 1
    ///
    /// # Returns
    ///
    /// The steady-state distribution if it exists and converges, or None otherwise.
    ///
    /// # Panics
    ///
    /// This function will panic if `T::from_f64` fails when attempting to cast
    /// numeric constants (e.g., tolerances or exponents) to the generic type `T`.
    #[verified_engine::verified]
    pub fn steady_state(&self) -> Option<DVector<T>> {
        // For irreducible CTMCs, we can find the steady state by computing
        // the null space of G^T and normalizing.

        // Alternative approach: simulate for a long time
        // π ≈ e^T·P(t) for large t, where e^T is any initial distribution

        let long_time = T::from_f64(100.0).unwrap();
        const MAX_ATTEMPTS: usize = 5;

        for attempt in 0..MAX_ATTEMPTS {
            let t = long_time * T::from_usize(1 + attempt).unwrap();

            if let Ok(p_t) = self.transition_probabilities(t) {
                // Use uniform initial distribution
                let mut pi = DVector::from_element(
                    self.num_states,
                    T::one() / T::from_usize(self.num_states).unwrap(),
                );
                pi = p_t.transpose() * pi;

                // Check if it's approximately stationary
                let pi_next = self.generator.transpose() * &pi;
                if pi_next.norm() < T::from_f64(math_commons::registry::TOLERANCE_FAST).unwrap() {
                    // Normalize to ensure exact sum to 1
                    let sum = pi.iter().fold(T::zero(), |acc, &x| acc + x);
                    if sum > T::from_f64(math_commons::registry::TOLERANCE_HIGH).unwrap() {
                        pi /= sum;
                        return Some(pi);
                    }
                }
            }
        }

        None
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
    /// use math_explorer::rand::SeedableRng;
    /// use math_explorer::rand::rngs::StdRng;
    ///
    /// // A simple 2-state Birth-Death process
    /// let generator = DMatrix::from_row_slice(2, 2, &[
    ///     -1.0,  1.0,
    ///      2.0, -2.0
    /// ]);
    /// let chain = ContinuousMarkovChain::new(generator).unwrap();
    /// let mut rng = math_explorer::rng::OxidizeRng::default();
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
        use rand_distr::{Distribution, Exp, WeightedIndex};

        if initial_state >= self.num_states {
            return Err(MarkovError::InvalidState {
                reason: format!("Initial state {} out of bounds", initial_state),
            });
        }

        let mut trajectory = vec![(T::zero(), initial_state)];
        let mut current_state = initial_state;
        let mut current_time = T::zero();

        while current_time < max_time {
            // Get rate out of current state
            let rate = -self.generator[(current_state, current_state)];

            if rate < T::from_f64(math_commons::registry::TOLERANCE_HIGH).unwrap() {
                // Absorbing state or very slow rate
                break;
            }

            // Sample holding time
            let exp_dist = Exp::new(rate.to_f64().unwrap_or(0.0)).map_err(|_| {
                crate::error::MarkovError::Math(math_commons::error::MathError::NumericalError {
                    reason: format!(
                        "Invalid rate for exponential: {}",
                        rate.to_f64().unwrap_or(f64::NAN)
                    ),
                })
            })?;
            let holding_time: f64 = exp_dist.sample(rng);
            current_time += T::from_f64(holding_time).unwrap();

            if current_time >= max_time {
                break;
            }

            // Sample next state
            let mut weights = Vec::new();
            let mut next_states = Vec::new();
            for j in 0..self.num_states {
                if j != current_state {
                    let transition_rate = self.generator[(current_state, j)];
                    if transition_rate
                        > T::from_f64(math_commons::registry::TOLERANCE_HIGH).unwrap()
                    {
                        weights.push(transition_rate.to_f64().unwrap_or(0.0));
                        next_states.push(j);
                    }
                }
            }

            if weights.is_empty() {
                // No transitions possible (shouldn't happen if rate > 0)
                break;
            }

            let dist = WeightedIndex::new(&weights).map_err(|_| {
                crate::error::MarkovError::Math(math_commons::error::MathError::NumericalError {
                    reason: "Failed to create weighted distribution".to_string(),
                })
            })?;

            let next_idx = dist.sample(rng);
            current_state = next_states[next_idx];
            trajectory.push((current_time, current_state));
        }

        Ok(trajectory)
    }
}

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
