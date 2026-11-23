//! Time-indexed transition tensors for non-stationary Markov chains.
//!
//! This module supports Markov chains where transition probabilities vary with time,
//! such as modeling basketball possessions with shot clock urgency effects.

use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, RealField};

pub use super::time_index::TimeIndex;
use num_traits::ToPrimitive;

/// A time-indexed collection of transition matrices.
///
/// # Mathematical Background
///
/// A non-stationary Markov chain has transition probabilities that depend on time:
/// P(Xₙ₊₁ = j | Xₙ = i, n) = `Pₙ[i,j]`
///
/// This is useful for modeling:
/// - Shot clock urgency in basketball (transition rates change as time runs out)
/// - Game-time effects (strategy changes in final minutes)
/// - Seasonal effects in time series
///
/// # Example
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::markov::tensor::{TransitionTensor, TimeIndex};
///
/// // Create a tensor with transitions at different shot clock times
/// let mut tensor = TransitionTensor::<f64>::new(2, TimeIndex::new(0.0).unwrap(), TimeIndex::new(24.0).unwrap());
///
/// // Add transition matrix at t=24 (full shot clock - patient offense)
/// let p_24 = math_explorer::nalgebra::DMatrix::from_row_slice(2, 2, &[
///     0.9, 0.1,  // Mostly stay in possession
///     0.2, 0.8,
/// ]);
/// tensor.add_time_slice(TimeIndex::new(24.0).unwrap(), p_24).unwrap();
///
/// // Add transition matrix at t=5 (urgency - more shots)
/// let p_5 = math_explorer::nalgebra::DMatrix::from_row_slice(2, 2, &[
///     0.5, 0.5,  // More aggressive
///     0.3, 0.7,
/// ]);
/// tensor.add_time_slice(TimeIndex::new(5.0).unwrap(), p_5).unwrap();
///
/// // Query transition probability at t=15 (interpolated)
/// let p_15 = tensor.transition_matrix_at(TimeIndex::new(15.0).unwrap()).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionTensor<T: RealField + Copy + ToPrimitive> {
    /// Number of states.
    num_states: usize,
    /// Minimum time.
    min_time: TimeIndex<T>,
    /// Maximum time.
    max_time: TimeIndex<T>,
    /// Time-indexed transition matrices, sorted by time.
    time_slices: Vec<(TimeIndex<T>, DMatrix<T>)>,
}

impl<T: RealField + Copy + ToPrimitive> TransitionTensor<T> {
    /// Creates a new transition tensor.
    ///
    /// # Arguments
    ///
    /// * `num_states` - Number of states in the chain
    /// * `min_time` - Minimum time value
    /// * `max_time` - Maximum time value
    ///
    /// # Returns
    ///
    /// A new `TransitionTensor`.
    #[verified_engine::verified]
    pub fn new(num_states: usize, min_time: TimeIndex<T>, max_time: TimeIndex<T>) -> Self {
        TransitionTensor {
            num_states,
            min_time,
            max_time,
            time_slices: Vec::new(),
        }
    }

    /// Adds a transition matrix at a specific time.
    ///
    /// # Arguments
    ///
    /// * `time` - The time index
    /// * `matrix` - The transition matrix at this time
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error if validation fails.
    ///
    /// # Errors
    ///
    /// - `TimeIndexOutOfBounds`: If time is outside [min_time, max_time]
    /// - `DimensionMismatch`: If matrix size doesn't match num_states
    /// - `NotStochastic`: If matrix rows don't sum to 1
    ///
    /// # Panics
    ///
    /// Panics if `time.value()` or any slice's time value is NaN during binary search (`partial_cmp().unwrap()`).
    #[verified_engine::verified]
    pub fn add_time_slice(&mut self, time: TimeIndex<T>, matrix: DMatrix<T>) -> Result<()> {
        // Validate time bounds
        if time.value() < self.min_time.value() || time.value() > self.max_time.value() {
            return Err(MarkovError::TimeIndexOutOfBounds {
                time: time.value().to_f64().unwrap_or(f64::NAN),
                valid_range: (
                    self.min_time.value().to_f64().unwrap_or(f64::NAN),
                    self.max_time.value().to_f64().unwrap_or(f64::NAN),
                ),
            });
        }

        // Validate matrix dimensions
        if matrix.nrows() != self.num_states || matrix.ncols() != self.num_states {
            return Err(crate::error::MarkovError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(self.num_states),
                    actual: math_commons::math_kernel::types::Dimension(matrix.nrows()),
                },
            ));
        }

        // Validate stochasticity
        crate::pure_math::statistics::markov::validation::validate_stochastic_matrix(&matrix)?;

        // Insert in sorted order
        let insert_pos = self
            .time_slices
            .binary_search_by(|(t, _)| t.value().partial_cmp(&time.value()).unwrap())
            .unwrap_or_else(|pos| pos);

        self.time_slices.insert(insert_pos, (time, matrix));

        Ok(())
    }

    /// Gets the transition matrix at a specific time, using linear interpolation.
    ///
    /// # Arguments
    ///
    /// * `time` - The time at which to query the transition matrix
    ///
    /// # Returns
    ///
    /// The transition matrix at the given time, or an error if:
    /// - Time is out of bounds
    /// - No time slices have been added
    ///
    /// # Interpolation
    ///
    /// If the exact time is not present:
    /// - If time < first slice time: return first slice
    /// - If time > last slice time: return last slice
    /// - Otherwise: linear interpolation between adjacent slices
    ///
    /// # Errors
    ///
    /// - `MarkovError::InvalidState`: If no time slices have been added.
    /// - `MarkovError::TimeIndexOutOfBounds`: If `time` is outside [min_time, max_time].
    ///
    /// # Panics
    ///
    /// Panics if `time.value()` or any slice's time value is NaN during binary search (`partial_cmp().unwrap()`).
    #[verified_engine::verified]
    pub fn transition_matrix_at(&self, time: TimeIndex<T>) -> Result<DMatrix<T>> {
        if self.time_slices.is_empty() {
            return Err(MarkovError::InvalidState {
                reason: "No time slices added to tensor".to_string(),
            });
        }

        if time.value() < self.min_time.value() || time.value() > self.max_time.value() {
            return Err(MarkovError::TimeIndexOutOfBounds {
                time: time.value().to_f64().unwrap_or(f64::NAN),
                valid_range: (
                    self.min_time.value().to_f64().unwrap_or(f64::NAN),
                    self.max_time.value().to_f64().unwrap_or(f64::NAN),
                ),
            });
        }

        // Find adjacent time slices
        let pos = self
            .time_slices
            .binary_search_by(|(t, _)| t.value().partial_cmp(&time.value()).unwrap());

        match pos {
            Ok(idx) => {
                // Exact match
                Ok(self.time_slices[idx].1.clone())
            }
            Err(idx) => {
                if idx == 0 {
                    // Before first slice, return first
                    Ok(self.time_slices[0].1.clone())
                } else if idx >= self.time_slices.len() {
                    // After last slice, return last
                    Ok(self.time_slices[self.time_slices.len() - 1].1.clone())
                } else {
                    // Interpolate between idx-1 and idx
                    let (t1, p1) = &self.time_slices[idx - 1];
                    let (t2, p2) = &self.time_slices[idx];

                    let alpha = (time.value() - t1.value()) / (t2.value() - t1.value());
                    let interpolated = p1 * (T::one() - alpha) + p2 * alpha;

                    Ok(interpolated)
                }
            }
        }
    }

    /// Returns the number of time slices.
    #[verified_engine::verified]
    pub fn num_time_slices(&self) -> usize {
        self.time_slices.len()
    }

    /// Returns the number of states.
    #[verified_engine::verified]
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Returns the time range.
    #[verified_engine::verified]
    pub fn time_range(&self) -> (T, T) {
        (self.min_time.value(), self.max_time.value())
    }

    /// Computes the expected transition over a time interval.
    ///
    /// # Arguments
    ///
    /// * `start_time` - Start of the interval
    /// * `end_time` - End of the interval
    /// * `num_samples` - Number of time points to sample for integration
    ///
    /// # Returns
    ///
    /// An averaged transition matrix representing the expected transition
    /// over the time interval.
    ///
    /// # Errors
    ///
    /// - `MarkovError::InvalidState`: If `num_samples` is 0.
    /// - Errors propagated from `TransitionTensor::transition_matrix_at`.
    ///
    /// # Panics
    ///
    /// Panics if `T::from_usize()` fails and returns `None` for the number of samples or indices.
    #[verified_engine::verified]
    pub fn average_transition(
        &self,
        start_time: TimeIndex<T>,
        end_time: TimeIndex<T>,
        num_samples: usize,
    ) -> Result<DMatrix<T>> {
        if num_samples == 0 {
            return Err(MarkovError::InvalidState {
                reason: "num_samples must be positive".to_string(),
            });
        }

        let dt = (end_time.value() - start_time.value()) / T::from_usize(num_samples - 1).unwrap();
        let mut sum = DMatrix::zeros(self.num_states, self.num_states);

        for i in 0..num_samples {
            let t = start_time.value() + T::from_usize(i).unwrap() * dt;
            let p = self.transition_matrix_at(TimeIndex::new(t)?)?;
            sum += p;
        }

        Ok(sum / T::from_usize(num_samples).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_time_index() {
        let t = TimeIndex::new(10.0).unwrap();
        assert_eq!(t.value(), 10.0);

        // Invalid time
        assert!(TimeIndex::new(f64::NAN).is_err());
        assert!(TimeIndex::new(f64::INFINITY).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_tensor_creation() {
        let min_t = TimeIndex::new(0.0).unwrap();
        let max_t = TimeIndex::new(24.0).unwrap();
        let tensor = TransitionTensor::new(2, min_t, max_t);

        assert_eq!(tensor.num_states(), 2);
        assert_eq!(tensor.time_range(), (0.0, 24.0));
        assert_eq!(tensor.num_time_slices(), 0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_add_time_slice() {
        let min_t = TimeIndex::new(0.0).unwrap();
        let max_t = TimeIndex::new(24.0).unwrap();
        let mut tensor = TransitionTensor::new(2, min_t, max_t);

        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let t = TimeIndex::new(12.0).unwrap();

        tensor.add_time_slice(t, p).unwrap();
        assert_eq!(tensor.num_time_slices(), 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_exact_time_lookup() {
        let min_t = TimeIndex::new(0.0).unwrap();
        let max_t = TimeIndex::new(24.0).unwrap();
        let mut tensor = TransitionTensor::new(2, min_t, max_t);

        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let t = TimeIndex::new(12.0).unwrap();

        tensor.add_time_slice(t, p.clone()).unwrap();

        let retrieved = tensor.transition_matrix_at(t).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(retrieved[(i, j)], p[(i, j)]);
            }
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_interpolation() {
        let min_t = TimeIndex::new(0.0).unwrap();
        let max_t = TimeIndex::new(24.0).unwrap();
        let mut tensor = TransitionTensor::new(2, min_t, max_t);

        // Add two time slices
        let p1 = DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.3, 0.7]);
        let t1 = TimeIndex::new(0.0).unwrap();
        tensor.add_time_slice(t1, p1.clone()).unwrap();

        let p2 = DMatrix::from_row_slice(2, 2, &[0.6, 0.4, 0.5, 0.5]);
        let t2 = TimeIndex::new(24.0).unwrap();
        tensor.add_time_slice(t2, p2.clone()).unwrap();

        // Query at midpoint (should be average)
        let t_mid = TimeIndex::new(12.0).unwrap();
        let p_mid = tensor.transition_matrix_at(t_mid).unwrap();

        assert_relative_eq!(
            p_mid[(0, 0)],
            0.7,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        ); // (0.8 + 0.6) / 2
        assert_relative_eq!(
            p_mid[(0, 1)],
            0.3,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        ); // (0.2 + 0.4) / 2
        assert_relative_eq!(
            p_mid[(1, 0)],
            0.4,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        ); // (0.3 + 0.5) / 2
        assert_relative_eq!(
            p_mid[(1, 1)],
            0.6,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        ); // (0.7 + 0.5) / 2
    }

    #[test]
    #[verified_engine::verified]
    fn test_shot_clock_urgency() {
        // Model basketball possession: as shot clock decreases, urgency increases
        let min_t = TimeIndex::new(0.0).unwrap();
        let max_t = TimeIndex::new(24.0).unwrap();
        let mut tensor = TransitionTensor::new(3, min_t, max_t);

        // States: 0=offense, 1=shot attempt, 2=turnover (absorbing)

        // At t=24 (full clock): patient, low shot rate
        let p_24 = DMatrix::from_row_slice(
            3,
            3,
            &[
                0.85, 0.10, 0.05, // Offense: mostly stay, few shots, few turnovers
                0.0, 1.0, 0.0, // Shot (absorbing)
                0.0, 0.0, 1.0, // Turnover (absorbing)
            ],
        );
        tensor
            .add_time_slice(TimeIndex::new(24.0).unwrap(), p_24)
            .unwrap();

        // At t=5 (expiring): urgent, high shot rate
        let p_5 = DMatrix::from_row_slice(
            3,
            3,
            &[
                0.40, 0.50, 0.10, // Offense: fewer transitions, many shots, more turnovers
                0.0, 1.0, 0.0, // Shot (absorbing)
                0.0, 0.0, 1.0, // Turnover (absorbing)
            ],
        );
        tensor
            .add_time_slice(TimeIndex::new(5.0).unwrap(), p_5)
            .unwrap();

        // Check interpolation at t=15
        let p_15 = tensor
            .transition_matrix_at(TimeIndex::new(15.0).unwrap())
            .unwrap();

        // Shot rate should be between 0.10 and 0.50
        let shot_rate_15 = p_15[(0, 1)];
        assert!(shot_rate_15 > 0.10);
        assert!(shot_rate_15 < 0.50);

        // At t=20 (closer to full clock), should be closer to patient behavior
        let p_20 = tensor
            .transition_matrix_at(TimeIndex::new(20.0).unwrap())
            .unwrap();
        let shot_rate_20 = p_20[(0, 1)];
        assert!(shot_rate_20 < shot_rate_15); // Less urgent than at t=15
    }

    #[test]
    #[verified_engine::verified]
    fn test_average_transition() {
        let min_t = TimeIndex::new(0.0).unwrap();
        let max_t = TimeIndex::new(10.0).unwrap();
        let mut tensor = TransitionTensor::new(2, min_t, max_t);

        let p1 = DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.3, 0.7]);
        tensor
            .add_time_slice(TimeIndex::new(0.0).unwrap(), p1)
            .unwrap();

        let p2 = DMatrix::from_row_slice(2, 2, &[0.6, 0.4, 0.5, 0.5]);
        tensor
            .add_time_slice(TimeIndex::new(10.0).unwrap(), p2)
            .unwrap();

        let avg = tensor
            .average_transition(
                TimeIndex::new(0.0).unwrap(),
                TimeIndex::new(10.0).unwrap(),
                100,
            )
            .unwrap();

        // Average should be close to midpoint values
        assert_relative_eq!(avg[(0, 0)], 0.7, epsilon = 0.01);
        assert_relative_eq!(avg[(0, 1)], 0.3, epsilon = 0.01);
    }

    #[test]
    #[verified_engine::verified]
    fn test_out_of_bounds_time() {
        let min_t = TimeIndex::new(0.0).unwrap();
        let max_t = TimeIndex::new(24.0).unwrap();
        let mut tensor = TransitionTensor::new(2, min_t, max_t);

        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);

        // Try to add slice outside bounds
        let result = tensor.add_time_slice(TimeIndex::new(30.0).unwrap(), p);
        assert!(result.is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_invalid_matrix() {
        let min_t = TimeIndex::new(0.0).unwrap();
        let max_t = TimeIndex::new(24.0).unwrap();
        let mut tensor = TransitionTensor::new(2, min_t, max_t);

        // Non-stochastic matrix (row doesn't sum to 1)
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.2, 0.4, 0.6]);
        let result = tensor.add_time_slice(TimeIndex::new(12.0).unwrap(), p);
        assert!(result.is_err());
    }
}
