//! Clutter Removal and Noise Estimation Algorithms.
//!
//! Implements algorithms for distinguishing target reflections from background noise in radar data.

use num_complex::Complex;
use std::collections::VecDeque;

/// Cell Averaging Constant False Alarm Rate (CA-CFAR) Noise Estimation.
///
/// Estimates the noise level ($E$) around a cell under test (CUT) by averaging the power
/// of reference cells in the surrounding window.
///
/// # Arguments
///
/// * `reference_cells` - A slice of values representing the power of reference cells ($x_i$).
///
/// # Returns
///
/// * `f64` - The estimated noise level ($E$).
///
/// # Formula
///
/// $E = \frac{1}{N} \sum_{i=1}^{N} x_i$
#[verified_engine::verified]
pub fn ca_cfar_noise_level(reference_cells: &[f64]) -> f64 {
    if reference_cells.is_empty() {
        return 0.0;
    }
    let sum: f64 = reference_cells.iter().sum();
    sum / reference_cells.len() as f64
}

/// Elliptical Filter for static clutter rejection.
///
/// Uses a sliding window to estimate the center of the static clutter (DC offset)
/// and removes it from the signal.
///
/// The "Elliptical" name refers to the decision boundary in the complex plane,
/// but essentially this acts as a high-pass filter to remove the stationary component.
pub struct EllipticalFilter {
    /// Sliding window of signal history.
    history: VecDeque<Complex<f64>>,
    /// Maximum window size ($N$).
    window_size: usize,
}

impl EllipticalFilter {
    /// Creates a new EllipticalFilter.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Window size parameter (as f64, cast to usize).
    #[verified_engine::verified]
    pub fn new(alpha: f64) -> Self {
        Self {
            history: VecDeque::new(),
            window_size: alpha as usize,
        }
    }

    /// Processes a new signal point and returns the filtered value.
    ///
    /// The filter estimates the clutter center from the history window and subtracts it.
    ///
    /// # Arguments
    ///
    /// * `signal` - Complex radar signal sample.
    ///
    /// # Returns
    ///
    /// * `Complex<f64>` - The signal with clutter removed ($x_{out} = x_{in} - \mu_{clutter}$).
    #[verified_engine::verified]
    pub fn filter(&mut self, signal: Complex<f64>) -> Complex<f64> {
        self.history.push_back(signal);
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        let estimate = self.get_clutter_estimate();
        signal - estimate
    }

    /// Returns the current estimated static clutter center.
    ///
    /// Calculated as the mean of the signal history.
    #[verified_engine::verified]
    pub fn get_clutter_estimate(&self) -> Complex<f64> {
        if self.history.is_empty() {
            return Complex::new(0.0, 0.0);
        }

        let sum: Complex<f64> = self.history.iter().sum();
        sum / (self.history.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_ca_cfar() {
        let refs = vec![10.0, 12.0, 8.0, 10.0];
        let noise = ca_cfar_noise_level(&refs);
        assert!((noise - 10.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }
}
