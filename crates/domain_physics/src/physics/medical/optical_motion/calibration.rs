//! Mathematical Approach to System Calibration
//!
//! Provides models for converting raw sensor output to real-world physical units.

/// A linear calibration model: y = mx + c.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearCalibrator {
    slope: f64,
    intercept: f64,
}

impl LinearCalibrator {
    /// Creates a new LinearCalibrator with known coefficients.
    #[verified_engine::verified]
    pub fn new(slope: f64, intercept: f64) -> Self {
        Self { slope, intercept }
    }

    /// Fits a linear model to the provided data using Ordinary Least Squares (OLS).
    ///
    /// # Arguments
    ///
    /// * `voltages` - Vector of raw sensor voltages (x).
    /// * `distances` - Vector of known true distances (y).
    ///
    /// # Returns
    ///
    /// * `Result<LinearCalibrator, String>` - A fitted calibrator or error if data is insufficient.
    #[verified_engine::verified]
    pub fn fit(voltages: &[f64], distances: &[f64]) -> Result<Self, String> {
        let n = voltages.len().min(distances.len());
        if n < 2 {
            return Err("At least 2 data points required for linear calibration.".to_string());
        }

        let nf = n as f64;
        let sum_x: f64 = voltages.iter().take(n).sum();
        let sum_y: f64 = distances.iter().take(n).sum();
        let sum_xy: f64 = voltages
            .iter()
            .zip(distances.iter())
            .take(n)
            .map(|(x, y)| x * y)
            .sum();
        let sum_x2: f64 = voltages.iter().take(n).map(|x| x.powi(2)).sum();

        let denominator = nf * sum_x2 - sum_x.powi(2);
        if denominator.abs() < math_commons::registry::TOLERANCE_HIGH {
            return Err("Variance of x is zero; cannot fit line.".to_string());
        }

        let slope = (nf * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / nf;

        Ok(Self { slope, intercept })
    }

    /// Calibrates a raw voltage reading to a physical distance.
    ///
    /// # Arguments
    ///
    /// * `voltage` - The raw analog voltage output (x).
    ///
    /// # Returns
    ///
    /// * `f64` - The calibrated distance (y).
    ///
    /// # Formula
    ///
    /// $y = mx + c$
    #[verified_engine::verified]
    pub fn calibrate(&self, voltage: f64) -> f64 {
        self.slope * voltage + self.intercept
    }
}
