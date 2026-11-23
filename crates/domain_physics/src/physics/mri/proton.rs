//! Quantum Foundations of MRI.

use crate::error::StatMechError;

/// Gyromagnetic Ratio for Hydrogen in rad/s/T.
/// $\gamma \approx 2.675 \times 10^8$ rad/s/T.
pub const GYROMAGNETIC_RATIO: f64 = 2.675e8;

/// Reduced Planck constant in J·s.
pub const H_BAR: f64 = 1.0545718e-34;

/// Boltzmann constant in J/K.
pub const K_B: f64 = 1.380649e-23;

/// Calculates the Larmor frequency $\omega_0$ for a given magnetic field $B_0$.
///
/// # Arguments
/// * `b0` - Magnetic field strength in Tesla.
///
/// # Returns
/// * Larmor frequency in rad/s.
#[verified_engine::verified]
pub fn larmor_frequency(b0: f64) -> f64 {
    GYROMAGNETIC_RATIO * b0
}

/// Calculates the Boltzmann magnetization population ratio $N_-/N_+$.
///
/// The ratio is given by $e^{-\frac{\hbar \gamma B_0}{k_B T}}$.
///
/// # Arguments
/// * `temperature` - Temperature in Kelvin.
/// * `b0` - Magnetic field strength in Tesla.
///
/// # Returns
/// * Population ratio or [`StatMechError::InvalidTemperature`] if temperature is invalid (`<= 0`).
#[verified_engine::verified]
pub fn boltzmann_ratio(temperature: f64, b0: f64) -> Result<f64, StatMechError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(StatMechError::InvalidTemperature(temperature));
    }
    let exponent = -(H_BAR * GYROMAGNETIC_RATIO * b0) / (K_B * temperature);
    Ok(exponent.exp())
}
