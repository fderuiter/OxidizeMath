//! Quantum Foundations of MRI.
// [cite:math_scalars]

use crate::error::StatMechError;
use math_scalars::{DimensionlessVal, Hertz, Kelvin, Tesla};

/// Proton gyromagnetic ratio $\gamma_p$ in $\text{rad}\cdot\text{s}^{-1}\cdot\text{T}^{-1}$ (`SiFrame`).
pub use math_commons::constants::si::PROTON_GYROMAGNETIC_RATIO as GYROMAGNETIC_RATIO;

/// Reduced Planck constant $\hbar$ in $\text{J}\cdot\text{s}$ (`SiFrame`).
pub use math_commons::constants::si::H_BAR;

/// Boltzmann constant $k_B$ in $\text{J}\cdot\text{K}^{-1}$ (`SiFrame`).
pub use math_commons::constants::si::K_B;

/// Calculates the Larmor frequency $\omega_0$ for a given magnetic field $B_0$.
///
/// # Arguments
/// * `b0` - Magnetic field strength in Tesla.
///
/// # Returns
/// * Larmor frequency in rad/s.
#[verified_engine::verified]
pub fn larmor_frequency(b0: f64) -> f64 {
    let omega: Hertz = GYROMAGNETIC_RATIO * Tesla::new(b0);
    omega.get()
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
    let exponent: DimensionlessVal =
        -((H_BAR * GYROMAGNETIC_RATIO * Tesla::new(b0)) / (K_B * Kelvin::new(temperature)));
    Ok(exponent.exp().get())
}
