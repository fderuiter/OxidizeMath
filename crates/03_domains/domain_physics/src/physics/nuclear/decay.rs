use super::types::*;
use math_commons::constants;
use std::f64::consts::PI;

/// Calculates the remaining amount of a substance.
///
/// Formula: N(t) = N0 * e^(-lambda * t)
///
/// # Arguments
/// * `initial_quantity` - Initial quantity (N0).
/// * `half_life` - Half-life in seconds.
/// * `time` - Time elapsed in seconds.
#[verified_engine::verified]
pub fn calculate_remaining(
    initial_quantity: f64,
    half_life: f64,
    time: f64,
) -> Result<f64, NuclearError> {
    if half_life <= 0.0 {
        return Err(NuclearError::InvalidHalfLife);
    }
    let lambda = 2.0_f64.ln() / half_life;
    Ok(initial_quantity * (-lambda * time).exp())
}

/// Calculates the Gamow factor for alpha decay.
///
/// # Arguments
/// * `z_daughter` - Atomic number of the daughter nucleus.
/// * `z_alpha` - Atomic number of the alpha particle (usually 2).
/// * `velocity` - Velocity of the alpha particle in fm/s.
#[verified_engine::verified]
pub fn gamow_factor(
    z_daughter: AtomicNumber,
    z_alpha: AtomicNumber,
    velocity: f64,
) -> Result<f64, NuclearError> {
    if velocity <= 0.0 {
        return Err(NuclearError::InvalidVelocity);
    }
    let hbar = constants::HBAR_C / constants::LIGHT_SPEED;
    let numerator = PI * z_alpha.value() as f64 * z_daughter.value() as f64 * constants::E_SQUARED;
    let denominator = hbar * velocity;
    Ok(numerator / denominator)
}
