// [cite:math_scalars]
use super::types::*;
use math_commons::constants::nuclear;
use math_scalars::{
    Dimensionless, FemtometersPerSecond, MeVFm, MeVSecond, NuclearMeVFmFrame, Quantity,
};
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

/// Calculates the dimensionless Sommerfeld/Gamow barrier-penetration parameter for alpha decay
/// using typed [`NuclearMeVFmFrame`] constants ($\hbar = (\hbar c) / c_{\text{nuc}}$).
///
/// # Arguments
/// * `z_daughter` - Atomic number of the daughter nucleus.
/// * `z_alpha` - Atomic number of the alpha particle (usually 2).
/// * `velocity` - Velocity of the alpha particle in $\text{fm}\cdot\text{s}^{-1}$.
#[verified_engine::verified]
pub fn gamow_factor(
    z_daughter: AtomicNumber,
    z_alpha: AtomicNumber,
    velocity: FemtometersPerSecond,
) -> Result<f64, NuclearError> {
    if !velocity.is_finite() || velocity.get() <= 0.0 {
        return Err(NuclearError::InvalidVelocity);
    }
    let hbar: MeVSecond = nuclear::HBAR_C / nuclear::LIGHT_SPEED;
    let numerator: MeVFm =
        nuclear::E_SQUARED * (PI * z_alpha.value() as f64 * z_daughter.value() as f64);
    let denominator: MeVFm = hbar * velocity;
    let eta: Quantity<f64, Dimensionless, NuclearMeVFmFrame> = numerator / denominator;
    Ok(eta.get())
}
