use super::types::*;
use math_commons::constants::property_constants;
use std::f64::consts::PI;

/// Calculates the nuclear radius using the formula R = R0 * A^(1/3).
///
/// # Arguments
/// * `mass_number` - Mass number A.
///
/// # Returns
/// * `f64` - The radius in femtometers (fm).
#[verified_engine::verified]
pub fn calculate_radius(mass_number: MassNumber) -> f64 {
    property_constants::R0 * (mass_number.value() as f64).powf(1.0 / 3.0)
}

/// Calculates the nucleon density (nucleons per volume).
///
/// # Arguments
/// * `mass_number` - Mass number A.
///
/// # Returns
/// * `Result<f64, NuclearError>` - The density in nucleons/fm^3.
#[verified_engine::verified]
pub fn calculate_nucleon_density(mass_number: MassNumber) -> Result<f64, NuclearError> {
    let radius = calculate_radius(mass_number);
    let volume = (4.0 / 3.0) * PI * radius.powi(3);
    if volume == 0.0 {
        return Err(NuclearError::VolumeZero);
    }
    Ok(mass_number.value() as f64 / volume)
}
