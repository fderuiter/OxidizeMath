use crate::error::HighEnergyError;
use math_commons::constants::C;

/// Calculates the density compression ratio r for a strong relativistic shock.
/// Formula: r = (gamma_hat + 1) / (gamma_hat - 1)
///
/// # Arguments
/// * `adiabatic_index` - The adiabatic index (gamma_hat).
///
/// # Errors
/// * `HighEnergyError::InvalidAdiabaticIndex` if `adiabatic_index <= 1`.
#[verified_engine::verified]
pub fn shock_compression_ratio(adiabatic_index: f64) -> Result<f64, HighEnergyError> {
    if adiabatic_index <= 1.0 {
        return Err(HighEnergyError::InvalidAdiabaticIndex {
            gamma: adiabatic_index,
        });
    }
    let r = (adiabatic_index + 1.0) / (adiabatic_index - 1.0);
    Ok(r)
}

/// Calculates the specific enthalpy h.
/// Formula: h = 1 + (gamma_hat / (gamma_hat - 1)) * (P / (rho * c^2))
///
/// # Errors
/// * `HighEnergyError::InvalidAdiabaticIndex` if `adiabatic_index <= 1`.
/// * `HighEnergyError::InvalidDensity` if `density <= 0`.
/// * `HighEnergyError::InvalidPressure` if `pressure < 0`.
#[verified_engine::verified]
pub fn specific_enthalpy(
    adiabatic_index: f64,
    pressure: f64,
    density: f64,
) -> Result<f64, HighEnergyError> {
    if adiabatic_index <= 1.0 {
        return Err(HighEnergyError::InvalidAdiabaticIndex {
            gamma: adiabatic_index,
        });
    }
    if density <= 0.0 {
        return Err(HighEnergyError::InvalidDensity { rho: density });
    }
    if pressure < 0.0 {
        return Err(HighEnergyError::InvalidPressure { p: pressure });
    }

    let term = (adiabatic_index / (adiabatic_index - 1.0)) * (pressure / (density * C.powi(2)));
    Ok(1.0 + term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_fluid_dynamics() {
        // Compression ratio, gamma_hat = 4/3.
        // r = (4/3 + 1) / (4/3 - 1) = (7/3) / (1/3) = 7.
        let r = shock_compression_ratio(4.0 / 3.0).expect("Failed to calc compression ratio");
        assert_relative_eq!(r, 7.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_errors() {
        assert!(shock_compression_ratio(1.0).is_err());
        assert!(specific_enthalpy(1.4, -1.0, 1.0).is_err());
        assert!(specific_enthalpy(1.4, 1.0, 0.0).is_err());
    }
}
