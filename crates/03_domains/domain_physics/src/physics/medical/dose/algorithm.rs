use crate::error::DoseFluenceError;

/// Calculates the Total Energy Released per Mass (TERMA) for a ray segment.
///
/// TERMA represents the primary energy fluence released into the medium at a point,
/// before accounting for secondary electron transport (scatter).
///
/// # Arguments
///
/// * `incident_fluence` ($\Psi_0$) - The initial radiant energy fluence.
/// * `mu` ($\mu$) - The linear attenuation coefficient of the medium (cm⁻¹).
/// * `depth` ($d$) - The radiological depth along the ray (cm).
///
/// # Returns
///
/// * `Result<f64, DoseFluenceError>` - The TERMA value.
///
/// # Formula
///
/// $T = \mu \Psi_0 e^{-\mu d}$
#[verified_engine::verified]
pub fn calculate_terma(
    incident_fluence: f64,
    mu: f64,
    depth: f64,
) -> Result<f64, DoseFluenceError> {
    if incident_fluence < 0.0 {
        return Err(DoseFluenceError::InvalidPhysicalQuantity(
            "incident_fluence".to_string(),
        ));
    }
    if mu < 0.0 {
        return Err(DoseFluenceError::InvalidPhysicalQuantity(
            "attenuation_coefficient".to_string(),
        ));
    }
    if depth < 0.0 {
        return Err(DoseFluenceError::InvalidPhysicalQuantity(
            "depth".to_string(),
        ));
    }

    Ok(mu * incident_fluence * (-mu * depth).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_terma_calculation() {
        assert!(calculate_terma(-1.0, 0.1, 10.0).is_err());
        assert_eq!(calculate_terma(100.0, 0.0, 10.0).unwrap(), 0.0);

        let t0 = calculate_terma(100.0, 0.1, 0.0).unwrap();
        assert!((t0 - 10.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }
}
