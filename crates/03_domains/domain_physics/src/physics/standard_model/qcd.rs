//! Quantum Chromodynamics (QCD) focuses on the strong interaction and asymptotic freedom.

use crate::error::StandardModelError;

/// Calculates the 1-loop running of the strong coupling constant $\alpha_s(Q^2)$.
///
/// QCD exhibits asymptotic freedom, meaning the coupling decreases at higher energy scales.
///
/// # Arguments
/// * `mu`: The starting energy scale $\mu$ (GeV).
/// * `alpha_mu`: The value of $\alpha_s$ at scale $\mu$.
/// * `q`: The target energy scale $Q$ (GeV).
/// * `nf`: The number of active quark flavors at these scales.
///
/// # Formula
/// $\alpha_s(Q^2) = \frac{\alpha_s(\mu^2)}{1 + \frac{\alpha_s(\mu^2)}{12\pi} (33 - 2N_f) \ln(Q^2/\mu^2)}$
///
/// # Returns
/// * `Ok(f64)`: The value of $\alpha_s(Q^2)$.
/// * `Err(StandardModelError)`: If an invalid parameter (e.g., negative energy) is provided.
#[verified_engine::verified]
pub fn running_coupling(
    mu: f64,
    alpha_mu: f64,
    q: f64,
    nf: f64,
) -> Result<f64, StandardModelError> {
    if mu <= 0.0 {
        return Err(StandardModelError::InvalidEnergyScale {
            scale: mu,
            context: "initial scale mu".to_string(),
        });
    }
    if q <= 0.0 {
        return Err(StandardModelError::InvalidEnergyScale {
            scale: q,
            context: "target scale q".to_string(),
        });
    }
    if alpha_mu <= 0.0 {
        return Err(StandardModelError::InvalidCouplingConstant { alpha: alpha_mu });
    }
    if nf < 0.0 {
        return Err(StandardModelError::InvalidFlavors { nf });
    }

    let beta0 = 33.0 - 2.0 * nf;
    let log_term = (q.powi(2) / mu.powi(2)).ln();
    let denominator = 1.0 + (alpha_mu / (12.0 * std::f64::consts::PI)) * beta0 * log_term;

    if denominator <= 0.0 {
        return Err(StandardModelError::LandauPole { scale: q });
    }

    Ok(alpha_mu / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_asymptotic_freedom() {
        // Test that alpha_s decreases as Q increases for Nf < 16
        let mu = 91.2; // Z mass scale
        let alpha_mu = 0.118;
        let nf = 5.0; // Active flavors at this scale

        let q_low = 10.0;
        let q_high = 1000.0;

        let alpha_low = running_coupling(mu, alpha_mu, q_low, nf).unwrap();
        let alpha_high = running_coupling(mu, alpha_mu, q_high, nf).unwrap();

        // At lower energy (q_low < mu), coupling should be higher (confinement direction)
        assert!(alpha_low > alpha_mu);

        // At higher energy (q_high > mu), coupling should be lower (asymptotic freedom)
        assert!(alpha_high < alpha_mu);
    }

    #[test]
    #[verified_engine::verified]
    fn test_errors() {
        assert!(running_coupling(-1.0, 0.1, 10.0, 5.0).is_err());
        assert!(running_coupling(10.0, 0.1, -10.0, 5.0).is_err());
        assert!(running_coupling(10.0, -0.1, 100.0, 5.0).is_err());
    }
}
