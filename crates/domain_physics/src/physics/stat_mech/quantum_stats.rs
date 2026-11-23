use super::KB;
use crate::error::StatMechError;

// --- Strategy Pattern Implementation ---

/// Strategy for calculating particle occupancy probability.
pub trait StatisticalDistribution {
    /// Calculates the occupancy probability / average occupation number.
    #[verified_engine::verified]
    fn occupancy(
        &self,
        energy: f64,
        chemical_potential: f64,
        temperature: f64,
    ) -> Result<f64, StatMechError>;
}

/// Fermi-Dirac Statistics (Fermions).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FermiDirac;

impl StatisticalDistribution for FermiDirac {
    #[verified_engine::verified]
    fn occupancy(
        &self,
        energy: f64,
        chemical_potential: f64,
        temperature: f64,
    ) -> Result<f64, StatMechError> {
        if temperature < 0.0 {
            return Err(StatMechError::InvalidTemperature(temperature));
        }
        if temperature <= 1e-12 {
            // T -> 0 limit
            if energy < chemical_potential {
                return Ok(1.0);
            } else {
                return Ok(0.0);
            }
        }
        let beta = 1.0 / (KB * temperature);
        let exponent = beta * (energy - chemical_potential);
        // Avoid overflow for large positive exponent
        if exponent > 700.0 {
            return Ok(0.0);
        }
        // Avoid overflow for large negative exponent
        if exponent < -700.0 {
            return Ok(1.0);
        }
        let denom = exponent.exp() + 1.0;
        Ok(1.0 / denom)
    }
}

/// Bose-Einstein Statistics (Bosons).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoseEinstein;

impl StatisticalDistribution for BoseEinstein {
    #[verified_engine::verified]
    fn occupancy(
        &self,
        energy: f64,
        chemical_potential: f64,
        temperature: f64,
    ) -> Result<f64, StatMechError> {
        if temperature < 0.0 {
            return Err(StatMechError::InvalidTemperature(temperature));
        }
        if temperature <= 1e-12 {
            if chemical_potential > energy {
                return Err(StatMechError::InvalidChemicalPotential {
                    chemical_potential,
                    energy,
                    reason: "Mu > E for Bosons at T=0".to_string(),
                });
            }
            // If mu == E, divergence (condensate).
            return Ok(0.0); // Assume ground state condensation handled elsewhere? Or infinity.
        }
        if chemical_potential > energy {
            return Err(StatMechError::InvalidChemicalPotential {
                chemical_potential,
                energy,
                reason: "Mu > E for Bosons".to_string(),
            });
        }
        let beta = 1.0 / (KB * temperature);
        let exponent = beta * (energy - chemical_potential);
        // Check for singularity if exponent is 0 (E=mu)
        if exponent.abs() < math_commons::registry::TOLERANCE_STANDARD {
            return Ok(f64::INFINITY);
        }
        // Avoid overflow
        if exponent > 700.0 {
            return Ok(0.0);
        }
        let denom = exponent.exp() - 1.0;
        Ok(1.0 / denom)
    }
}

/// Maxwell-Boltzmann Statistics (Classical).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaxwellBoltzmann;

impl StatisticalDistribution for MaxwellBoltzmann {
    #[verified_engine::verified]
    fn occupancy(
        &self,
        energy: f64,
        chemical_potential: f64,
        temperature: f64,
    ) -> Result<f64, StatMechError> {
        if temperature < 0.0 {
            return Err(StatMechError::InvalidTemperature(temperature));
        }
        if temperature <= 1e-12 {
            // e^-inf or e^inf
            if energy < chemical_potential {
                return Ok(f64::INFINITY);
            } else {
                return Ok(0.0);
            }
        }
        let beta = 1.0 / (KB * temperature);
        let exponent = beta * (energy - chemical_potential);
        // Avoid overflow
        if exponent > 700.0 {
            // exp(-large) -> 0
            return Ok(0.0);
        }
        Ok((-exponent).exp())
    }
}

/// Generic function to calculate occupancy using a specific strategy.
#[verified_engine::verified]
pub fn calculate_occupancy<D: StatisticalDistribution>(
    distribution: D,
    energy: f64,
    chemical_potential: f64,
    temperature: f64,
) -> Result<f64, StatMechError> {
    distribution.occupancy(energy, chemical_potential, temperature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_fermion_limit() {
        // T -> 0 (use very small T)
        // E < mu -> prob 1
        // E > mu -> prob 0
        let t = math_commons::registry::TOLERANCE_STANDARD; // effectively 0
        // Actually, formulas use raw Joules.
        // Let's use E in Joules.
        let mu_j = 1.0e-20;
        let e_below = 0.5e-20;
        let e_above = 1.5e-20;

        let prob_below = calculate_occupancy(FermiDirac, e_below, mu_j, t).unwrap();

        let prob_above = calculate_occupancy(FermiDirac, e_above, mu_j, t).unwrap();

        assert!(
            (prob_below - 1.0).abs() < math_commons::registry::TOLERANCE_FAST,
            "Fermion below mu should be occupied at T~0"
        );
        assert!(
            prob_above.abs() < math_commons::registry::TOLERANCE_FAST,
            "Fermion above mu should be empty at T~0"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_strategy_usage() {
        // Test using the Strategy API directly
        let t = 300.0;
        let e = 1.0 * KB * t; // Energy = kT
        let mu = 0.0;

        // Fermi-Dirac: 1 / (e^1 + 1)
        let fd = FermiDirac;
        let prob_fd = calculate_occupancy(fd, e, mu, t).unwrap();
        let expected_fd = 1.0 / (std::f64::consts::E + 1.0);
        assert!((prob_fd - expected_fd).abs() < math_commons::registry::TOLERANCE_STANDARD);

        // Maxwell-Boltzmann: e^-1
        let mb = MaxwellBoltzmann;
        let prob_mb = calculate_occupancy(mb, e, mu, t).unwrap();
        let expected_mb = (-1.0_f64).exp();
        assert!((prob_mb - expected_mb).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }
}
