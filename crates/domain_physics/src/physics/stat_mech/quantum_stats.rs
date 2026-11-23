use super::KB;
use crate::error::StatMechError;
use crate::physics::solid_state::types::{ElectronVolts, Kelvin};

// --- Strategy Pattern Implementation ---

/// Strategy for calculating particle occupancy probability.
pub trait StatisticalDistribution {
    /// Calculates the occupancy probability / average occupation number (inputs in Joules and Kelvin).
    #[verified_engine::verified]
    fn occupancy(
        &self,
        energy: f64,
        chemical_potential: f64,
        temperature: f64,
    ) -> Result<f64, StatMechError>;
}

/// Fermi-Dirac Statistics (Fermions).
#[derive(Debug, Clone, Copy, Default)]
pub struct FermiDirac;

impl StatisticalDistribution for FermiDirac {
    #[verified_engine::verified]
    fn occupancy(
        &self,
        energy: f64,
        chemical_potential: f64,
        temperature: f64,
    ) -> Result<f64, StatMechError> {
        if temperature < 0.0 || !temperature.is_finite() {
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
#[derive(Debug, Clone, Copy, Default)]
pub struct BoseEinstein;

impl StatisticalDistribution for BoseEinstein {
    #[verified_engine::verified]
    fn occupancy(
        &self,
        energy: f64,
        chemical_potential: f64,
        temperature: f64,
    ) -> Result<f64, StatMechError> {
        if temperature < 0.0 || !temperature.is_finite() {
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
            if (energy - chemical_potential).abs() <= 1e-30 {
                return Ok(f64::INFINITY);
            }
            return Ok(0.0);
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
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxwellBoltzmann;

impl StatisticalDistribution for MaxwellBoltzmann {
    #[verified_engine::verified]
    fn occupancy(
        &self,
        energy: f64,
        chemical_potential: f64,
        temperature: f64,
    ) -> Result<f64, StatMechError> {
        if temperature < 0.0 || !temperature.is_finite() {
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

/// Generic function to calculate occupancy using a specific strategy (SI Joules and Kelvin).
#[verified_engine::verified]
pub fn calculate_occupancy<D: StatisticalDistribution>(
    distribution: D,
    energy: f64,
    chemical_potential: f64,
    temperature: f64,
) -> Result<f64, StatMechError> {
    distribution.occupancy(energy, chemical_potential, temperature)
}

/// Evaluates particle occupancy from strongly typed [`ElectronVolts`] energies and [`Kelvin`] temperature.
///
/// Automatically converts `energy` and `chemical_potential` to SI Joules via [`ElectronVolts::to_joules`]
/// so Boltzmann exponents $(E - \mu) / (k_B T)$ remain dimensionless and accurately scaled.
///
/// # Errors
///
/// Returns [`StatMechError::InvalidTemperature`] for negative or non-finite temperatures, or
/// [`StatMechError::InvalidChemicalPotential`] (reported in `eV`) when occupation diverges or violates
/// bosonic $\mu < E$ bounds.
#[verified_engine::verified]
pub fn calculate_occupancy_ev<D: StatisticalDistribution>(
    distribution: D,
    energy: ElectronVolts,
    chemical_potential: ElectronVolts,
    temperature: Kelvin,
) -> Result<f64, StatMechError> {
    let e_ev = energy.as_f64();
    let mu_ev = chemical_potential.as_f64();
    let t_k = temperature.as_f64();
    if !e_ev.is_finite() || !mu_ev.is_finite() {
        return Err(StatMechError::InvalidChemicalPotential {
            chemical_potential: mu_ev,
            energy: e_ev,
            reason: "Energy and chemical potential must be finite eV values".to_string(),
        });
    }

    let val = distribution
        .occupancy(energy.to_joules(), chemical_potential.to_joules(), t_k)
        .map_err(|err| match err {
            StatMechError::InvalidChemicalPotential { reason, .. } => {
                StatMechError::InvalidChemicalPotential {
                    chemical_potential: mu_ev,
                    energy: e_ev,
                    reason,
                }
            }
            other => other,
        })?;

    if !val.is_finite() {
        return Err(StatMechError::InvalidChemicalPotential {
            chemical_potential: mu_ev,
            energy: e_ev,
            reason: "Occupation diverges at chemical potential boundary (mu >= E)".to_string(),
        });
    }
    Ok(val)
}

// [cite:graph_parameters_rust]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_fermion_limit() {
        let t = math_commons::registry::TOLERANCE_STANDARD;
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
        let t = 300.0;
        let e = 1.0 * KB * t;
        let mu = 0.0;

        let fd = FermiDirac;
        let prob_fd = calculate_occupancy(fd, e, mu, t).unwrap();
        let expected_fd = 1.0 / (std::f64::consts::E + 1.0);
        assert!((prob_fd - expected_fd).abs() < math_commons::registry::TOLERANCE_STANDARD);

        let mb = MaxwellBoltzmann;
        let prob_mb = calculate_occupancy(mb, e, mu, t).unwrap();
        let expected_mb = (-1.0_f64).exp();
        assert!((prob_mb - expected_mb).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_ev_kelvin_bridge_eliminates_fermi_clamping_error_and_guards_bosons() {
        let t = Kelvin::new(300.0);
        let mu = ElectronVolts::new(0.500);
        // 1 meV below mu at 300 K (k_B T ≈ 25.85 meV): f(E) = 1 / (exp(-1/25.852) + 1) ≈ 0.509669
        // Raw float without Joule conversion clamped to 1.0 (~0.49 error).
        let e_near = ElectronVolts::new(0.499);
        let occ_fd = calculate_occupancy_ev(FermiDirac, e_near, mu, t).unwrap();
        let kt_ev = t.thermal_energy_ev().as_f64();
        let expected = 1.0 / (((-0.001_f64) / kt_ev).exp() + 1.0);
        assert!((occ_fd - expected).abs() < 1e-12);
        assert!((occ_fd - 0.509_669).abs() < 1e-5);

        // At E == mu for T > 0, Fermi-Dirac occupancy is exactly 0.5
        let occ_half = calculate_occupancy_ev(FermiDirac, mu, mu, t).unwrap();
        assert!((occ_half - 0.5).abs() < 1e-12);

        // Bose-Einstein valid for mu < E, structured StatMechError for mu == E and mu > E
        let occ_be = calculate_occupancy_ev(BoseEinstein, ElectronVolts::new(0.55), mu, t).unwrap();
        let expected_be = 1.0 / ((0.05 / kt_ev).exp() - 1.0);
        assert!((occ_be - expected_be).abs() < 1e-12);

        assert!(matches!(
            calculate_occupancy_ev(BoseEinstein, mu, mu, t),
            Err(StatMechError::InvalidChemicalPotential { .. })
        ));
        assert!(matches!(
            calculate_occupancy_ev(BoseEinstein, ElectronVolts::new(0.45), mu, t),
            Err(StatMechError::InvalidChemicalPotential { .. })
        ));
    }
}
