use super::KB;
use crate::error::StatMechError;
use crate::physics::solid_state::types::{ELEMENTARY_CHARGE_C, ElectronVolts, Kelvin};

/// Canonical ensemble thermodynamic potentials $(Z, F, U, S)$ evaluated from [`ElectronVolts`] levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanonicalThermodynamics {
    /// Canonical partition function $Z = \sum_i e^{-\beta E_i}$.
    pub partition_function: f64,
    /// Helmholtz free energy $F = -k_B T \ln Z$ (`eV`, log-sum-exp stabilized).
    pub free_energy: ElectronVolts,
    /// Canonical internal energy $U = \langle E \rangle$ (`eV`, log-sum-exp stabilized).
    pub internal_energy: ElectronVolts,
    /// Canonical entropy $S = (U - F) / T$ in `eV / K`.
    pub entropy_ev_per_k: f64,
    /// Canonical entropy $S = (U - F) / T$ in SI `J / K`.
    pub entropy_j_per_k: f64,
}

/// Calculates the Boltzmann factor for a given energy state.
///
/// Formula: e^(-beta * E) where beta = 1/(k_B * T).
///
/// # Arguments
/// * `energy` - The energy of the state (Joules).
/// * `temperature` - The temperature of the system (Kelvin).
///
/// # Returns
/// * `f64` - The unnormalized probability weight.
#[verified_engine::verified]
pub fn boltzmann_factor(energy: f64, temperature: f64) -> f64 {
    if temperature <= 0.0 && temperature == 0.0 {
        return 0.0;
    }
    let beta = 1.0 / (KB * temperature);
    (-beta * energy).exp()
}

/// Calculates the Partition Function (Z).
///
/// Formula: Z = sum_i e^(-beta * E_i)
///
/// # Arguments
/// * `energies` - A slice of energy values for all microstates.
/// * `temperature` - Temperature in Kelvin.
///
/// # Returns
/// * `f64` - The partition function Z.
#[verified_engine::verified]
pub fn calculate_partition_function(energies: &[f64], temperature: f64) -> f64 {
    energies
        .iter()
        .map(|&e| boltzmann_factor(e, temperature))
        .sum()
}

/// Calculates the Helmholtz Free Energy (F).
///
/// Formula: F = -k_B * T * ln(Z)
///
/// # Arguments
/// * `partition_function` - The pre-calculated partition function Z.
/// * `temperature` - Temperature in Kelvin.
///
/// # Returns
/// * `f64` - Helmholtz Free Energy in Joules.
#[verified_engine::verified]
pub fn helmholtz_free_energy(partition_function: f64, temperature: f64) -> f64 {
    -KB * temperature * partition_function.ln()
}

/// Calculates the Average Energy (Internal Energy U).
///
/// Formula: U = (1/Z) * sum_i E_i * e^(-beta * E_i)
///
/// # Arguments
/// * `energies` - Slice of energy states.
/// * `temperature` - Temperature in Kelvin.
///
/// # Returns
/// * `f64` - Internal Energy U in Joules.
#[verified_engine::verified]
pub fn average_energy(energies: &[f64], temperature: f64) -> f64 {
    let z = calculate_partition_function(energies, temperature);
    if z == 0.0 {
        return 0.0;
    }
    let sum_weighted_energy: f64 = energies
        .iter()
        .map(|&e| e * boltzmann_factor(e, temperature))
        .sum();
    sum_weighted_energy / z
}

/// Calculates the Entropy (S).
///
/// Formula: S = (U - F) / T
///
/// # Arguments
/// * `internal_energy` - Internal Energy U.
/// * `free_energy` - Helmholtz Free Energy F.
/// * `temperature` - Temperature in Kelvin.
///
/// # Returns
/// * `f64` - Entropy S in J/K.
#[verified_engine::verified]
pub fn entropy(internal_energy: f64, free_energy: f64, temperature: f64) -> f64 {
    if temperature == 0.0 {
        return 0.0;
    }
    (internal_energy - free_energy) / temperature
}

/// Evaluates the Boltzmann factor $e^{-E / (k_B T)}$ from [`ElectronVolts`] and [`Kelvin`].
#[verified_engine::verified]
pub fn boltzmann_factor_ev(energy: ElectronVolts, temperature: Kelvin) -> f64 {
    boltzmann_factor(energy.to_joules(), temperature.as_f64())
}

/// Evaluates the canonical partition function $Z = \sum_i e^{-E_i / (k_B T)}$ over [`ElectronVolts`] levels.
#[verified_engine::verified]
pub fn calculate_partition_function_ev(energies: &[ElectronVolts], temperature: Kelvin) -> f64 {
    let t_k = temperature.as_f64();
    if t_k <= 0.0 {
        return 0.0;
    }
    let mut z = 0.0;
    for &e in energies {
        z += boltzmann_factor(e.to_joules(), t_k);
    }
    z
}

#[inline]
fn min_and_ground_degeneracy(energies: &[ElectronVolts]) -> (f64, usize) {
    let mut e_min = f64::INFINITY;
    for &e in energies {
        let v = e.as_f64();
        if v < e_min {
            e_min = v;
        }
    }
    let mut g0 = 0usize;
    for &e in energies {
        if (e.as_f64() - e_min).abs() <= 1e-12 {
            g0 += 1;
        }
    }
    (e_min, g0.max(1))
}

/// Evaluates underflow-safe Helmholtz free energy $F = -k_B T \ln Z$ (`eV`) over [`ElectronVolts`] levels.
#[verified_engine::verified]
pub fn helmholtz_free_energy_ev(energies: &[ElectronVolts], temperature: Kelvin) -> ElectronVolts {
    if energies.is_empty() {
        return ElectronVolts::new(0.0);
    }
    let (e_min, g0) = min_and_ground_degeneracy(energies);
    if temperature.as_f64() <= 1e-12 {
        let _ = g0;
        return ElectronVolts::new(e_min);
    }
    let kt_ev = temperature.thermal_energy_ev().as_f64();
    let mut z_shifted = 0.0;
    for &e in energies {
        let delta = (e.as_f64() - e_min) / kt_ev;
        if delta <= 700.0 {
            z_shifted += (-delta).exp();
        }
    }
    ElectronVolts::new(e_min - kt_ev * z_shifted.ln())
}

/// Evaluates underflow-safe canonical internal energy $U = \langle E \rangle$ (`eV`) over [`ElectronVolts`] levels.
#[verified_engine::verified]
pub fn average_energy_ev(energies: &[ElectronVolts], temperature: Kelvin) -> ElectronVolts {
    if energies.is_empty() {
        return ElectronVolts::new(0.0);
    }
    let (e_min, _) = min_and_ground_degeneracy(energies);
    if temperature.as_f64() <= 1e-12 {
        return ElectronVolts::new(e_min);
    }
    let kt_ev = temperature.thermal_energy_ev().as_f64();
    let mut z_shifted = 0.0;
    let mut weighted_sum = 0.0;
    for &e in energies {
        let e_val = e.as_f64();
        let delta = (e_val - e_min) / kt_ev;
        if delta <= 700.0 {
            let w = (-delta).exp();
            z_shifted += w;
            weighted_sum += e_val * w;
        }
    }
    if z_shifted == 0.0 {
        ElectronVolts::new(e_min)
    } else {
        ElectronVolts::new(weighted_sum / z_shifted)
    }
}

/// Evaluates canonical entropy $S = (U - F) / T$ in `eV / K`.
#[verified_engine::verified]
pub fn entropy_ev_per_kelvin(
    internal_energy: ElectronVolts,
    free_energy: ElectronVolts,
    temperature: Kelvin,
) -> f64 {
    let t_k = temperature.as_f64();
    if t_k <= 1e-12 {
        return 0.0;
    }
    (internal_energy.as_f64() - free_energy.as_f64()) / t_k
}

/// Computes canonical $(Z, F, U, S)$ thermodynamics from [`ElectronVolts`] microstates and [`Kelvin`] temperature.
///
/// # Errors
///
/// Returns [`StatMechError::InvalidTemperature`] if `temperature < 0` or any microstate energy is non-finite.
#[verified_engine::verified]
pub fn canonical_thermodynamics_ev(
    energies: &[ElectronVolts],
    temperature: Kelvin,
) -> Result<CanonicalThermodynamics, StatMechError> {
    let t_k = temperature.as_f64();
    if t_k < 0.0 || !t_k.is_finite() {
        return Err(StatMechError::InvalidTemperature(t_k));
    }
    if energies.is_empty() || energies.iter().any(|e| !e.as_f64().is_finite()) {
        return Err(StatMechError::InvalidChemicalPotential {
            chemical_potential: 0.0,
            energy: f64::NAN,
            reason: "Canonical ensemble requires at least one finite eV microstate".to_string(),
        });
    }
    let (e_min, g0) = min_and_ground_degeneracy(energies);
    if t_k <= 1e-12 {
        let s_j_k = KB * (g0 as f64).ln();
        let s_ev_k = s_j_k / ELEMENTARY_CHARGE_C;
        let z0 = if e_min.abs() <= 1e-15 { g0 as f64 } else { 0.0 };
        return Ok(CanonicalThermodynamics {
            partition_function: z0,
            free_energy: ElectronVolts::new(e_min),
            internal_energy: ElectronVolts::new(e_min),
            entropy_ev_per_k: s_ev_k,
            entropy_j_per_k: s_j_k,
        });
    }
    let z = calculate_partition_function_ev(energies, temperature);
    let f = helmholtz_free_energy_ev(energies, temperature);
    let u = average_energy_ev(energies, temperature);
    let s_ev_k = entropy_ev_per_kelvin(u, f, temperature);
    Ok(CanonicalThermodynamics {
        partition_function: z,
        free_energy: f,
        internal_energy: u,
        entropy_ev_per_k: s_ev_k,
        entropy_j_per_k: s_ev_k * ELEMENTARY_CHARGE_C,
    })
}

// [cite:graph_parameters_rust]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_z_consistency() {
        let energies = vec![0.0];
        let t = 300.0;

        let z = calculate_partition_function(&energies, t);
        let f = helmholtz_free_energy(z, t);

        assert!(
            (z - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD,
            "Z should be 1 for single E=0 state"
        );
        assert!(
            f.abs() < math_commons::registry::TOLERANCE_STANDARD,
            "F should be 0 for Z=1"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_canonical_ev_bridges_and_cryogenic_underflow_stability() {
        let spectrum = [
            ElectronVolts::new(0.0),
            ElectronVolts::new(0.025),
            ElectronVolts::new(0.050),
        ];
        let t_room = Kelvin::new(300.0);
        let summary = canonical_thermodynamics_ev(&spectrum, t_room).unwrap();
        assert!(summary.partition_function > 1.3);
        assert!(summary.free_energy.as_f64() < 0.0);
        assert!(summary.internal_energy.as_f64() > 0.0);
        assert!(summary.entropy_ev_per_k > 0.0);
        // Verify F = U - T * S identity
        let reconstructed_f =
            summary.internal_energy.as_f64() - t_room.as_f64() * summary.entropy_ev_per_k;
        assert!((reconstructed_f - summary.free_energy.as_f64()).abs() < 1e-14);

        // Shifted positive spectrum at cryogenic T = 2 K (beta * E_min ≈ 5800 > 709):
        // Raw exp(-beta E) underflows Z -> 0, whereas log-sum-exp preserves F ≈ U ≈ E_min.
        let shifted = [ElectronVolts::new(1.0), ElectronVolts::new(1.05)];
        let cryo = canonical_thermodynamics_ev(&shifted, Kelvin::new(2.0)).unwrap();
        assert!((cryo.free_energy.as_f64() - 1.0).abs() < 1e-9);
        assert!((cryo.internal_energy.as_f64() - 1.0).abs() < 1e-9);
        assert!(cryo.entropy_ev_per_k >= 0.0 && cryo.entropy_ev_per_k.is_finite());
    }
}
