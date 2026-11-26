//! Lattice Dynamics (Phonons)
//!
//! Describes the collective excitations (vibrations) of the crystal lattice.

use std::f64::consts::PI;

/// Acoustic phonon dispersion relation (Debye Approximation).
///
/// \omega(k) = v_s * k
/// Linearly dependent on wavevector k for small k.
#[verified_engine::verified]
pub fn acoustic_dispersion(k: f64, v_s: f64) -> f64 {
    v_s * k
}

/// Optical phonon dispersion relation (Einstein Model).
///
/// \omega(k) = \omega_E (constant)
/// Assumes independent oscillators.
#[verified_engine::verified]
pub fn optical_dispersion(_k: f64, w_e: f64) -> f64 {
    w_e
}

/// Debye Heat Capacity Cv at low temperatures.
///
/// C_v \propto T^3
/// Formula: (12 \pi^4 / 5) * N * k_B * (T / \Theta_D)^3
#[verified_engine::verified]
pub fn debye_heat_capacity_low_temp(t: f64, theta_d: f64, n_atoms: f64, k_b: f64) -> f64 {
    let prefactor = (12.0 * PI.powi(4)) / 5.0;
    prefactor * n_atoms * k_b * (t / theta_d).powi(3)
}
