//! Magnetism (Heisenberg Model)
//!
//! Models magnetic ordering via exchange interactions between spins.

use nalgebra::Vector3;

/// Calculates the energy of a spin configuration under the Heisenberg Hamiltonian.
///
/// H = -J \sum_{<i,j>} S_i \cdot S_j
///
/// * J > 0: Ferromagnetic
/// * J < 0: Antiferromagnetic
#[verified_engine::verified]
pub fn calculate_heisenberg_energy(
    j: f64,
    spins: &[Vector3<f64>],
    neighbors: &[(usize, usize)],
) -> f64 {
    let mut sum_dot_products = 0.0;
    for &(idx1, idx2) in neighbors {
        if idx1 < spins.len() && idx2 < spins.len() {
            sum_dot_products += spins[idx1].dot(&spins[idx2]);
        }
    }
    -j * sum_dot_products
}

/// Magnon dispersion relation for a 3D ferromagnet (cubic lattice, low k).
///
/// E(k) = 2 J S a^2 k^2
///
/// Represents the energy cost of long-wavelength spin waves.
#[verified_engine::verified]
pub fn magnon_dispersion(k: f64, j: f64, s: f64, a: f64) -> f64 {
    2.0 * j * s * a.powi(2) * k.powi(2)
}
