use crate::physics::quantum::types::{QuantumOperator, QuantumState};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::f64::consts::PI;

/// Constructs a 1D Hamiltonian operator for a particle in a potential.
///
/// The Hamiltonian is constructed using the finite difference method for the kinetic energy term.
/// H = - (h_bar^2 / 2m) * d^2/dx^2 + V(x)
///
/// # Arguments
/// * `potential` - A vector representing the potential energy V(x) at each grid point.
/// * `dx` - The spatial step size.
/// * `mass` - The mass of the particle.
/// * `h_bar` - The reduced Planck constant.
#[verified_engine::verified]
pub fn construct_1d_hamiltonian(
    potential: &DVector<f64>,
    dx: f64,
    mass: f64,
    h_bar: f64,
) -> QuantumOperator {
    let n_points = potential.len();
    let mut matrix = DMatrix::<Complex<f64>>::zeros(n_points, n_points);

    // Coefficients for the kinetic energy term
    // - (h_bar^2 / 2m) * (psi[i+1] - 2psi[i] + psi[i-1]) / dx^2
    // Diagonal term: - (h_bar^2 / 2m) * (-2 / dx^2) = h_bar^2 / (m * dx^2)
    // Off-diagonal term: - (h_bar^2 / 2m) * (1 / dx^2) = - h_bar^2 / (2m * dx^2)

    let h_bar_sq = h_bar * h_bar;
    let diag_kin = h_bar_sq / (mass * dx * dx);
    let off_diag_kin = -h_bar_sq / (2.0 * mass * dx * dx);

    for i in 0..n_points {
        // Kinetic Energy: Diagonal
        let v = potential[i];
        matrix[(i, i)] = Complex::new(diag_kin + v, 0.0);

        // Kinetic Energy: Off-diagonal
        if i > 0 {
            matrix[(i, i - 1)] = Complex::new(off_diag_kin, 0.0);
        }
        if i < n_points - 1 {
            matrix[(i, i + 1)] = Complex::new(off_diag_kin, 0.0);
        }
    }

    QuantumOperator::new(matrix)
}

/// Constructs a $2 \times 2$ Hamiltonian in the two-eigenstate subspace $\{|\psi_1\rangle, |\psi_2\rangle\}$.
///
/// Diagonal elements store the unperturbed eigenenergies $(E_1, E_2)$, while `coupling` adds a
/// real symmetric off-diagonal dipole interaction $V_{12} = V_{21}$ (e.g. $0.0$ for free phase beating).
#[verified_engine::verified]
pub fn construct_two_state_eigen_hamiltonian(e1: f64, e2: f64, coupling: f64) -> QuantumOperator {
    let matrix = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex::new(e1, 0.0),
            Complex::new(coupling, 0.0),
            Complex::new(coupling, 0.0),
            Complex::new(e2, 0.0),
        ],
    );
    QuantumOperator::new(matrix)
}

/// Creates a Gaussian wavepacket state.
///
/// psi(x) = A * exp(-(x - x0)^2 / (2 * sigma^2)) * exp(i * k0 * x)
///
/// # Arguments
/// * `x_grid` - The spatial grid points.
/// * `x0` - The initial position of the packet center.
/// * `k0` - The initial momentum (wave number).
/// * `sigma` - The width of the packet.
#[verified_engine::verified]
pub fn gaussian_wavepacket(x_grid: &[f64], x0: f64, k0: f64, sigma: f64) -> QuantumState {
    let n_points = x_grid.len();
    let mut psi_vec = DVector::<Complex<f64>>::zeros(n_points);

    // Normalization factor (approximate for now, will normalize the state at the end)
    let normalization = 1.0 / (sigma * (PI).sqrt()).sqrt();

    for (i, &x) in x_grid.iter().enumerate() {
        let gauss = (-((x - x0).powi(2)) / (2.0 * sigma * sigma)).exp();
        let plane_wave = Complex::new(0.0, k0 * x).exp();
        psi_vec[i] = Complex::new(normalization * gauss, 0.0) * plane_wave;
    }

    QuantumState::new(psi_vec).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_construct_1d_hamiltonian_structure() {
        let n = 5;
        let potential = DVector::from_element(n, 0.0); // Free particle
        let dx = 1.0;
        let mass = 1.0;
        let h_bar = 1.0;

        let hamiltonian = construct_1d_hamiltonian(&potential, dx, mass, h_bar);
        let mat = &hamiltonian.matrix;

        // Check dimensions
        assert_eq!(mat.nrows(), n);
        assert_eq!(mat.ncols(), n);

        // Check tridiagonal structure
        // Diagonal: h_bar^2 / (m * dx^2) = 1.0
        // Off-diagonal: - h_bar^2 / (2m * dx^2) = -0.5

        for i in 0..n {
            assert!((mat[(i, i)].re - 1.0).abs() < math_commons::registry::TOLERANCE_HIGH);
            if i > 0 {
                assert!((mat[(i, i - 1)].re + 0.5).abs() < math_commons::registry::TOLERANCE_HIGH);
            }
            if i < n - 1 {
                assert!((mat[(i, i + 1)].re + 0.5).abs() < math_commons::registry::TOLERANCE_HIGH);
            }
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_gaussian_wavepacket_normalization() {
        let x_grid: Vec<f64> = (0..100).map(|i| i as f64 * 0.1 - 5.0).collect();
        let psi = gaussian_wavepacket(&x_grid, 0.0, 1.0, 1.0);
        assert!((psi.norm() - 1.0).abs() < math_commons::registry::TOLERANCE_HIGH);
    }
}
