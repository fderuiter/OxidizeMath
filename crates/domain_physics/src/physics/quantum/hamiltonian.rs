use crate::physics::quantum::types::{QuantumOperator, QuantumState};
use core::f64::consts::PI;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

#[inline]
#[verified_engine::verified]
fn kinetic_couplings(dx: f64, mass: f64, h_bar: f64) -> (f64, f64) {
    let denom = mass * dx * dx;
    if dx > 0.0
        && dx.is_finite()
        && mass > 0.0
        && denom > 0.0
        && denom.is_finite()
        && h_bar.is_finite()
    {
        let diag_kin = (h_bar * h_bar) / denom;
        (diag_kin, -0.5 * diag_kin)
    } else {
        (f64::NAN, f64::NAN)
    }
}

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
    let mut out = QuantumOperator::new(DMatrix::<Complex<f64>>::zeros(n_points, n_points));
    construct_1d_hamiltonian_into(potential, dx, mass, h_bar, &mut out);
    out
}

/// Updates an existing [`QuantumOperator`] in place with the 1D finite-difference Hamiltonian
/// $\hat{H} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x)$, performing zero heap allocations
/// when `out.matrix` already matches `potential.len()`.
#[verified_engine::verified]
pub fn construct_1d_hamiltonian_into(
    potential: &DVector<f64>,
    dx: f64,
    mass: f64,
    h_bar: f64,
    out: &mut QuantumOperator,
) {
    let n_points = potential.len();
    if out.matrix.nrows() != n_points || out.matrix.ncols() != n_points {
        out.matrix = DMatrix::<Complex<f64>>::zeros(n_points, n_points);
    } else {
        out.matrix.fill(Complex::new(0.0, 0.0));
    }
    let (diag_kin, off_diag_kin) = kinetic_couplings(dx, mass, h_bar);

    for i in 0..n_points {
        out.matrix[(i, i)] = Complex::new(diag_kin + potential[i], 0.0);
        if i > 0 {
            out.matrix[(i, i - 1)] = Complex::new(off_diag_kin, 0.0);
        }
        if i + 1 < n_points {
            out.matrix[(i, i + 1)] = Complex::new(off_diag_kin, 0.0);
        }
    }
}

/// Writes a $2 \times 2$ Hamiltonian in the two-eigenstate subspace $\{|\psi_1\rangle, |\psi_2\rangle\}$
/// into `out` with zero heap allocations when `out.matrix` is already $2 \times 2$.
#[verified_engine::verified]
pub fn construct_two_state_eigen_hamiltonian_into(
    e1: f64,
    e2: f64,
    coupling: f64,
    out: &mut QuantumOperator,
) {
    if out.matrix.nrows() != 2 || out.matrix.ncols() != 2 {
        out.matrix = DMatrix::<Complex<f64>>::zeros(2, 2);
    }
    out.matrix[(0, 0)] = Complex::new(e1, 0.0);
    out.matrix[(0, 1)] = Complex::new(coupling, 0.0);
    out.matrix[(1, 0)] = Complex::new(coupling, 0.0);
    out.matrix[(1, 1)] = Complex::new(e2, 0.0);
}

/// Constructs a $2 \times 2$ Hamiltonian in the two-eigenstate subspace $\{|\psi_1\rangle, |\psi_2\rangle\}$.
///
/// Diagonal elements store the unperturbed eigenenergies $(E_1, E_2)$, while `coupling` adds a
/// real symmetric off-diagonal dipole interaction $V_{12} = V_{21}$ (e.g. $0.0$ for free phase beating).
#[verified_engine::verified]
pub fn construct_two_state_eigen_hamiltonian(e1: f64, e2: f64, coupling: f64) -> QuantumOperator {
    let mut out = QuantumOperator::new(DMatrix::<Complex<f64>>::zeros(2, 2));
    construct_two_state_eigen_hamiltonian_into(e1, e2, coupling, &mut out);
    out
}

/// Populates `out` in place with a normalized Gaussian wavepacket
/// $\psi(x) \propto \exp\!\left(-\frac{(x - x_0)^2}{2\sigma^2}\right) e^{i k_0 x}$,
/// performing zero heap allocations when `out.vector.len() == x_grid.len()`.
#[verified_engine::verified]
pub fn gaussian_wavepacket_into(
    x_grid: &[f64],
    x0: f64,
    k0: f64,
    sigma: f64,
    out: &mut QuantumState,
) {
    let n_points = x_grid.len();
    if out.vector.len() != n_points {
        out.vector = DVector::<Complex<f64>>::zeros(n_points);
    }
    if n_points == 0 {
        return;
    }
    let denom = 2.0 * sigma * sigma;
    if !sigma.is_finite()
        || sigma <= 0.0
        || denom <= 0.0
        || !denom.is_finite()
        || !x0.is_finite()
        || !k0.is_finite()
    {
        out.vector.fill(Complex::new(f64::NAN, f64::NAN));
        return;
    }
    let inv_two_sigma_sq = 1.0 / denom;
    let normalization = 1.0 / (sigma * PI.sqrt()).sqrt();
    for (dst, &x) in out.vector.as_mut_slice().iter_mut().zip(x_grid.iter()) {
        let dx_x0 = x - x0;
        let amp = normalization * (-(dx_x0 * dx_x0) * inv_two_sigma_sq).exp();
        let (sin_kx, cos_kx) = (k0 * x).sin_cos();
        *dst = Complex::new(amp * cos_kx, amp * sin_kx);
    }
    out.normalize_in_place();
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
    let mut out = QuantumState::new(DVector::<Complex<f64>>::zeros(x_grid.len()));
    gaussian_wavepacket_into(x_grid, x0, k0, sigma, &mut out);
    out
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
