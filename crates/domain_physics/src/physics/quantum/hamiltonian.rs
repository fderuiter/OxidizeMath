//! 1D finite-difference Hamiltonians, two-state subspace operators, Gaussian wavepackets,
//! potential sculpting strokes, preset profiles, and Born region probability partitioning.

use crate::physics::quantum::types::{QuantumOperator, QuantumState};
use core::f64::consts::PI;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

/// Minimum clamped potential energy permitted when sculpting a 1D quantum potential.
pub const MIN_SCULPT_POTENTIAL: f64 = -200.0;

/// Maximum clamped potential energy permitted when sculpting a 1D quantum potential.
pub const MAX_SCULPT_POTENTIAL: f64 = 200.0;

/// Canonical 1D spatial potential presets for wavepacket scattering and confinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PotentialPreset1D {
    /// Zero interior potential ($V(x) = 0$) inside hard Dirichlet box boundaries.
    #[default]
    FreeParticle,
    /// Quadratic harmonic confinement $V(x) = \frac{1}{2} x^2$.
    HarmonicTrap,
    /// Single centered rectangular tunneling barrier of height $V_0$ and width $w$.
    FiniteSquareBarrier,
    /// Symmetric double-barrier Fabry-Pérot resonant tunneling cavity.
    DoubleBarrierResonant,
    /// Smooth logistic potential step $V(x) = \frac{V_0}{1 + \exp(-x / s)}$.
    AsymmetricStep,
}

impl PotentialPreset1D {
    /// Maps a numeric UI parameter code (`0..=4`) to the corresponding [`PotentialPreset1D`].
    #[inline]
    #[must_use]
    pub fn from_code(code: f64) -> Self {
        if !code.is_finite() {
            return Self::FreeParticle;
        }
        match code.round() as i32 {
            1 => Self::HarmonicTrap,
            2 => Self::FiniteSquareBarrier,
            3 => Self::DoubleBarrierResonant,
            4 => Self::AsymmetricStep,
            _ => Self::FreeParticle,
        }
    }

    /// Numeric UI parameter code (`0.0..=4.0`) corresponding to `self`.
    #[inline]
    #[must_use]
    pub const fn code(self) -> f64 {
        match self {
            Self::FreeParticle => 0.0,
            Self::HarmonicTrap => 1.0,
            Self::FiniteSquareBarrier => 2.0,
            Self::DoubleBarrierResonant => 3.0,
            Self::AsymmetricStep => 4.0,
        }
    }
}

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

/// Applies a zero-allocation 1D discrete Bresenham linear interpolation stroke between
/// `(i0, v0)` and `(i1, v1)` across `potential`, clamping all written values to
/// `[MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL]`.
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::apply_bresenham_potential_stroke;
///
/// let mut v = [0.0_f64; 5];
/// apply_bresenham_potential_stroke(&mut v, 0, 0.0, 4, 40.0);
/// assert!((v[2] - 20.0).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn apply_bresenham_potential_stroke(
    potential: &mut [f64],
    i0: usize,
    v0: f64,
    i1: usize,
    v1: f64,
) {
    let len = potential.len();
    if len == 0 || !v0.is_finite() || !v1.is_finite() {
        return;
    }
    let last = len - 1;
    let a = i0.min(last);
    let b = i1.min(last);
    let clamped_v0 = v0.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    let clamped_v1 = v1.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    if a == b {
        potential[a] = clamped_v1;
        return;
    }
    let (start_idx, start_v, end_idx, end_v) = if a <= b {
        (a, clamped_v0, b, clamped_v1)
    } else {
        (b, clamped_v1, a, clamped_v0)
    };
    let span = (end_idx - start_idx) as f64;
    for (step, slot) in potential[start_idx..=end_idx].iter_mut().enumerate() {
        let t = (step as f64) / span;
        let interp = start_v + t * (end_v - start_v);
        *slot = interp.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    }
}

/// Populates `out_potential` in-place with a canonical [`PotentialPreset1D`] profile over `x_grid`
/// without heap allocations.
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::{PotentialPreset1D, build_1d_potential_preset};
///
/// let x = [-1.0, 0.0, 1.0];
/// let mut v = [0.0; 3];
/// build_1d_potential_preset(&x, PotentialPreset1D::FiniteSquareBarrier, 15.0, 0.5, &mut v);
/// assert_eq!(v, [0.0, 15.0, 0.0]);
/// ```
#[verified_engine::verified]
pub fn build_1d_potential_preset(
    x_grid: &[f64],
    preset: PotentialPreset1D,
    barrier_height: f64,
    barrier_width: f64,
    out_potential: &mut [f64],
) {
    let n = x_grid.len().min(out_potential.len());
    let v0 = if barrier_height.is_finite() {
        barrier_height.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL)
    } else {
        0.0
    };
    let width = if barrier_width.is_finite() {
        barrier_width.clamp(0.05, 4.0)
    } else {
        0.4
    };
    let half_w = 0.5 * width;
    for (&x, slot) in x_grid[..n].iter().zip(out_potential[..n].iter_mut()) {
        let raw = match preset {
            PotentialPreset1D::FreeParticle => 0.0,
            PotentialPreset1D::HarmonicTrap => 0.5 * x * x,
            PotentialPreset1D::FiniteSquareBarrier => {
                if x.abs() <= half_w {
                    v0
                } else {
                    0.0
                }
            }
            PotentialPreset1D::DoubleBarrierResonant => {
                let well_half = width;
                let outer = well_half + width;
                if x.abs() >= well_half && x.abs() <= outer {
                    v0
                } else {
                    0.0
                }
            }
            PotentialPreset1D::AsymmetricStep => {
                let softness = (0.25 * width).max(0.05);
                v0 / (1.0 + (-x / softness).exp())
            }
        };
        *slot = raw.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    }
}

/// Computes `(reflected_prob, transmitted_prob)` by partitioning $|\psi_i|^2$ across `split_index`
/// with zero heap allocations.
///
/// Indices `0..split_index` accumulate into `reflected_prob`; indices `split_index..len`
/// accumulate into `transmitted_prob`.
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::DVector;
/// use math_explorer::num_complex::Complex;
/// use math_explorer::physics::quantum::{QuantumState, partition_region_probabilities};
///
/// let psi = QuantumState::new(DVector::from_vec(vec![
///     Complex::new(0.6, 0.0),
///     Complex::new(0.0, 0.8),
/// ]));
/// let (p_ref, p_trans) = partition_region_probabilities(&psi, 1);
/// assert!((p_ref - 0.36).abs() < 1e-12);
/// assert!((p_trans - 0.64).abs() < 1e-12);
/// ```
#[verified_engine::verified]
#[must_use]
pub fn partition_region_probabilities(state: &QuantumState, split_index: usize) -> (f64, f64) {
    let amps = state.vector.as_slice();
    let split = split_index.min(amps.len());
    let mut reflected = 0.0;
    for c in &amps[..split] {
        let p = c.norm_sqr();
        if p.is_finite() {
            reflected += p;
        }
    }
    let mut transmitted = 0.0;
    for c in &amps[split..] {
        let p = c.norm_sqr();
        if p.is_finite() {
            transmitted += p;
        }
    }
    (reflected, transmitted)
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

    #[test]
    #[verified_engine::verified]
    fn test_bresenham_stroke_interpolates_reversed_and_clamps() {
        let mut v = [0.0_f64; 9];
        apply_bresenham_potential_stroke(&mut v, 6, 300.0, 2, -300.0);
        assert!((v[2] - MIN_SCULPT_POTENTIAL).abs() < 1e-12);
        assert!((v[4] - 0.0).abs() < 1e-12);
        assert!((v[6] - MAX_SCULPT_POTENTIAL).abs() < 1e-12);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[8], 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_presets_and_region_probability_conservation() {
        let x_grid: Vec<f64> = (0..101).map(|i| -5.0 + i as f64 * 0.1).collect();
        let mut pot = vec![0.0; x_grid.len()];
        for code in 0..=4 {
            let preset = PotentialPreset1D::from_code(code as f64);
            assert_eq!(preset.code() as i32, code);
            build_1d_potential_preset(&x_grid, preset, 18.0, 0.6, &mut pot);
            assert!(
                pot.iter()
                    .all(|&val| (MIN_SCULPT_POTENTIAL..=MAX_SCULPT_POTENTIAL).contains(&val))
            );
        }

        let psi = gaussian_wavepacket(&x_grid, -2.0, 5.0, 0.5);
        let (p_ref, p_trans) = partition_region_probabilities(&psi, 50);
        assert!((p_ref + p_trans - 1.0).abs() < 1e-12);
        assert!(p_ref > 0.99);
    }
}
