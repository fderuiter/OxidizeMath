//! 1D quantum Hamiltonian construction, Gaussian wavepacket initialization,
//! zero-allocation Bresenham potential sculpting, canonical 1D potential presets,
//! and barrier transmission/reflection probability partitioning.

// [cite:quantum_mechanics]
use crate::physics::quantum::types::{QuantumOperator, QuantumState};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::f64::consts::PI;

/// Minimum allowed sculpted 1D potential energy $V(x)$ in natural units.
pub const MIN_SCULPT_POTENTIAL: f64 = -20.0;

/// Maximum allowed sculpted 1D potential energy $V(x)$ in natural units.
pub const MAX_SCULPT_POTENTIAL: f64 = 40.0;

/// Canonical 1D potential energy profiles $V(x)$ for Schrödinger wavepacket evolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PotentialPreset1D {
    /// Flat interior potential $V(x) = 0$ bounded by Dirichlet domain walls.
    InfiniteWell,
    /// Quadratic harmonic trapping potential $V(x) = \frac{1}{2} x^2$.
    HarmonicOscillator,
    /// Rectangular quantum tunneling barrier centered at $x = 0$.
    TunnelingBarrier,
    /// Symmetric dual-barrier Fabry-Pérot / double-well cavity.
    ResonantDoubleWell,
    /// Linear electric-field Stark potential ramp $V(x) = 3 x$.
    StarkLinearRamp,
}

impl PotentialPreset1D {
    /// Maps a discrete numeric selector (`0.0..=4.0`) to a [`PotentialPreset1D`].
    #[must_use]
    #[verified_engine::verified]
    pub fn from_code(code: f64) -> Self {
        if !code.is_finite() || code < 0.5 {
            Self::InfiniteWell
        } else if code < 1.5 {
            Self::HarmonicOscillator
        } else if code < 2.5 {
            Self::TunnelingBarrier
        } else if code < 3.5 {
            Self::ResonantDoubleWell
        } else {
            Self::StarkLinearRamp
        }
    }

    /// Returns the canonical numeric selector value for this preset.
    #[must_use]
    #[verified_engine::verified]
    pub fn code(self) -> f64 {
        match self {
            Self::InfiniteWell => 0.0,
            Self::HarmonicOscillator => 1.0,
            Self::TunnelingBarrier => 2.0,
            Self::ResonantDoubleWell => 3.0,
            Self::StarkLinearRamp => 4.0,
        }
    }
}

/// Constructs a 1D Hamiltonian operator for a particle in a potential.
///
/// The Hamiltonian is constructed using the finite difference method for the kinetic energy term:
/// $\hat{H} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x)$.
///
/// # Arguments
/// * `potential` - A vector representing the potential energy $V(x)$ at each grid point.
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

    let h_bar_sq = h_bar * h_bar;
    let diag_kin = h_bar_sq / (mass * dx * dx);
    let off_diag_kin = -h_bar_sq / (2.0 * mass * dx * dx);

    for i in 0..n_points {
        let v = potential[i];
        matrix[(i, i)] = Complex::new(diag_kin + v, 0.0);

        if i > 0 {
            matrix[(i, i - 1)] = Complex::new(off_diag_kin, 0.0);
        }
        if i < n_points - 1 {
            matrix[(i, i + 1)] = Complex::new(off_diag_kin, 0.0);
        }
    }

    QuantumOperator::new(matrix)
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
        *out = construct_1d_hamiltonian(potential, dx, mass, h_bar);
        return;
    }
    out.matrix.fill(Complex::new(0.0, 0.0));
    let h_bar_sq = h_bar * h_bar;
    let diag_kin = h_bar_sq / (mass * dx * dx);
    let off_diag_kin = -h_bar_sq / (2.0 * mass * dx * dx);

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

/// Creates a normalized 1D Gaussian wavepacket state
/// $\psi(x) = A \exp\left(-\frac{(x - x_0)^2}{2\sigma^2}\right)\exp(i k_0 x)$.
///
/// # Arguments
/// * `x_grid` - The spatial grid points.
/// * `x0` - The initial position of the packet center.
/// * `k0` - The initial momentum (wave number).
/// * `sigma` - The spatial width of the packet (clamped away from zero).
#[verified_engine::verified]
pub fn gaussian_wavepacket(x_grid: &[f64], x0: f64, k0: f64, sigma: f64) -> QuantumState {
    let n_points = x_grid.len();
    let mut psi_vec = DVector::<Complex<f64>>::zeros(n_points);
    let safe_sigma = if sigma.is_finite() {
        sigma.abs().max(1e-6)
    } else {
        0.5
    };
    let safe_x0 = if x0.is_finite() { x0 } else { 0.0 };
    let safe_k0 = if k0.is_finite() { k0 } else { 0.0 };
    let normalization = 1.0 / (safe_sigma * PI.sqrt()).sqrt();

    for (i, &x) in x_grid.iter().enumerate() {
        let gauss = (-((x - safe_x0).powi(2)) / (2.0 * safe_sigma * safe_sigma)).exp();
        let plane_wave = Complex::new(0.0, safe_k0 * x).exp();
        psi_vec[i] = Complex::new(normalization * gauss, 0.0) * plane_wave;
    }

    QuantumState::new(psi_vec).normalize()
}

/// Interpolates a 1D potential stroke from `(i0, v0)` to `(i1, v1)` in place with zero heap
/// allocations, clamping grid indices to `0..potential.len()` and $V(x)$ to `[-20.0, 40.0]`.
///
/// Non-finite `v0` or `v1` values or empty potential vectors are ignored without mutation.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::DVector;
/// use math_explorer::physics::quantum::apply_bresenham_potential_stroke;
///
/// let mut v = DVector::zeros(200);
/// apply_bresenham_potential_stroke(&mut v, 95, 10.0, 105, 20.0);
/// assert!((v[95] - 10.0).abs() < 1e-12);
/// assert!((v[100] - 15.0).abs() < 1e-12);
/// assert!((v[105] - 20.0).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn apply_bresenham_potential_stroke(
    potential: &mut DVector<f64>,
    i0: usize,
    v0: f64,
    i1: usize,
    v1: f64,
) {
    if potential.is_empty() || !v0.is_finite() || !v1.is_finite() {
        return;
    }
    let last = potential.len() - 1;
    let start_idx = i0.min(last);
    let end_idx = i1.min(last);
    let start_v = v0.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    let end_v = v1.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    let steps = start_idx.abs_diff(end_idx);

    if steps == 0 {
        potential[start_idx] = end_v;
        return;
    }

    for step in 0..=steps {
        let idx = if end_idx >= start_idx {
            start_idx + step
        } else {
            start_idx - step
        };
        let t = step as f64 / steps as f64;
        let interpolated = start_v + (end_v - start_v) * t;
        potential[idx] = interpolated.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    }
}

/// Evaluates a canonical [`PotentialPreset1D`] profile over `x_grid`, clamped to
/// `[MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL]`.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::{PotentialPreset1D, build_1d_potential_preset};
///
/// let x_grid: Vec<f64> = (0..200).map(|i| -5.0 + i as f64 * (10.0 / 199.0)).collect();
/// let barrier = build_1d_potential_preset(PotentialPreset1D::TunnelingBarrier, &x_grid);
/// assert!(barrier[100] > 10.0);
/// assert_eq!(barrier[0], 0.0);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn build_1d_potential_preset(preset: PotentialPreset1D, x_grid: &[f64]) -> DVector<f64> {
    let mut potential = DVector::zeros(x_grid.len());
    for (i, &raw_x) in x_grid.iter().enumerate() {
        let x = if raw_x.is_finite() { raw_x } else { 0.0 };
        let v = match preset {
            PotentialPreset1D::InfiniteWell => 0.0,
            PotentialPreset1D::HarmonicOscillator => 0.5 * x * x,
            PotentialPreset1D::TunnelingBarrier => {
                if x.abs() <= 0.45 {
                    14.0
                } else {
                    0.0
                }
            }
            PotentialPreset1D::ResonantDoubleWell => {
                if (x + 0.85).abs() <= 0.28 || (x - 0.85).abs() <= 0.28 {
                    14.0
                } else {
                    0.0
                }
            }
            PotentialPreset1D::StarkLinearRamp => 3.0 * x,
        };
        potential[i] = v.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
    }
    potential
}

/// Computes transmitted and reflected region probabilities `(p_transmitted, p_reflected)` around
/// `barrier_idx` with zero heap allocations:
/// $P_T = \sum_{i > i_b} |\psi_i|^2 \in [0, 1]$ and $P_R = 1 - P_T$.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::{gaussian_wavepacket, partition_region_probabilities};
///
/// let x_grid: Vec<f64> = (0..200).map(|i| -5.0 + i as f64 * (10.0 / 199.0)).collect();
/// let psi = gaussian_wavepacket(&x_grid, -2.5, 5.0, 0.4);
/// let (p_t, p_r) = partition_region_probabilities(&psi, 100);
/// assert!(p_t < 1e-3);
/// assert!((p_t + p_r - 1.0).abs() < 1e-12);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn partition_region_probabilities(state: &QuantumState, barrier_idx: usize) -> (f64, f64) {
    let n = state.vector.len();
    if n == 0 || barrier_idx >= n - 1 {
        return (0.0, 1.0);
    }
    let mut p_transmitted = 0.0;
    for c in state.vector.iter().skip(barrier_idx + 1) {
        if c.re.is_finite() && c.im.is_finite() {
            p_transmitted += c.norm_sqr();
        }
    }
    let clamped_pt = if p_transmitted.is_finite() {
        p_transmitted.clamp(0.0, 1.0)
    } else {
        0.0
    };
    (clamped_pt, 1.0 - clamped_pt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_construct_1d_hamiltonian_structure() {
        let n = 5;
        let potential = DVector::from_element(n, 0.0);
        let hamiltonian = construct_1d_hamiltonian(&potential, 1.0, 1.0, 1.0);
        let mat = &hamiltonian.matrix;

        assert_eq!(mat.nrows(), n);
        assert_eq!(mat.ncols(), n);

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
    fn test_gaussian_wavepacket_normalization_and_zero_sigma_guard() {
        let x_grid: Vec<f64> = (0..100).map(|i| i as f64 * 0.1 - 5.0).collect();
        let psi = gaussian_wavepacket(&x_grid, 0.0, 1.0, 1.0);
        assert!((psi.norm() - 1.0).abs() < math_commons::registry::TOLERANCE_HIGH);

        let degenerate = gaussian_wavepacket(&x_grid, 0.0, 1.0, 0.0);
        assert!(
            degenerate
                .vector
                .iter()
                .all(|c| c.re.is_finite() && c.im.is_finite())
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_bresenham_potential_stroke_continuity_reverse_and_clamping() {
        let mut v = DVector::zeros(200);
        apply_bresenham_potential_stroke(&mut v, 20, -30.0, 30, 50.0);
        assert!((v[20] - MIN_SCULPT_POTENTIAL).abs() < 1e-12);
        assert!((v[25] - 10.0).abs() < 1e-12);
        assert!((v[30] - MAX_SCULPT_POTENTIAL).abs() < 1e-12);
        for idx in 20..30 {
            assert!(v[idx + 1] > v[idx], "Stroke must interpolate monotonically");
        }

        // Reverse stroke direction and out-of-bounds index clamping
        apply_bresenham_potential_stroke(&mut v, 250, 8.0, 195, 4.0);
        assert!((v[199] - 8.0).abs() < 1e-12);
        assert!((v[195] - 4.0).abs() < 1e-12);

        // Non-finite stroke inputs are rejected without corrupting state
        apply_bresenham_potential_stroke(&mut v, 10, f64::NAN, 15, 5.0);
        assert_eq!(v[10], 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_potential_presets_and_probability_partition_conservation() {
        let x_grid: Vec<f64> = (0..200).map(|i| -5.0 + i as f64 * (10.0 / 199.0)).collect();
        for code in [0.0, 1.0, 2.0, 3.0, 4.0] {
            let preset = PotentialPreset1D::from_code(code);
            assert!((preset.code() - code).abs() < 1e-12);
            let pot = build_1d_potential_preset(preset, &x_grid);
            assert_eq!(pot.len(), 200);
            assert!(pot.iter().all(|&val| val.is_finite()
                && (MIN_SCULPT_POTENTIAL..=MAX_SCULPT_POTENTIAL).contains(&val)));
        }

        let left_psi = gaussian_wavepacket(&x_grid, -2.5, 5.0, 0.4);
        let right_psi = gaussian_wavepacket(&x_grid, 2.5, 5.0, 0.4);
        let (pt_left, pr_left) = partition_region_probabilities(&left_psi, 100);
        let (pt_right, pr_right) = partition_region_probabilities(&right_psi, 100);
        assert!(pt_left < 1e-3 && pr_left > 0.999);
        assert!(pt_right > 0.999 && pr_right < 1e-3);
        assert!((pt_left + pr_left - 1.0).abs() < 1e-15);
        assert!((pt_right + pr_right - 1.0).abs() < 1e-15);
    }
}
