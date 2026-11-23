// [cite:quantum_mechanics]
//! # Tridiagonal 1D Quantum Hamiltonian & Crank-Nicolson Propagator
//!
//! Provides $O(N)$-memory tridiagonal representations of 1D finite-difference Schrödinger
//! Hamiltonians and a zero-allocation, strictly unitary Crank-Nicolson (Cayley transform)
//! propagator solved in linear $O(N)$ time via preallocated Thomas sweeps:
//!
//! $$ \left(I + \frac{i\Delta t}{2\hbar}\hat{H}\right)|\psi(t + \Delta t)\rangle
//!    = \left(I - \frac{i\Delta t}{2\hbar}\hat{H}\right)|\psi(t)\rangle $$

use crate::physics::quantum::types::QuantumState;
use nalgebra::DVector;
use num_complex::Complex;

/// Compact $O(N)$ representation of a 1D real symmetric tridiagonal Hamiltonian
/// $\hat{H} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x)$.
///
/// Stores the $N$ main-diagonal entries $d_i$ and $N - 1$ symmetric off-diagonal couplings
/// $e_i = H_{i, i+1} = H_{i+1, i}$ in preallocated contiguous buffers so potential sculpting
/// and parameter sweeps run without dynamic heap allocations when $N \le \text{capacity}$.
#[derive(Clone, Debug)]
pub struct TridiagonalHamiltonian1D {
    diagonal: Vec<f64>,
    off_diagonal: Vec<f64>,
    n_points: usize,
}

impl TridiagonalHamiltonian1D {
    /// Preallocates tridiagonal storage for grids up to `max_points` with initial length `0`.
    #[verified_engine::verified]
    #[must_use]
    pub fn with_capacity(max_points: usize) -> Self {
        Self {
            diagonal: vec![0.0; max_points],
            off_diagonal: vec![0.0; max_points.saturating_sub(1)],
            n_points: 0,
        }
    }

    /// Creates a [`TridiagonalHamiltonian1D`] directly from explicit `diagonal` and `off_diagonal` vectors.
    ///
    /// # Panics
    /// Panics if `off_diagonal.len() != diagonal.len().saturating_sub(1)`.
    #[verified_engine::verified]
    #[must_use]
    pub fn new(diagonal: Vec<f64>, off_diagonal: Vec<f64>) -> Self {
        let n_points = diagonal.len();
        assert_eq!(
            off_diagonal.len(),
            n_points.saturating_sub(1),
            "Off-diagonal length must equal diagonal.len().saturating_sub(1)"
        );
        Self {
            diagonal,
            off_diagonal,
            n_points,
        }
    }

    /// Constructs the 1D finite-difference Hamiltonian from a [`DVector`] potential $V(x_i)$.
    #[verified_engine::verified]
    #[must_use]
    pub fn from_potential(potential: &DVector<f64>, dx: f64, mass: f64, h_bar: f64) -> Self {
        Self::from_potential_slice(potential.as_slice(), dx, mass, h_bar)
    }

    /// Constructs the 1D finite-difference Hamiltonian from a slice of potential samples $V(x_i)$.
    #[verified_engine::verified]
    #[must_use]
    pub fn from_potential_slice(potential: &[f64], dx: f64, mass: f64, h_bar: f64) -> Self {
        let mut ham = Self::with_capacity(potential.len());
        ham.update_potential_in_place(potential, dx, mass, h_bar);
        ham
    }

    /// Updates this Hamiltonian in place from a [`DVector`] potential without heap allocation
    /// when `potential.len()` does not exceed preallocated capacity.
    #[verified_engine::verified]
    pub fn update_from_potential(
        &mut self,
        potential: &DVector<f64>,
        dx: f64,
        mass: f64,
        h_bar: f64,
    ) {
        self.update_potential_in_place(potential.as_slice(), dx, mass, h_bar);
    }

    /// Updates this Hamiltonian in place from a potential slice $V(x_i)$, performing zero heap
    /// allocations whenever `potential.len()` fits within the existing buffer capacity.
    #[verified_engine::verified]
    pub fn update_potential_in_place(&mut self, potential: &[f64], dx: f64, mass: f64, h_bar: f64) {
        let n = potential.len();
        if n > self.diagonal.len() {
            self.diagonal.resize(n, 0.0);
        }
        let n_off = n.saturating_sub(1);
        if n_off > self.off_diagonal.len() {
            self.off_diagonal.resize(n_off, 0.0);
        }
        self.n_points = n;
        if n == 0 {
            return;
        }
        let h_bar_sq = h_bar * h_bar;
        let denom = mass * dx * dx;
        let diag_kin = h_bar_sq / denom;
        let off_diag_kin = -0.5 * diag_kin;
        for (dst, &v) in self.diagonal[..n].iter_mut().zip(potential.iter()) {
            *dst = diag_kin + v;
        }
        for dst in &mut self.off_diagonal[..n_off] {
            *dst = off_diag_kin;
        }
    }

    /// Number of active spatial grid points $N$.
    #[verified_engine::verified]
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_points
    }

    /// Returns `true` when the active grid contains zero points.
    #[verified_engine::verified]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_points == 0
    }

    /// Active main-diagonal entries $d_0, \dots, d_{N-1}$.
    #[verified_engine::verified]
    #[must_use]
    pub fn diagonal(&self) -> &[f64] {
        &self.diagonal[..self.n_points]
    }

    /// Active symmetric off-diagonal couplings $e_0, \dots, e_{N-2}$.
    #[verified_engine::verified]
    #[must_use]
    pub fn off_diagonal(&self) -> &[f64] {
        &self.off_diagonal[..self.n_points.saturating_sub(1)]
    }

    /// Computes the real energy expectation value $\langle\psi|\hat{H}|\psi\rangle$ with zero heap allocations.
    ///
    /// # Panics
    /// Panics if `state.vector.len() != self.len()`.
    #[verified_engine::verified]
    #[must_use]
    pub fn expectation_value(&self, state: &QuantumState) -> f64 {
        let psi = state.vector.as_slice();
        assert_eq!(
            psi.len(),
            self.n_points,
            "QuantumState length must match TridiagonalHamiltonian1D length"
        );
        let mut energy = 0.0;
        for (&d, amp) in self.diagonal().iter().zip(psi.iter()) {
            energy += d * amp.norm_sqr();
        }
        for (i, &e) in self.off_diagonal().iter().enumerate() {
            let cross = psi[i].conj() * psi[i + 1];
            energy += 2.0 * e * cross.re;
        }
        energy
    }
}

/// Zero-allocation 1D unitary Crank-Nicolson propagator solving
/// $(I + i\alpha\hat{H})\psi^{n+1} = (I - i\alpha\hat{H})\psi^n$ with $\alpha = \frac{\Delta t}{2\hbar}$.
///
/// Because $\hat{H}$ is real symmetric, every Thomas elimination pivot satisfies $\operatorname{Re}(\tilde{a}_k) \ge 1$,
/// guaranteeing unconditional numerical stability without pivoting and exact discrete norm conservation.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::{
///     CrankNicolsonPropagator1D, TridiagonalHamiltonian1D, gaussian_wavepacket,
/// };
/// use math_explorer::nalgebra::DVector;
///
/// let n = 64;
/// let dx = 0.1;
/// let x_grid: Vec<f64> = (0..n).map(|i| -3.2 + i as f64 * dx).collect();
/// let potential = DVector::from_fn(n, |i, _| 0.5 * x_grid[i] * x_grid[i]);
/// let hamiltonian = TridiagonalHamiltonian1D::from_potential(&potential, dx, 1.0, 1.0);
/// let mut propagator = CrankNicolsonPropagator1D::new(&hamiltonian, 0.01, 1.0);
///
/// let mut psi = gaussian_wavepacket(&x_grid, -0.8, 2.5, 0.5);
/// propagator.evolve_substeps_in_place(&mut psi, 10);
/// assert!((psi.norm() - 1.0).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct CrankNicolsonPropagator1D {
    inv_denom: Vec<Complex<f64>>,
    c_prime: Vec<Complex<f64>>,
    rhs_diag: Vec<Complex<f64>>,
    lhs_off: Vec<Complex<f64>>,
    scratch_rhs: Vec<Complex<f64>>,
    n_points: usize,
    configured_dt: Option<f64>,
    configured_h_bar: Option<f64>,
}

impl CrankNicolsonPropagator1D {
    /// Preallocates Thomas sweep workspace buffers for grids up to `max_points`.
    #[verified_engine::verified]
    #[must_use]
    pub fn with_capacity(max_points: usize) -> Self {
        let zero = Complex::new(0.0, 0.0);
        let off_cap = max_points.saturating_sub(1);
        Self {
            inv_denom: vec![zero; max_points],
            c_prime: vec![zero; off_cap],
            rhs_diag: vec![zero; max_points],
            lhs_off: vec![zero; off_cap],
            scratch_rhs: vec![zero; max_points],
            n_points: 0,
            configured_dt: None,
            configured_h_bar: None,
        }
    }

    /// Preallocates and factors a [`CrankNicolsonPropagator1D`] for `(hamiltonian, dt, h_bar)`.
    #[verified_engine::verified]
    #[must_use]
    pub fn new(hamiltonian: &TridiagonalHamiltonian1D, dt: f64, h_bar: f64) -> Self {
        let mut prop = Self::with_capacity(hamiltonian.len());
        prop.configure_dt(hamiltonian, dt, h_bar);
        prop
    }

    /// Currently configured step size $\Delta t$, or `None` if invalidated or not yet configured.
    #[verified_engine::verified]
    #[must_use]
    pub fn dt(&self) -> Option<f64> {
        self.configured_dt
    }

    /// Active grid point count $N$ for the current factorization.
    #[verified_engine::verified]
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_points
    }

    /// Returns `true` when configured for an empty (`0`-point) grid.
    #[verified_engine::verified]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_points == 0
    }

    /// Marks the cached factorization stale so callers know to re-run [`Self::configure_dt`].
    #[verified_engine::verified]
    pub fn invalidate(&mut self) {
        self.configured_dt = None;
        self.configured_h_bar = None;
    }

    fn ensure_buffer_capacity(&mut self, n: usize) {
        let zero = Complex::new(0.0, 0.0);
        if n > self.inv_denom.len() {
            self.inv_denom.resize(n, zero);
            self.rhs_diag.resize(n, zero);
            self.scratch_rhs.resize(n, zero);
        }
        let n_off = n.saturating_sub(1);
        if n_off > self.c_prime.len() {
            self.c_prime.resize(n_off, zero);
            self.lhs_off.resize(n_off, zero);
        }
    }

    fn poison_non_finite(&mut self, n: usize) {
        let nan = Complex::new(f64::NAN, f64::NAN);
        let zero = Complex::new(0.0, 0.0);
        self.inv_denom[..n].fill(nan);
        self.rhs_diag[..n].fill(nan);
        let n_off = n.saturating_sub(1);
        self.c_prime[..n_off].fill(zero);
        self.lhs_off[..n_off].fill(zero);
    }

    /// Precomputes the $O(N)$ Thomas LU multipliers for `(hamiltonian, dt, h_bar)` with zero
    /// heap allocations whenever `hamiltonian.len()` fits within preallocated capacity.
    #[verified_engine::verified]
    pub fn configure_dt(&mut self, hamiltonian: &TridiagonalHamiltonian1D, dt: f64, h_bar: f64) {
        let n = hamiltonian.len();
        self.ensure_buffer_capacity(n);
        self.n_points = n;
        self.configured_dt = Some(dt);
        self.configured_h_bar = Some(h_bar);
        if n == 0 {
            return;
        }
        let diag = hamiltonian.diagonal();
        let off = hamiltonian.off_diagonal();
        let alpha = dt / (2.0 * h_bar);
        if !dt.is_finite()
            || dt <= 0.0
            || !h_bar.is_finite()
            || h_bar <= 0.0
            || !alpha.is_finite()
            || diag.iter().any(|v| !v.is_finite())
            || off.iter().any(|v| !v.is_finite())
        {
            self.poison_non_finite(n);
            return;
        }
        self.factor_cayley_system(diag, off, alpha);
    }

    fn factor_cayley_system(&mut self, diag: &[f64], off: &[f64], alpha: f64) {
        let n = diag.len();
        let one = Complex::new(1.0, 0.0);
        for (i, &d) in diag.iter().enumerate() {
            self.rhs_diag[i] = Complex::new(1.0, -alpha * d);
        }
        for (i, &e) in off.iter().enumerate() {
            self.lhs_off[i] = Complex::new(0.0, alpha * e);
        }
        let inv0 = one / Complex::new(1.0, alpha * diag[0]);
        self.inv_denom[0] = inv0;
        if n > 1 {
            self.c_prime[0] = self.lhs_off[0] * inv0;
        }
        for (i, &d) in diag.iter().enumerate().skip(1) {
            let l_prev = self.lhs_off[i - 1];
            let denom = Complex::new(1.0, alpha * d) - l_prev * self.c_prime[i - 1];
            let inv_d = one / denom;
            self.inv_denom[i] = inv_d;
            if i + 1 < n {
                self.c_prime[i] = self.lhs_off[i] * inv_d;
            }
        }
    }

    /// Advances `state` in place by one Crank-Nicolson step $\Delta t$ with zero heap allocations.
    ///
    /// # Panics
    /// Panics if `state.vector.len() != self.len()`.
    #[verified_engine::verified]
    pub fn evolve_in_place(&mut self, state: &mut QuantumState) {
        let n = self.n_points;
        assert_eq!(
            state.vector.len(),
            n,
            "QuantumState length must match CrankNicolsonPropagator1D length"
        );
        if n == 0 {
            return;
        }
        let psi = state.vector.as_mut_slice();
        if n == 1 {
            psi[0] = (self.rhs_diag[0] * psi[0]) * self.inv_denom[0];
            return;
        }
        self.sweep_tridiagonal(psi);
    }

    fn sweep_tridiagonal(&mut self, psi: &mut [Complex<f64>]) {
        let n = psi.len();
        let r0 = self.rhs_diag[0] * psi[0] - self.lhs_off[0] * psi[1];
        let mut d_prev = r0 * self.inv_denom[0];
        self.scratch_rhs[0] = d_prev;

        for i in 1..(n - 1) {
            let l_prev = self.lhs_off[i - 1];
            let r_i =
                -l_prev * psi[i - 1] + self.rhs_diag[i] * psi[i] - self.lhs_off[i] * psi[i + 1];
            d_prev = (r_i - l_prev * d_prev) * self.inv_denom[i];
            self.scratch_rhs[i] = d_prev;
        }

        let l_last = self.lhs_off[n - 2];
        let r_last = -l_last * psi[n - 2] + self.rhs_diag[n - 1] * psi[n - 1];
        let mut next_psi = (r_last - l_last * d_prev) * self.inv_denom[n - 1];
        self.scratch_rhs[n - 1] = next_psi;
        psi[n - 1] = next_psi;

        for i in (0..(n - 1)).rev() {
            next_psi = self.scratch_rhs[i] - self.c_prime[i] * next_psi;
            psi[i] = next_psi;
        }
    }

    /// Advances `state` in place by `substeps` consecutive Crank-Nicolson sweeps with zero heap allocations.
    #[verified_engine::verified]
    pub fn evolve_substeps_in_place(&mut self, state: &mut QuantumState, substeps: usize) {
        for _ in 0..substeps {
            self.evolve_in_place(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::quantum::hamiltonian::{construct_1d_hamiltonian, gaussian_wavepacket};
    use crate::physics::quantum::schrodinger::evolve_state;

    #[test]
    #[verified_engine::verified]
    fn test_crank_nicolson_unitarity_and_dense_parity() {
        let n = 48;
        let dx = 0.15;
        let x_grid: Vec<f64> = (0..n).map(|i| -3.5 + i as f64 * dx).collect();
        let potential = DVector::from_fn(n, |i, _| 0.5 * x_grid[i] * x_grid[i]);
        let tri_h = TridiagonalHamiltonian1D::from_potential(&potential, dx, 1.0, 1.0);
        let dense_h = construct_1d_hamiltonian(&potential, dx, 1.0, 1.0);

        let psi0 = gaussian_wavepacket(&x_grid, -0.8, 2.2, 0.55);
        let e_tri = tri_h.expectation_value(&psi0);
        let e_dense = dense_h.expectation_value(&psi0).re;
        assert!((e_tri - e_dense).abs() < 1e-12);

        let dt = 0.02;
        let substeps = 4;
        let mut cn = CrankNicolsonPropagator1D::new(&tri_h, dt / substeps as f64, 1.0);
        let mut psi_cn = psi0.clone();
        for _ in 0..10 {
            cn.evolve_substeps_in_place(&mut psi_cn, substeps);
        }
        assert!((psi_cn.norm() - 1.0).abs() < 1e-13);

        let psi_exact = evolve_state(&psi0, &dense_h, 10.0 * dt, 1.0);
        assert!((&psi_cn.vector - &psi_exact.vector).norm() < 5e-3);
    }

    #[test]
    #[verified_engine::verified]
    fn test_empty_single_point_and_invalid_dt_edge_cases() {
        let empty_v = DVector::<f64>::zeros(0);
        let empty_h = TridiagonalHamiltonian1D::from_potential(&empty_v, 0.1, 1.0, 1.0);
        assert!(empty_h.is_empty());
        let mut empty_prop = CrankNicolsonPropagator1D::new(&empty_h, 0.01, 1.0);
        assert!(empty_prop.is_empty());
        let mut empty_state = QuantumState::new(DVector::zeros(0));
        empty_prop.evolve_in_place(&mut empty_state);
        assert_eq!(empty_state.vector.len(), 0);

        let one_h = TridiagonalHamiltonian1D::new(vec![2.5], vec![]);
        let mut one_prop = CrankNicolsonPropagator1D::new(&one_h, 0.05, 1.0);
        let mut one_state = QuantumState::new(DVector::from_vec(vec![Complex::new(0.6, -0.8)]));
        one_prop.evolve_in_place(&mut one_state);
        assert!((one_state.norm() - 1.0).abs() < 1e-14);

        one_prop.configure_dt(&one_h, -0.01, 1.0);
        one_prop.evolve_in_place(&mut one_state);
        assert!(!one_state.vector[0].re.is_finite());
    }
}
