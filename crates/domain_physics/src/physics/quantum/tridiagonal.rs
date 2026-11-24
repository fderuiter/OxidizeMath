// [cite:quantum_mechanics]
//! # Tridiagonal 1D Quantum Hamiltonian, `SolverWorkspace`, & Propagators
//!
//! Provides $O(N)$-memory tridiagonal representations of 1D finite-difference Schrödinger
//! Hamiltonians and zero-allocation, strictly unitary Crank-Nicolson (Cayley transform)
//! solvers backed by [`pure_math::pure_math::analysis::pde::tridiagonal`]:
//!
//! $$ \left(I + \frac{i\Delta t}{2\hbar}\hat{H}\right)|\psi(t + \Delta t)\rangle
//!    = \left(I - \frac{i\Delta t}{2\hbar}\hat{H}\right)|\psi(t)\rangle $$

pub use crate::physics::quantum::crank_nicolson::CrankNicolson1D;
use crate::physics::quantum::types::QuantumState;
use nalgebra::DVector;
use num_complex::Complex;
use pure_math::pure_math::analysis::pde::tridiagonal::{
    apply_tridiagonal, factor_tridiagonal, solve_tridiagonal_factored,
};

#[inline]
#[verified_engine::verified]
pub(crate) fn valid_positive_scalar(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Preallocated complex band and Thomas-sweep scratch buffers for zero-allocation 1D Cayley stepping.
#[derive(Clone, Debug)]
pub struct SolverWorkspace {
    lhs_off: Vec<Complex<f64>>,
    lhs_main: Vec<Complex<f64>>,
    rhs_off: Vec<Complex<f64>>,
    rhs_main: Vec<Complex<f64>>,
    c_prime: Vec<Complex<f64>>,
    inv_denom: Vec<Complex<f64>>,
    rhs_vec: Vec<Complex<f64>>,
    sweep_scratch: Vec<Complex<f64>>,
}

impl SolverWorkspace {
    /// Preallocates all complex tridiagonal bands and Thomas scratch buffers for up to `max_points`.
    #[verified_engine::verified]
    #[must_use]
    pub fn with_capacity(max_points: usize) -> Self {
        let zero = Complex::new(0.0, 0.0);
        let off_cap = max_points.saturating_sub(1);
        Self {
            lhs_off: vec![zero; off_cap],
            lhs_main: vec![zero; max_points],
            rhs_off: vec![zero; off_cap],
            rhs_main: vec![zero; max_points],
            c_prime: vec![zero; off_cap],
            inv_denom: vec![zero; max_points],
            rhs_vec: vec![zero; max_points],
            sweep_scratch: vec![zero; max_points],
        }
    }

    /// Maximum grid capacity currently allocated without triggering heap reallocation.
    #[verified_engine::verified]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.lhs_main.len()
    }

    /// Ensures workspace buffers hold at least `n` grid points (zero heap allocations if `n <= capacity()`).
    #[verified_engine::verified]
    pub fn ensure_capacity(&mut self, n: usize) {
        let zero = Complex::new(0.0, 0.0);
        if n > self.lhs_main.len() {
            self.lhs_main.resize(n, zero);
            self.rhs_main.resize(n, zero);
            self.inv_denom.resize(n, zero);
            self.rhs_vec.resize(n, zero);
            self.sweep_scratch.resize(n, zero);
        }
        let n_off = n.saturating_sub(1);
        if n_off > self.lhs_off.len() {
            self.lhs_off.resize(n_off, zero);
            self.rhs_off.resize(n_off, zero);
            self.c_prime.resize(n_off, zero);
        }
    }

    #[verified_engine::verified]
    pub(crate) fn poison(&mut self, n: usize) {
        let nan = Complex::new(f64::NAN, f64::NAN);
        let zero = Complex::new(0.0, 0.0);
        self.lhs_main[..n].fill(nan);
        self.rhs_main[..n].fill(nan);
        self.inv_denom[..n].fill(nan);
        let n_off = n.saturating_sub(1);
        self.lhs_off[..n_off].fill(zero);
        self.rhs_off[..n_off].fill(zero);
        self.c_prime[..n_off].fill(zero);
    }

    #[verified_engine::verified]
    pub(crate) fn factor_cayley(&mut self, diag: &[f64], off: &[f64], alpha: f64) -> bool {
        let n = diag.len();
        self.ensure_capacity(n);
        if n == 0 {
            return true;
        }
        let n_off = n - 1;
        if off.len() != n_off || !alpha.is_finite() {
            self.poison(n);
            return false;
        }
        for (i, &d) in diag.iter().enumerate() {
            if !d.is_finite() {
                self.poison(n);
                return false;
            }
            self.lhs_main[i] = Complex::new(1.0, alpha * d);
            self.rhs_main[i] = Complex::new(1.0, -alpha * d);
        }
        for (i, &e) in off.iter().enumerate() {
            if !e.is_finite() {
                self.poison(n);
                return false;
            }
            self.lhs_off[i] = Complex::new(0.0, alpha * e);
            self.rhs_off[i] = Complex::new(0.0, -alpha * e);
        }
        let ok = factor_tridiagonal(
            &self.lhs_off[..n_off],
            &self.lhs_main[..n],
            &self.lhs_off[..n_off],
            &mut self.c_prime[..n_off],
            &mut self.inv_denom[..n],
        );
        if !ok {
            self.poison(n);
        }
        ok
    }

    #[verified_engine::verified]
    pub(crate) fn sweep_cayley(&mut self, n: usize, psi: &mut [Complex<f64>]) -> bool {
        if psi.len() != n {
            return false;
        }
        if n == 0 {
            return true;
        }
        if n == 1 {
            psi[0] = (self.rhs_main[0] * psi[0]) * self.inv_denom[0];
            return psi[0].re.is_finite() && psi[0].im.is_finite();
        }
        let n_off = n - 1;
        if !apply_tridiagonal(
            &self.rhs_off[..n_off],
            &self.rhs_main[..n],
            &self.rhs_off[..n_off],
            psi,
            &mut self.rhs_vec[..n],
        ) {
            psi.fill(Complex::new(f64::NAN, f64::NAN));
            return false;
        }
        let ok = solve_tridiagonal_factored(
            &self.lhs_off[..n_off],
            &self.c_prime[..n_off],
            &self.inv_denom[..n],
            &self.rhs_vec[..n],
            &mut self.sweep_scratch[..n],
            psi,
        );
        if !ok {
            psi.fill(Complex::new(f64::NAN, f64::NAN));
        }
        ok
    }
}

/// Compact $O(N)$ representation of a 1D real symmetric tridiagonal Hamiltonian
/// $\hat{H} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x)$.
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

    /// Updates this Hamiltonian in place from a potential slice $V(x_i)$, rejecting non-positive or
    /// non-finite `(dx, mass, h_bar)` before division and performing zero heap allocations when
    /// `potential.len()` fits within capacity.
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
        let denom = mass * dx * dx;
        if !valid_positive_scalar(dx)
            || !valid_positive_scalar(mass)
            || !h_bar.is_finite()
            || !valid_positive_scalar(denom)
        {
            self.diagonal[..n].fill(f64::NAN);
            self.off_diagonal[..n_off].fill(f64::NAN);
            return;
        }
        let diag_kin = (h_bar * h_bar) / denom;
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
/// Reference: Crank & Nicolson (1947), DOI: 10.1017/S0305004100023197;
/// [`docs/math_specs/domain_physics.md`](../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// Cayley rational approximation $(I + i\alpha\hat{H})^{-1}(I - i\alpha\hat{H})$ is unconditionally
/// unitary for real symmetric $\hat{H}$ and strictly diagonally dominant in the complex plane ($|1 + i\alpha d_j| \ge 1$).
///
/// # Domain of Validity
/// Requires finite [`TridiagonalHamiltonian1D`] bands, strictly positive finite `dt > 0` and
/// `h_bar > 0`, and `state.vector.len() == self.len()`.
#[derive(Clone, Debug)]
pub struct CrankNicolsonPropagator1D {
    workspace: SolverWorkspace,
    n_points: usize,
    configured_dt: Option<f64>,
    configured_h_bar: Option<f64>,
}

impl CrankNicolsonPropagator1D {
    /// Preallocates Thomas sweep workspace buffers for grids up to `max_points`.
    #[verified_engine::verified]
    #[must_use]
    pub fn with_capacity(max_points: usize) -> Self {
        Self {
            workspace: SolverWorkspace::with_capacity(max_points),
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

    /// Precomputes the $O(N)$ Thomas LU multipliers for `(hamiltonian, dt, h_bar)` with zero
    /// heap allocations whenever `hamiltonian.len()` fits within preallocated capacity.
    #[verified_engine::verified]
    pub fn configure_dt(&mut self, hamiltonian: &TridiagonalHamiltonian1D, dt: f64, h_bar: f64) {
        let n = hamiltonian.len();
        self.workspace.ensure_capacity(n);
        self.n_points = n;
        self.configured_dt = Some(dt);
        self.configured_h_bar = Some(h_bar);
        if n == 0 {
            return;
        }
        if !valid_positive_scalar(dt) || !valid_positive_scalar(h_bar) {
            self.workspace.poison(n);
            return;
        }
        let alpha = dt / (2.0 * h_bar);
        let _ =
            self.workspace
                .factor_cayley(hamiltonian.diagonal(), hamiltonian.off_diagonal(), alpha);
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
        let _ = self.workspace.sweep_cayley(n, state.vector.as_mut_slice());
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
#[path = "tridiagonal_tests.rs"]
mod tests;
