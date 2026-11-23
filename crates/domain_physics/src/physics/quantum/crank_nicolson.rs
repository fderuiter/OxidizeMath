// [cite:quantum_mechanics]
//! # 1D Crank-Nicolson Quantum Solver (`CrankNicolson1D`)
//!
//! Provides a self-contained $O(N)$ unitary 1D Schrödinger stepper backed by
//! [`TridiagonalHamiltonian1D`] and [`SolverWorkspace`], enabling instant zero-allocation
//! potential sculpting and $\Delta t$ updates.

use crate::physics::quantum::tridiagonal::{
    SolverWorkspace, TridiagonalHamiltonian1D, valid_positive_scalar,
};
use crate::physics::quantum::types::QuantumState;
use num_complex::Complex;

/// Self-contained $O(N)$ 1D Crank-Nicolson quantum solver over preallocated [`SolverWorkspace`] buffers.
///
/// Combines in-place tridiagonal Hamiltonian assembly and Thomas LU Cayley stepping so spatial
/// potential sculpting and dynamic $\Delta t$ updates execute in linear $O(N)$ time with zero
/// post-initialization heap allocations.
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::{CrankNicolson1D, gaussian_wavepacket};
///
/// let n = 64;
/// let dx = 0.1;
/// let x_grid: Vec<f64> = (0..n).map(|i| -3.2 + i as f64 * dx).collect();
/// let potential: Vec<f64> = x_grid.iter().map(|&x| 0.5 * x * x).collect();
/// let mut solver = CrankNicolson1D::new(&potential, dx, 0.01, 1.0, 1.0);
/// let mut psi = gaussian_wavepacket(&x_grid, -0.8, 2.5, 0.5);
/// assert!(solver.step_substeps_in_place(&mut psi, 8));
/// assert!((psi.norm() - 1.0).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct CrankNicolson1D {
    hamiltonian: TridiagonalHamiltonian1D,
    workspace: SolverWorkspace,
    configured_dx: Option<f64>,
    configured_dt: Option<f64>,
    configured_h_bar: Option<f64>,
    valid: bool,
}

impl CrankNicolson1D {
    /// Preallocates Hamiltonian and [`SolverWorkspace`] buffers for grids up to `max_points`.
    #[verified_engine::verified]
    #[must_use]
    pub fn with_capacity(max_points: usize) -> Self {
        Self {
            hamiltonian: TridiagonalHamiltonian1D::with_capacity(max_points),
            workspace: SolverWorkspace::with_capacity(max_points),
            configured_dx: None,
            configured_dt: None,
            configured_h_bar: None,
            valid: false,
        }
    }

    /// Constructs and factors a [`CrankNicolson1D`] solver for `(potential, dx, dt, mass, h_bar)`.
    #[verified_engine::verified]
    #[must_use]
    pub fn new(potential: &[f64], dx: f64, dt: f64, mass: f64, h_bar: f64) -> Self {
        let mut solver = Self::with_capacity(potential.len());
        let _ = solver.configure(potential, dx, dt, mass, h_bar);
        solver
    }

    /// Configures potential, grid spacing `dx`, and step `dt` in $O(N)$ time, rejecting invalid scalars before division.
    #[verified_engine::verified]
    pub fn configure(
        &mut self,
        potential: &[f64],
        dx: f64,
        dt: f64,
        mass: f64,
        h_bar: f64,
    ) -> bool {
        if !self.update_potential(potential, dx, mass, h_bar) {
            return false;
        }
        self.update_dt(dt, h_bar)
    }

    /// Updates $V(x_i)$ and `dx` in $O(N)$ time without heap allocation when `potential.len() <= capacity()`.
    #[verified_engine::verified]
    pub fn update_potential(&mut self, potential: &[f64], dx: f64, mass: f64, h_bar: f64) -> bool {
        let n = potential.len();
        self.workspace.ensure_capacity(n);
        if !valid_positive_scalar(dx)
            || !valid_positive_scalar(mass)
            || !valid_positive_scalar(h_bar)
            || potential.iter().any(|v| !v.is_finite())
        {
            self.hamiltonian
                .update_potential_in_place(potential, dx, mass, h_bar);
            self.workspace.poison(n);
            self.configured_dx = None;
            self.configured_dt = None;
            self.valid = false;
            return false;
        }
        self.hamiltonian
            .update_potential_in_place(potential, dx, mass, h_bar);
        self.configured_dx = Some(dx);
        self.configured_dt = None;
        self.valid = false;
        true
    }

    /// Re-factors the Cayley Thomas multipliers for `(dt, h_bar)` in $O(N)$ time with zero heap allocations.
    #[verified_engine::verified]
    pub fn update_dt(&mut self, dt: f64, h_bar: f64) -> bool {
        let n = self.hamiltonian.len();
        self.workspace.ensure_capacity(n);
        if !valid_positive_scalar(dt) || !valid_positive_scalar(h_bar) {
            self.workspace.poison(n);
            self.configured_dt = None;
            self.configured_h_bar = None;
            self.valid = false;
            return false;
        }
        let alpha = dt / (2.0 * h_bar);
        let ok = self.workspace.factor_cayley(
            self.hamiltonian.diagonal(),
            self.hamiltonian.off_diagonal(),
            alpha,
        );
        self.configured_dt = if ok { Some(dt) } else { None };
        self.configured_h_bar = if ok { Some(h_bar) } else { None };
        self.valid = ok;
        ok
    }

    /// Invalidates the cached `dt` factorization so the next step refreshes Thomas multipliers.
    #[verified_engine::verified]
    pub fn invalidate(&mut self) {
        self.configured_dt = None;
        self.configured_h_bar = None;
        self.valid = false;
    }

    /// Immutable access to the preallocated [`SolverWorkspace`].
    #[verified_engine::verified]
    #[must_use]
    pub fn workspace(&self) -> &SolverWorkspace {
        &self.workspace
    }

    /// Mutable access to the preallocated [`SolverWorkspace`].
    #[verified_engine::verified]
    #[must_use]
    pub fn workspace_mut(&mut self) -> &mut SolverWorkspace {
        &mut self.workspace
    }

    /// Immutable access to the underlying [`TridiagonalHamiltonian1D`].
    #[verified_engine::verified]
    #[must_use]
    pub fn hamiltonian(&self) -> &TridiagonalHamiltonian1D {
        &self.hamiltonian
    }

    /// Currently configured step size $\Delta t$, if valid.
    #[verified_engine::verified]
    #[must_use]
    pub fn dt(&self) -> Option<f64> {
        self.configured_dt
    }

    /// Currently configured spatial grid spacing $\Delta x$, if valid.
    #[verified_engine::verified]
    #[must_use]
    pub fn dx(&self) -> Option<f64> {
        self.configured_dx
    }

    /// Active spatial grid length $N$.
    #[verified_engine::verified]
    #[must_use]
    pub fn len(&self) -> usize {
        self.hamiltonian.len()
    }

    /// Returns `true` when configured for an empty (`0`-point) grid.
    #[verified_engine::verified]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hamiltonian.is_empty()
    }

    /// Returns `true` when both the Hamiltonian and Cayley Thomas factors are valid and finite.
    #[verified_engine::verified]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Advances complex wavefunction amplitudes `psi` in place by one Crank-Nicolson step $\Delta t$.
    #[verified_engine::verified]
    pub fn step_slice_in_place(&mut self, psi: &mut [Complex<f64>]) -> bool {
        let n = self.hamiltonian.len();
        if !self.valid || psi.len() != n {
            return false;
        }
        self.workspace.sweep_cayley(n, psi)
    }

    /// Advances `state` in place by one unitary Crank-Nicolson step $\Delta t$ with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_in_place(&mut self, state: &mut QuantumState) -> bool {
        self.step_slice_in_place(state.vector.as_mut_slice())
    }

    /// Advances `state` in place by `substeps` consecutive Crank-Nicolson sweeps with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_substeps_in_place(&mut self, state: &mut QuantumState, substeps: usize) -> bool {
        if !self.valid || state.vector.len() != self.hamiltonian.len() {
            return false;
        }
        for _ in 0..substeps {
            if !self.step_in_place(state) {
                return false;
            }
        }
        true
    }
}
