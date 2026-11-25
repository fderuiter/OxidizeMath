//! # Morphogenesis (Turing Patterns)
//!
//! This module implements a Reaction-Diffusion system capable of generating Turing patterns.
//! It uses a 1D grid to simulate the interaction between an activator ($u$) and an inhibitor ($v$).
//!
//! Turing patterns arise when a stable uniform state becomes unstable due to diffusion (Diffusion-driven instability),
//! typically when the inhibitor diffuses much faster than the activator ($D_v \gg D_u$).
//!
//! ##  The Mechanism
//!
//! ```mermaid
//! graph TD
//!     subgraph "Local Reaction"
//!     A[Activator U] -->|Self-Catalysis| A
//!     A -->|Activates| B[Inhibitor V]
//!     B -->|Inhibits| A
//!     end
//!
//!     subgraph "Spatial Diffusion"
//!     DiffA[Diffusion of U]
//!     DiffB[Diffusion of V]
//!     end
//!
//!     A --- DiffA
//!     B --- DiffB
//!
//!     DiffA -->|Short Range| Patterns
//!     DiffB -->|Long Range| Patterns
//!
//!     style A fill:#a5d6a7,stroke:#2e7d32
//!     style B fill:#ef9a9a,stroke:#c62828
//! ```
//!
//! ##  Quick Start
//!
//! Simulate the emergence of a pattern from random noise.
//!
//! ```rust
//! # use domain_biology::__doctest_facade as math_explorer;
//! use math_explorer::biology::morphogenesis::{TuringSystem, SchnakenbergKinetics};
//!
//! // 1. System Configuration
//! // Activator diffuses slowly (1.0), Inhibitor diffuses fast (40.0)
//! let n = 100;
//! let mut system = TuringSystem::new(math_explorer::math_kernel::types::Dimension(n), math_explorer::biology::morphogenesis::DiffusionCoeff(1.0), math_explorer::biology::morphogenesis::DiffusionCoeff(40.0), math_explorer::math_kernel::types::StepSize(1.0));
//!
//! // 2. Initialize with Random Noise
//! // A uniform state would be stable without noise
//! use math_explorer::rng::OxidizeRng;
//! use math_explorer::rand::Rng;
//! let mut rng = OxidizeRng::default();
//! for i in 0..n {
//!     system.u_mut()[i] = 1.0 + rng.gen_range(-0.1..0.1); // Add some perturbation
//!     system.v_mut()[i] = 0.5 + rng.gen_range(-0.1..0.1);
//! }
//!
//! // 3. Run Simulation
//! let dt = 0.01;
//! for _ in 0..100 {
//!     system.step(dt);
//! }
//!
//! // 4. Analyze Results
//! let u_center = system.u()[50];
//! println!("Concentration of Activator at center: {:.4}", u_center);
//! ```

pub mod advection;
/// Local N-species autocatalytic reaction kinetics (Schnakenberg and Gray-Scott).
pub mod reaction;
/// Fused forward-Euler and generic ODE-adapter Turing time-stepping strategies.
pub mod solvers;
/// Spatially heterogeneous Schnakenberg kinetics with preallocated masks.
pub mod spatial_reaction;
/// Contiguous N-species spatial concentration state buffers.
pub mod state;

use crate::biology::diffusion::{FiniteDifference1D, SpatialDiffusion};
use pure_math::pure_math::analysis::ode::{OdeSystem, TimeStepper};

pub use reaction::{GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics};
pub use solvers::{FusedEulerSolver, StandardSolverAdapter, TuringDynamics, TuringSolverStrategy};
pub use spatial_reaction::SpatialSchnakenbergKinetics;
pub use state::TuringState;

/// Represents a Reaction-Diffusion system.
///
/// # Generics
/// * `N`: Number of species (default 2).
/// * `K`: The reaction kinetics strategy.
/// * `D`: The spatial diffusion strategy.
/// * `S`: The numerical solver strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct TuringSystem<
    const N: usize = 2,
    K: ReactionKinetics<N> = SchnakenbergKinetics,
    D: SpatialDiffusion<N> = FiniteDifference1D,
    S: TuringSolverStrategy<N> = FusedEulerSolver,
> {
    /// The current state of the system.
    pub state: TuringState<N>,

    // Double buffer for the next state.
    next_state: TuringState<N>,

    /// Diffusion coefficients for each species
    pub diffusion_coeffs: [f64; N],
    /// Reaction kinetics strategy
    pub kinetics: K,
    /// Spatial diffusion strategy
    pub diffusion: D,
    /// Numerical solver strategy
    pub solver: S,
    pub(crate) time: f64,
}

use math_commons::math_kernel::types::{Dimension, StepSize};
use std::ops::{Deref, DerefMut};

/// Diffusion coefficient
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct DiffusionCoeff(
    /// Non-negative species diffusivity $D \ge 0$.
    pub f64,
);

impl Deref for DiffusionCoeff {
    type Target = f64;
    #[verified_engine::verified]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DiffusionCoeff {
    #[verified_engine::verified]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Convenience implementation for the standard 2-species case
impl TuringSystem<2, SchnakenbergKinetics, FiniteDifference1D, FusedEulerSolver> {
    /// Creates a new Turing System with default Schnakenberg kinetics, 1D Finite Difference, and Fused Euler Solver.
    #[verified_engine::verified]
    pub fn new(size: Dimension, d_u: DiffusionCoeff, d_v: DiffusionCoeff, dx: StepSize) -> Self {
        Self {
            state: TuringState::new(*size),
            next_state: TuringState::new(*size),
            diffusion_coeffs: [*d_u, *d_v],
            kinetics: SchnakenbergKinetics::default(),
            diffusion: FiniteDifference1D::new(dx),
            solver: FusedEulerSolver::new(),
            time: 0.0,
        }
    }
}

// Convenience implementation for custom kinetics/diffusion in 2-species case
impl<K: ReactionKinetics<2>, D: SpatialDiffusion<2>> TuringSystem<2, K, D, FusedEulerSolver> {
    /// Creates a new Turing System with custom kinetics and diffusion strategy, using the default Fused Euler solver.
    #[verified_engine::verified]
    pub fn new_with_kinetics(
        size: Dimension,
        d_u: DiffusionCoeff,
        d_v: DiffusionCoeff,
        kinetics: K,
        diffusion: D,
    ) -> Self {
        Self {
            state: TuringState::new(*size),
            next_state: TuringState::new(*size),
            diffusion_coeffs: [*d_u, *d_v],
            kinetics,
            diffusion,
            solver: FusedEulerSolver::new(),
            time: 0.0,
        }
    }
}

impl
    TuringSystem<
        2,
        SchnakenbergKinetics,
        crate::biology::diffusion::FiniteDifference3D,
        FusedEulerSolver,
    >
{
    /// Creates a 3D two-species Schnakenberg Turing system backed by 7-point Neumann [`crate::biology::diffusion::FiniteDifference3D`].
    #[verified_engine::verified]
    pub fn new_3d(
        dims: (Dimension, Dimension, Dimension),
        d_u: DiffusionCoeff,
        d_v: DiffusionCoeff,
        h: StepSize,
        kinetics: SchnakenbergKinetics,
    ) -> Self {
        let diffusion =
            crate::biology::diffusion::FiniteDifference3D::isotropic(dims.0, dims.1, dims.2, h);
        let total = Dimension(diffusion.grid_len());
        Self::new_with_kinetics(total, d_u, d_v, kinetics, diffusion)
    }

    /// Copies activator concentrations `u` into `out_voxels` as `f32` with zero heap allocations.
    #[verified_engine::verified]
    pub fn write_activator_f32_into(&self, out_voxels: &mut [f32]) {
        let u = self.u();
        let len = out_voxels.len().min(u.len());
        for i in 0..len {
            out_voxels[i] = u[i] as f32;
        }
    }
}

// Generic implementation
impl<const N: usize, K: ReactionKinetics<N>, D: SpatialDiffusion<N>, S: TuringSolverStrategy<N>>
    TuringSystem<N, K, D, S>
{
    /// Creates a new Turing System with custom kinetics, diffusion strategy, and solver.
    #[verified_engine::verified]
    pub fn new_with_solver(
        size: Dimension,
        diffusion_coeffs: [DiffusionCoeff; N],
        kinetics: K,
        diffusion: D,
        solver: S,
    ) -> Self {
        let mut coeffs = [0.0; N];
        for i in 0..N {
            coeffs[i] = *diffusion_coeffs[i];
        }
        Self {
            state: TuringState::new(*size),
            next_state: TuringState::new(*size),
            diffusion_coeffs: coeffs,
            kinetics,
            diffusion,
            solver,
            time: 0.0,
        }
    }

    /// Returns the elapsed simulation time $t$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Sets the simulation clock to `t`.
    #[verified_engine::verified]
    pub fn set_time(&mut self, t: f64) {
        self.time = t;
    }

    #[inline]
    #[verified_engine::verified]
    fn step_bufs<KAlt: ReactionKinetics<N>>(
        time: &mut f64,
        state: &mut TuringState<N>,
        next_state: &mut TuringState<N>,
        dynamics: TuringDynamics<'_, N, KAlt, D>,
        solver: &mut S,
        dt: f64,
    ) {
        let n = state.len();
        if n == 0 {
            return;
        }
        if next_state.len() != n {
            *next_state = TuringState::new(n);
        }
        solver.step_at(*time, state, next_state, &dynamics, dt);
        std::mem::swap(state, next_state);
        *time += dt;
    }

    /// Updates the grid using the solver strategy.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let dyn_ref = TuringDynamics {
            kinetics: &self.kinetics,
            diffusion: &self.diffusion,
            diffusion_coeffs: self.diffusion_coeffs,
        };
        Self::step_bufs(
            &mut self.time,
            &mut self.state,
            &mut self.next_state,
            dyn_ref,
            &mut self.solver,
            dt,
        );
    }

    /// Advances one time step `dt` using alternate reaction `kinetics` while reusing pre-allocated
    /// `state` and `next_state` buffers with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_with_kinetics<KAlt: ReactionKinetics<N>>(&mut self, kinetics: &KAlt, dt: f64) {
        let dyn_ref = TuringDynamics {
            kinetics,
            diffusion: &self.diffusion,
            diffusion_coeffs: self.diffusion_coeffs,
        };
        Self::step_bufs(
            &mut self.time,
            &mut self.state,
            &mut self.next_state,
            dyn_ref,
            &mut self.solver,
            dt,
        );
    }
}

// Convenience accessors for N=2
impl<K: ReactionKinetics<2>, D: SpatialDiffusion<2>, S: TuringSolverStrategy<2>>
    TuringSystem<2, K, D, S>
{
    /// Accessor for the activator concentrations (backward compatibility/convenience).
    #[verified_engine::verified]
    pub fn u(&self) -> &[f64] {
        self.state.u()
    }

    /// Accessor for the inhibitor concentrations (backward compatibility/convenience).
    #[verified_engine::verified]
    pub fn v(&self) -> &[f64] {
        self.state.v()
    }

    /// Mutable accessor for the activator concentrations.
    #[verified_engine::verified]
    pub fn u_mut(&mut self) -> &mut [f64] {
        self.state.u_mut()
    }

    /// Mutable accessor for the inhibitor concentrations.
    #[verified_engine::verified]
    pub fn v_mut(&mut self) -> &mut [f64] {
        self.state.v_mut()
    }
}

impl<const N: usize, K: ReactionKinetics<N>, D: SpatialDiffusion<N>, S: TuringSolverStrategy<N>>
    OdeSystem<TuringState<N>> for TuringSystem<N, K, D, S>
{
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &TuringState<N>) -> TuringState<N> {
        let mut out = TuringState::new(state.len());
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &TuringState<N>, out: &mut TuringState<N>) {
        let dynamics = TuringDynamics {
            kinetics: &self.kinetics,
            diffusion: &self.diffusion,
            diffusion_coeffs: self.diffusion_coeffs,
        };
        dynamics.derivative_in_place(t, state, out);
    }
}

impl<const N: usize, K: ReactionKinetics<N>, D: SpatialDiffusion<N>, S: TuringSolverStrategy<N>>
    TimeStepper<TuringState<N>> for TuringSystem<N, K, D, S>
{
    #[verified_engine::verified]
    fn get_state(&self) -> &TuringState<N> {
        &self.state
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut TuringState<N> {
        &mut self.state
    }

    #[verified_engine::verified]
    fn current_time(&self) -> f64 {
        self.time
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        self.time += dt;
    }

    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        let dyn_ref = TuringDynamics {
            kinetics: &self.kinetics,
            diffusion: &self.diffusion,
            diffusion_coeffs: self.diffusion_coeffs,
        };
        let mut step_t = t;
        Self::step_bufs(
            &mut step_t,
            &mut self.state,
            &mut self.next_state,
            dyn_ref,
            &mut self.solver,
            dt,
        );
    }

    #[verified_engine::verified(
        allow_recursion = "delegates trait step to inherent TuringSystem::step"
    )]
    fn step(&mut self, dt: f64) {
        // Delegate to the optimized inherent method
        self.step(dt);
    }
}

#[cfg(test)]
mod tests;

// [cite:dwarf_galaxy_empirical_dependencies]
