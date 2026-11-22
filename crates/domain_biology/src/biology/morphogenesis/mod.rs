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

pub mod gray_scott;
#[allow(missing_docs)]
pub mod reaction;
#[allow(missing_docs)]
pub mod solvers;
#[allow(missing_docs)]
pub mod state;

use crate::biology::diffusion::{FiniteDifference1D, SpatialDiffusion};
use pure_math::pure_math::analysis::ode::{OdeSystem, TimeStepper};

pub use gray_scott::GrayScottKinetics;
pub use reaction::{ReactionKinetics, SchnakenbergKinetics};
pub use solvers::{FusedEulerSolver, StandardSolverAdapter, TuringDynamics, TuringSolverStrategy};
pub use state::TuringState;

/// Represents a Reaction-Diffusion system.
///
/// # Generics
/// * `N`: Number of species (default 2).
/// * `K`: The reaction kinetics strategy.
/// * `D`: The spatial diffusion strategy.
/// * `S`: The numerical solver strategy.
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
}

use math_commons::math_kernel::types::{Dimension, StepSize};
use std::ops::{Deref, DerefMut};

/// Diffusion coefficient
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct DiffusionCoeff(pub f64);

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
        }
    }

    /// Updates the grid using the solver strategy.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        step_turing_buffers(
            &mut self.state,
            &mut self.next_state,
            &mut self.solver,
            &self.kinetics,
            &self.diffusion,
            self.diffusion_coeffs,
            dt,
        );
    }

    /// Updates the grid using an alternate stack-inlined kinetics strategy without reallocating buffers.
    #[verified_engine::verified]
    pub fn step_with_kinetics<K2: ReactionKinetics<N>>(&mut self, kinetics: &K2, dt: f64) {
        step_turing_buffers(
            &mut self.state,
            &mut self.next_state,
            &mut self.solver,
            kinetics,
            &self.diffusion,
            self.diffusion_coeffs,
            dt,
        );
    }
}

#[verified_engine::verified]
fn step_turing_buffers<
    const N: usize,
    K: ReactionKinetics<N>,
    D: SpatialDiffusion<N>,
    S: TuringSolverStrategy<N>,
>(
    state: &mut TuringState<N>,
    next_state: &mut TuringState<N>,
    solver: &mut S,
    kinetics: &K,
    diffusion: &D,
    diffusion_coeffs: [f64; N],
    dt: f64,
) {
    let n = state.len();
    if n == 0 {
        return;
    }
    if next_state.len() != n {
        *next_state = TuringState::new(n);
    }
    let dynamics = TuringDynamics {
        kinetics,
        diffusion,
        diffusion_coeffs,
    };
    solver.step(state, next_state, &dynamics, dt);
    std::mem::swap(state, next_state);
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

    #[verified_engine::verified(
        allow_recursion = "delegates trait step to inherent TuringSystem::step"
    )]
    fn step(&mut self, dt: f64) {
        // Delegate to the optimized inherent method
        self.step(dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "State vector 1 length mismatch")]
    #[verified_engine::verified]
    fn test_derivative_in_place_safety_check() {
        let n = 10;
        let system = TuringSystem::new(
            math_commons::math_kernel::types::Dimension(n),
            DiffusionCoeff(1.0),
            DiffusionCoeff(1.0),
            math_commons::math_kernel::types::StepSize(1.0),
        );
        let mut state = TuringState::new(n);

        // Corrupt the state (simulate internal bug or misuse)
        // Access concentrations directly via pub(crate)
        state.concentrations[1].pop(); // Make v shorter than u

        let mut out = TuringState::new(n);

        system.derivative_in_place(0.0, &state, &mut out);
    }

    #[test]
    #[verified_engine::verified]
    fn test_turing_system_logic_preservation() {
        // Setup a small system
        let n = 10;
        let d_u = 1.0;
        let d_v = 0.5;
        let dx = 1.0;
        let mut system = TuringSystem::new(
            math_commons::math_kernel::types::Dimension(n),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            math_commons::math_kernel::types::StepSize(dx),
        );

        // Initialize with some pattern
        for i in 0..n {
            system.state.u_mut()[i] = 1.0 + 0.1 * (i as f64);
            system.state.v_mut()[i] = 0.5 - 0.05 * (i as f64);
        }

        // Run for a few steps
        let dt = 0.01;
        for _ in 0..5 {
            system.step(dt);
        }

        // Capture output
        let u_out = system.u().to_vec();
        let v_out = system.v().to_vec();

        // Expected values captured from baseline run (from original morphogenesis.rs)
        let expected_u = [
            0.9798926377401955,
            1.0722504645444493,
            1.1685990805783317,
            1.2642647090938448,
            1.359028327357602,
            1.4527705800845148,
            1.5453811790730032,
            1.6367576303434268,
            1.7267186541725483,
            1.8109737170223916,
        ];
        let expected_v = [
            0.47709091921002866,
            0.4263770084483741,
            0.3750152156844884,
            0.32443296992262166,
            0.2747722006954079,
            0.22615405798594523,
            0.1786914141249832,
            0.1324883911255509,
            0.08765106523936222,
            0.04531981611374585,
        ];

        // Assert with tolerance
        let tolerance = math_commons::registry::TOLERANCE_HIGH;
        for i in 0..n {
            assert!(
                (u_out[i] - expected_u[i]).abs() < tolerance,
                "U mismatch at {}: {} vs {}",
                i,
                u_out[i],
                expected_u[i]
            );
            assert!(
                (v_out[i] - expected_v[i]).abs() < tolerance,
                "V mismatch at {}: {} vs {}",
                i,
                v_out[i],
                expected_v[i]
            );
        }
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]
