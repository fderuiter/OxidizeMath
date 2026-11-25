//! # Generic Reaction-Diffusion System
//!
//! This module provides a flexible framework for simulating N-species reaction-diffusion systems.
//!
//! ## Why does this exist?
//!
//! While specific implementations like `TuringSystem` are useful, biological modeling often requires
//! testing novel hypotheses. This module exists to decouple the core mechanics of a simulation
//! (the mathematical integration and spatial grids) from the domain-specific biology (the chemical reactions).
//! By enforcing the Strategy Pattern, researchers can swap solvers, dimensions (1D vs 2D), or kinetics
//! without touching the underlying engine.
//!
//! ## Architecture
//!
//! ```mermaid
//! classDiagram
//!     class ReactionDiffusionSystem {
//!         +model: ReactionDiffusionModel
//!         +state: ChemicalState
//!         +solver: Solver
//!         +step(dt)
//!     }
//!
//!     class ReactionDiffusionModel {
//!         +reaction: ReactionModel
//!         +diffusion: DiffusionModel
//!         +diffusion_coeffs: Vec~f64~
//!     }
//!
//!     class ReactionModel {
//!         <<Trait>>
//!         +reaction()
//!     }
//!
//!     class DiffusionModel {
//!         <<Trait>>
//!         +apply()
//!     }
//!
//!     ReactionDiffusionSystem o-- ReactionDiffusionModel
//!     ReactionDiffusionModel o-- ReactionModel
//!     ReactionDiffusionModel o-- DiffusionModel
//! ```
//!
//! ## Example
//!
//! Implementing a simple decay system using the generic framework:
//!
//! ```rust
//! # use domain_biology::__doctest_facade as math_explorer;
//! use math_explorer::biology::reaction_diffusion::{ReactionDiffusionSystem, ReactionModel, ChemicalState};
//! use math_explorer::biology::diffusion::FiniteDifference1D;
//!
//! // 1. Define custom kinetics (e.g., simple exponential decay)
//! struct DecayKinetics { rate: f64 }
//! impl ReactionModel for DecayKinetics {
//!     fn reaction(&self, concs: &[f64], rates: &mut [f64]) {
//!         rates[0] = -self.rate * concs[0];
//!     }
//! }
//!
//! // 2. Setup the generic system (1 species, 10 grid points)
//! let kinetics = DecayKinetics { rate: 0.1 };
//! let diffusion = FiniteDifference1D::new(math_explorer::math_kernel::types::StepSize(1.0));
//! let mut system = ReactionDiffusionSystem::builder()
//!     .num_species(1)
//!     .grid_size(10)
//!     .reaction(kinetics)
//!     .diffusion(diffusion)
//!     .diffusion_coeffs(vec![0.5])
//!     .build()
//!     .unwrap();
//!
//! // 3. Initialize state and run
//! system.state.species_mut(0)[5] = 100.0; // Spike in the middle
//! system.step(0.1);
//!
//! assert!(system.state.species(0)[5] < 100.0); // Concentration decayed
//! ```

/// Generic $N$-species reaction-diffusion ODE model, system wrapper, and builder.
pub mod model;
/// Concrete biochemical reaction kinetics for multi-species reaction-diffusion systems.
pub mod models;
/// Column-major multi-species spatial concentration matrix (`ChemicalState`).
pub mod state;
/// Pluggable `ReactionModel` and `DiffusionModel` strategy traits.
pub mod traits;

pub use model::{ReactionDiffusionModel, ReactionDiffusionSystem, ReactionDiffusionSystemBuilder};
pub use models::HillSubstrateReaction;
pub use state::ChemicalState;
pub use traits::{DiffusionModel, ReactionModel, STACK_REACTION_SPECIES_CAP};

/// Errors related to Reaction-Diffusion systems.
#[derive(Debug, Clone, Copy, thiserror::Error, PartialEq, Eq)]
pub enum ReactionDiffusionError {
    /// Underlying dimension-mismatch or numerical error from `math_commons`.
    #[error("Mathematical error: {0}")]
    Math(#[from] math_commons::error::MathError),
    /// Builder configured zero chemical species (`num_species == 0`).
    #[error("System requires at least one species")]
    ZeroSpecies,
    /// Builder configured zero spatial grid cells (`grid_size == 0`).
    #[error("Grid size cannot be zero")]
    ZeroGridSize,
    /// Required builder field was omitted prior to `build()`.
    #[error("Missing parameter: {0}")]
    MissingParameter(&'static str),
    /// Spatial step size along an axis was non-positive or non-finite.
    #[error("Invalid spatial step size on axis '{axis}'")]
    InvalidStepSize {
        /// Axis identifier (`"dx"`, `"dy"`, or `"dz"`).
        axis: &'static str,
    },
    /// Byte slice supplied to `CellMask2D::set_roles_slice` did not match `width * height`.
    #[error("CellMask2D byte slice length {actual} does not match expected grid size {expected}")]
    CellMaskLengthMismatch {
        /// Expected byte length (`width * height`).
        expected: usize,
        /// Actual byte slice length supplied.
        actual: usize,
    },
}

#[cfg(test)]
mod tests;

// [cite:partitions_implementation]
