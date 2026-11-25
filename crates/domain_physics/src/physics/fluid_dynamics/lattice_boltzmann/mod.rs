//! Lattice Boltzmann Method (LBM) for fluid simulation.
//!
//! Implements the D2Q9 model with BGK collision operator.
//! This is a mesoscopic method that simulates fluid dynamics by tracking
//! distribution functions on a discrete lattice.

pub mod aerodynamics;
/// Streaming, bounce-back obstacle, Zou-He inlet, and collision step kernels.
pub mod algorithms;
/// Momentum-exchange aerodynamic drag and lift force estimators.
pub mod forces;
/// Discrete velocity lattices (`Lattice2D`, `D2Q9`) and BGK collision operators.
pub mod model;
/// Double-buffered mesoscopic particle distribution grids (`LatticeState`).
pub mod state;

#[cfg(test)]
mod tests;

pub use aerodynamics::*;
pub use forces::compute_drag_coefficient;
pub use model::*;
pub use state::*;

// [cite:gaussian_splatting]
