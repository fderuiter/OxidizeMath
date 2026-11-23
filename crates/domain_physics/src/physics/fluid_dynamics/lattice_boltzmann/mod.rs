//! Lattice Boltzmann Method (LBM) for fluid simulation.
//!
//! Implements the D2Q9 model with BGK collision operator.
//! This is a mesoscopic method that simulates fluid dynamics by tracking
//! distribution functions on a discrete lattice.

#[allow(missing_docs)]
pub mod algorithms;
#[allow(missing_docs)]
pub mod forces;
#[allow(missing_docs)]
pub mod model;
#[allow(missing_docs)]
pub mod state;
mod telemetry;

#[cfg(test)]
mod tests;

pub use forces::compute_drag_coefficient;
pub use model::*;
pub use state::*;

// [cite:gaussian_splatting]
