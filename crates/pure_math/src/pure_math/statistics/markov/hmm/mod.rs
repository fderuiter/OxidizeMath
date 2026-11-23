/// Zero-allocation and simplex-floored Baum-Welch Expectation-Maximization parameter estimation.
pub mod baum_welch;
mod baum_welch_floor;
/// Configuration, convergence reports, and scratch workspaces for Baum-Welch EM.
pub mod baum_welch_types;
#[allow(missing_docs)]
pub mod forward_backward;
#[allow(missing_docs)]
pub mod generation;
#[allow(missing_docs)]
pub mod model;
mod serde_impl;
#[allow(missing_docs)]
pub mod viterbi;

#[cfg(test)]
mod tests;

pub use baum_welch_types::{BaumWelchConfig, BaumWelchReport, BaumWelchResult, BaumWelchWorkspace};
pub use model::HiddenMarkovModel;
pub use viterbi::ViterbiWorkspace;

// [cite:clinical_trials_statistics]
