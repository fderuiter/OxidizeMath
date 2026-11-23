pub mod baum_welch;
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

pub use baum_welch::MAX_BAUM_WELCH_ITERATIONS;
pub use baum_welch_types::{BaumWelchConfig, BaumWelchReport, BaumWelchWorkspace};
pub use model::HiddenMarkovModel;
pub use viterbi::ViterbiWorkspace;

// [cite:clinical_trials_statistics]
