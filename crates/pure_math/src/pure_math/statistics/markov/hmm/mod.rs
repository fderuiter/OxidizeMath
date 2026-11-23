pub mod baum_welch;
pub mod baum_welch_types;
/// Submodule providing `forward_backward` types, traits, and algorithms.
pub mod forward_backward;
/// Submodule providing `generation` types, traits, and algorithms.
pub mod generation;
/// Submodule providing `model` types, traits, and algorithms.
pub mod model;
mod serde_impl;
/// Submodule providing `viterbi` types, traits, and algorithms.
pub mod viterbi;

#[cfg(test)]
mod tests;

pub use baum_welch_types::{BaumWelchConfig, BaumWelchReport, BaumWelchWorkspace};
pub use model::HiddenMarkovModel;
pub use viterbi::ViterbiWorkspace;

// [cite:clinical_trials_statistics]
