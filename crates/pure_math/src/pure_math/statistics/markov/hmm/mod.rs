#[allow(missing_docs)]
pub mod forward_backward;
#[allow(missing_docs)]
pub mod generation;
#[allow(missing_docs)]
pub mod model;
#[allow(missing_docs)]
pub mod viterbi;
pub mod workspace;

#[cfg(test)]
mod tests;

pub use model::HiddenMarkovModel;
pub use workspace::{HmmEvaluation, HmmScratch};

// [cite:clinical_trials_statistics]
