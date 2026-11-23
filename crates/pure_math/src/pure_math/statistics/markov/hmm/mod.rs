/// Baum-Welch Expectation-Maximization parameter estimation with probability floors.
pub mod baum_welch;
#[allow(missing_docs)]
pub mod forward_backward;
#[allow(missing_docs)]
pub mod generation;
#[allow(missing_docs)]
pub mod model;
#[allow(missing_docs)]
pub mod viterbi;

#[cfg(test)]
mod tests;

pub use baum_welch::BaumWelchResult;
pub use model::HiddenMarkovModel;

// [cite:clinical_trials_statistics]
