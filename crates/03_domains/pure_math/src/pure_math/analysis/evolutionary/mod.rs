//! Evolutionary game-theoretic replicator dynamics, linear payoff matrices, and fitness strategies.
// [cite:pure_math]

/// Replicator ODE system $\dot{x}_i = x_i (f_i(\mathbf{x}) - \bar{f}(\mathbf{x}))$.
pub mod replicator;
/// Linear matrix payoff landscape $\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x}$.
pub mod strategies;
/// Shared [`FitnessStrategy`] trait and [`GameTheoryError`] validation variants.
pub mod traits;

pub use replicator::ReplicatorDynamics;
pub use strategies::MatrixPayoff;
pub use traits::{FitnessStrategy, GameTheoryError};
