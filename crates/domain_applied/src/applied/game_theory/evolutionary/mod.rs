//! # Evolutionary Dynamics
//!
//! This module models the evolution of strategies in a population over time. Unlike classical game theory,
//! which focuses on static equilibria (Nash), evolutionary game theory studies the dynamic process
//! of strategy adoption based on **biological fitness**.
//!
//! ## Core Concepts
//!
//! - **Replicator Dynamics**: The most common equation governing population change. Strategies that perform better than average increase in frequency.
//! - **Evolutionarily Stable Strategy (ESS)**: A strategy which, if adopted by a population of players, cannot be invaded by any alternative mutant strategy that is initially rare.
//!
//! ## Submodules
//!
//! - `replicator`: Implements the Replicator Dynamics ODE system.
//! - `strategies`: Defines payoff structures (e.g., Matrix Games).
//! - `traits`: Defines the `FitnessStrategy` trait for custom fitness landscapes.

mod graph_replicator;
mod replicator;
mod strategies;
mod traits;

pub use graph_replicator::GraphReplicatorSystem;
pub use replicator::ReplicatorDynamics;
pub use strategies::MatrixPayoff;
pub use traits::FitnessStrategy;

// [cite:graph_parameters_rust]
