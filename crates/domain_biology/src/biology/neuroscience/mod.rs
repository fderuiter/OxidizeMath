#![doc = include_str!("README.md")]

pub mod kinetics;
pub mod kuramoto;
pub mod model;
pub mod neuron;
pub mod types;

// Re-export the public facade to maintain backward compatibility
pub use kuramoto::KuramotoNetwork;
pub use neuron::HodgkinHuxleyNeuron;

// Optionally re-export types if we want users to use the advanced API
pub use kinetics::{GatingKinetics, StandardKinetics};
pub use model::HodgkinHuxleyModel;
pub use types::{HodgkinHuxleyParameters, HodgkinHuxleyState};

// [cite:graph_parameters_rust]
