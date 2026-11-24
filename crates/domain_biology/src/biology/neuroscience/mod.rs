#![doc = include_str!("README.md")]

pub mod cortical_sheet;
pub mod kinetics;
pub mod kuramoto;
pub mod model;
pub mod neuron;
pub mod synaptic_csr;
pub mod types;

// Re-export the public facade to maintain backward compatibility
pub use cortical_sheet::CorticalSheetRunner;
pub use kuramoto::KuramotoNetwork;
pub use neuron::HodgkinHuxleyNeuron;
pub use synaptic_csr::{MAX_CSR_HH_NEURONS, step_csr_hodgkin_huxley_network};

// Optionally re-export types if we want users to use the advanced API
pub use kinetics::{GatingKinetics, StandardKinetics};
pub use model::HodgkinHuxleyModel;
pub use types::{HodgkinHuxleyParameters, HodgkinHuxleyState};

// [cite:graph_parameters_rust]
