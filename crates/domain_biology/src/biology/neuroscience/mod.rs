#![doc = include_str!("README.md")]

pub mod cable;
pub mod cortical_sheet;
pub mod kinetics;
pub mod kuramoto;
pub mod model;
pub mod neuron;
pub mod sheet;
pub mod sheet_presets;
pub mod synaptic_csr;
pub mod types;

// Re-export the public facade to maintain backward compatibility
pub use cable::{DEFAULT_CABLE_LEN, HodgkinHuxleyCable};
pub use cortical_sheet::CorticalSheetRunner;
pub use kuramoto::KuramotoNetwork;
pub use neuron::HodgkinHuxleyNeuron;
pub use sheet::{DEFAULT_SHEET_DIM, ExcitableSheet2D, ScheduledS2Pulse};
pub use sheet_presets::SheetPresetKind;
pub use synaptic_csr::{MAX_CSR_HH_NEURONS, step_csr_hodgkin_huxley_network};

// Optionally re-export types if we want users to use the advanced API
pub use kinetics::{GatingKinetics, StandardKinetics};
pub use model::HodgkinHuxleyModel;
pub use types::{HodgkinHuxleyParameters, HodgkinHuxleyState};

#[cfg(test)]
mod sheet_tests;

// [cite:graph_parameters_rust]
