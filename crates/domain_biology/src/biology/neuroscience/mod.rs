#![doc = include_str!("README.md")]

pub mod cable;
pub mod kinetics;
pub mod model;
pub mod neuron;
pub mod sheet;
pub mod sheet_presets;
pub mod types;

// Re-export the public facade to maintain backward compatibility
pub use cable::{DEFAULT_CABLE_LEN, HodgkinHuxleyCable};
pub use neuron::HodgkinHuxleyNeuron;
pub use sheet::{DEFAULT_SHEET_DIM, ExcitableSheet2D, ScheduledS2Pulse};
pub use sheet_presets::SheetPresetKind;

// Optionally re-export types if we want users to use the advanced API
pub use kinetics::{GatingKinetics, StandardKinetics};
pub use model::HodgkinHuxleyModel;
pub use types::{HodgkinHuxleyParameters, HodgkinHuxleyState};

#[cfg(test)]
mod sheet_tests;

// [cite:graph_parameters_rust]
