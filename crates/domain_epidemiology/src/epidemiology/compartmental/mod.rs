#[allow(missing_docs)]
pub mod common;
#[allow(missing_docs)]
pub mod macros;
#[allow(missing_docs)]
pub mod seir;
#[allow(missing_docs)]
pub mod sir;

pub use common::basic_reproduction_number;
pub use seir::{SEIRDynamics, SEIRModel, SEIRState};
pub use sir::{SIRModel, SIRState};

// [cite:graph_parameters_rust]
