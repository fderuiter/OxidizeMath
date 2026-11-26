//! Deterministic SIR and SEIR compartmental ODE dynamics and builders.

/// Shared parameter validators and scalar basic reproduction number $R_0 = \beta / \gamma$.
pub mod common;
/// Vector-space arithmetic macros for fixed-compartment state structs.
pub mod macros;
/// Four-compartment Susceptible-Exposed-Infectious-Recovered (SEIR) ODE model.
pub mod seir;
/// Three-compartment Susceptible-Infectious-Recovered (SIR) ODE model.
pub mod sir;

pub use common::basic_reproduction_number;
pub use seir::{SEIRModel, SEIRState};
pub use sir::{SIRModel, SIRState};

// [cite:graph_parameters_rust]
