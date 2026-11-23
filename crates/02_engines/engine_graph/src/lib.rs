#![no_std]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
extern crate std;

/// Hidden doctest shim mapping `math_explorer::engine_graph` to this crate.
#[doc(hidden)]
pub mod __doctest_facade {
    pub use crate as engine_graph;
}

pub mod bp_max_product;
pub mod bp_sum_product;
pub mod chain_mrf;
pub mod error;
pub mod factor_graph;

#[cfg(feature = "alloc")]
pub use bp_max_product::max_product_bp;
pub use bp_max_product::{MaxProductReport, evaluate_assignment_log_score, max_product_bp_into};

#[cfg(feature = "alloc")]
pub use bp_sum_product::sum_product_bp;
pub use bp_sum_product::{BpConfig, BpReport, sum_product_bp_into};

pub use chain_mrf::{
    ChainFbWorkspace, ChainModelView, chain_accumulate_xi_into, chain_forward_backward_into,
    potts_neighborhood_posteriors_into,
};
pub use error::GraphEngineError;
pub use factor_graph::{FactorGraphView, FactorSpec, MAX_CARDINALITY, MAX_FACTOR_ARITY};

// [cite:graph_parameters_rust]
// [cite:clinical_trials_statistics]
