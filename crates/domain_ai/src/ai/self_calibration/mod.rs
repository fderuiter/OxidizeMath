//! Self-calibration, split-conformal prediction, and anytime-valid e-value uncertainty estimation.
//!
//! This module contains functionality for:
//! - Defining response types.
//! - Calculating soft self-consistency scores, conformal nonconformity quantiles, prediction sets,
//!   and sequential calibration e-values via [`engine_prob::calibration`].
//! - Temperature scaling based on answer entropy.
//! - Training via KL divergence loss.

#[allow(missing_docs)]
pub mod scoring;
#[allow(missing_docs)]
pub mod temperature;
#[allow(missing_docs)]
pub mod training;
pub mod types;

pub use scoring::{
    EValue, calculate_soft_self_consistency_scores, calibrate_response_threshold,
    conformal_response_mask, conformal_response_set, response_conformal_p_values,
    response_nonconformity_scores, sequential_calibration_e_value,
};

// [cite:self_calibration_paper]
