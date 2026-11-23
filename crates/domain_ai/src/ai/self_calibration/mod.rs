//! Self-calibration module for AI models.
//!
//! This module contains functionality for:
//! - Defining response types.
//! - Calculating soft self-consistency scores.
//! - Temperature scaling based on answer entropy.
//! - Training via KL divergence loss.

#[allow(missing_docs)]
pub mod scoring;
#[allow(missing_docs)]
pub mod temperature;
#[allow(missing_docs)]
pub mod training;
pub mod types;

pub use scoring::calculate_soft_self_consistency_scores;
pub use temperature::{calculate_answer_entropy, map_entropy_to_temperature};
pub use training::{calculate_kl_divergence_loss, calculate_target_distribution};
pub use types::Response;

// [cite:self_calibration_paper]
// [cite:domain_ai]
