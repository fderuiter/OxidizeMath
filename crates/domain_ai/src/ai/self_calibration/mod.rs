//! Label-free LLM self-consistency and self-distillation module (`self_calibration_paper.tex`).
//!
//! This module implements unlabeled answer-cluster aggregation:
//! - Defining sampled response types ([`types`]).
//! - Calculating soft self-consistency cluster scores ([`scoring`]).
//! - Entropy-based label-free temperature heuristics alongside re-exported supervised
//!   [`temperature::TemperatureScaler`] ([`temperature`]).
//! - Self-distillation training via KL divergence loss ([`training`]).
//!
//! For supervised proper scoring rules (`brier_score`, Murphy decomposition, `log_loss`, `CRPS`),
//! `ECE`/`MCE` reliability curves, `ROC`/`PR-AUC`, and Platt or `PAVA` isotonic recalibrators,
//! see [`crate::ai::calibration`].

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
