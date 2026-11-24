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

// [cite:self_calibration_paper]
