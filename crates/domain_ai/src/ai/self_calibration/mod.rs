//! Self-calibration module for AI models.
//!
//! This module contains functionality for:
//! - Defining response types.
//! - Calculating soft self-consistency scores.
//! - Temperature scaling based on answer entropy.
//! - Training via KL divergence loss.

/// Soft self-consistency agreement scoring across sampled responses.
pub mod scoring;
/// Answer-distribution Shannon entropy and adaptive temperature scaling.
pub mod temperature;
/// Kullback-Leibler divergence calibration loss against temperature-scaled targets.
pub mod training;
pub mod types;

// [cite:self_calibration_paper]
