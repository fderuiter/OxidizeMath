//! # Applied Mathematics
//!
//! This module serves as a collection of mathematical models applied to specific,
//! often niche or complex, domains. It demonstrates how core mathematical concepts
//! (calculus, statistics, game theory) translate into practical solutions.
//!
//! ## Domains
//!
//! - **Biology & Medicine**: `clinical_trials`, `pharmacokinetics`, `freesurfer`, `cannibalism`.
//! - **Scientific Visualization**: `isosurface`.
//! - **Artificial Intelligence**: `lorahub`, `grpo`.
//! - **Social Science (Satire)**: `favoritism`.
//! - **Quantitative Finance & Decision Theory**: `portfolio`, `game_theory`.
//! - **General Algorithms**: `algorithms`, `win_ratio`.

/// Population dynamics models focusing on intraspecific predation (Cannibalism).
/// Includes McKendrick-von Foerster equations.
pub mod cannibalism;

/// Closed-loop stabilization and UPO control primitives (`upo_control`).
pub mod engineering;

/// A satirical yet rigorously implemented model to calculate a "Favoritism Score"
/// for children based on wealth, social utility, and proximity.
pub mod favoritism;

/// Neuroimaging pipeline for **Cortical Reconstruction**, including surface extraction,
/// thickness measurement, and GLM statistics.
pub mod freesurfer;

/// Statistical methods for comparing outcomes using the Win Ratio approach, common
/// in clinical trials with composite endpoints.
pub mod win_ratio;

/// Modeling of drug absorption, distribution, metabolism, and excretion (ADME)
/// using Bateman functions and multi-dose superposition.
pub mod pharmacokinetics;

/// Logic for merging Low-Rank Adaptation (LoRA) weights for Large Language Models
/// (LLMs), including ensemble composition.
pub mod lorahub;

/// Applied Game Theory, including Mean Field Games and Evolutionary Dynamics.
pub mod game_theory;

pub mod grpo;

/// Algorithms for extracting surfaces from volumetric data, such as Marching Cubes.
pub mod isosurface;

/// Statistical design and analysis for clinical trials, including sample size
/// calculation and survival analysis.
pub mod clinical_trials;

/// General purpose algorithms, including Sorting and other utility structures.
pub mod algorithms;

/// Constrained Markowitz mean-variance portfolio optimization (`GMV`, efficient frontier,
/// maximum-Sharpe tangency, and Capital Allocation Line).
pub mod portfolio;

/// Error types for applied mathematics modules.
pub use crate::error;

// [cite:clinical_trials_statistics]
