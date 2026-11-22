//! # Applied Mathematics
//!
//! This module serves as a collection of mathematical models applied to specific,
//! often niche or complex, domains. It demonstrates how core mathematical concepts
//! (calculus, statistics, game theory) translate into practical solutions.
//!
//! ## Domains
//!
//! - **Biology & Medicine**: `clinical_trials`, `pharmacokinetics`, `freesurfer`, `cannibalism`.
//! - **Physics & Engineering**: `battery_degradation`, `isosurface`.
//! - **Artificial Intelligence**: `lorahub`, `grpo`.
//! - **Social Science (Satire)**: `favoritism`.
//! - **General Algorithms**: `algorithms`, `win_ratio`.

/// Modeling of battery health and capacity fade over time.
pub mod battery_degradation;

/// Population dynamics models focusing on intraspecific predation (Cannibalism).
/// Includes McKendrick-von Foerster equations.
pub mod cannibalism;

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
pub mod engineering;

/// Error types for applied mathematics modules.
pub use crate::error;

// [cite:graph_parameters_rust]
