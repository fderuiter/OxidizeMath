#![doc = include_str!("../README.md")]
//!
//! # Module Summary
//!
//! - [`theory`]: Defines the [`TheoryDescribable`](theory::TheoryDescribable) trait
//!   and [`ParameterConstraint`](theory::ParameterConstraint) struct for model metadata,
//!   phonetic accessibility descriptions, citations, and tunable parameter ranges.
//! - [`citation_registry`]: Provides [`CitationRegistry`](citation_registry::CitationRegistry),
//!   a thread-safe global registry for storing and querying academic citations.

#![warn(missing_docs)]

/// Centralized academic citation registry module.
pub mod citation_registry;

/// Theory description traits and parameter constraint contracts module.
pub mod theory;
