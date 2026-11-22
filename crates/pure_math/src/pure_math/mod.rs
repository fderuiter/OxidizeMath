//! # Pure Mathematics
//!
//! Foundational algorithms and structures from abstract mathematics.
//!
//! Unlike the `applied` or `physics` modules, which focus on modeling real-world phenomena,
//! this module provides the rigorous mathematical building blocks. It covers fields from
//! Number Theory to Differential Geometry.
//!
//! ##  Quick Start: Prime Check
//!
//! Check if a number is prime using optimized algorithms.
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::number_theory;
//!
//! // Check a small prime
//! assert!(number_theory::is_prime(17));
//!
//! // Check a composite number
//! assert!(!number_theory::is_prime(20));
//!
//! // Check a larger prime
//! assert!(number_theory::is_prime(1_000_000_007));
//! ```
//!
//! ## Modules
//!
//! *   **[Algebra](algebra)**: Abstract algebraic structures (Groups, Rings, Fields).
//! *   **[Algorithmic Information](algorithmic_information)**: Kolmogorov complexity and information theory.
//! *   **[Analysis](analysis)**: Calculus, Integration, and Differential Equations.
//! *   **[Differential Geometry](differential_geometry)**: Manifolds and curvature.
//! *   **[Elliptic Curves](elliptic_curves)**: Cryptography primitives and geometric arithmetic.
//! *   **[Graph Theory](graph_theory)**: Network analysis and traversal algorithms.
//! *   **[Number Theory](number_theory)**: Primes, Partitions, and Modular Arithmetic.
//! *   **[Statistics](statistics)**: Competitive modeling (Glicko-2), Stochastic processes (Markov), and Data Analysis.
//!
//! ## Philosophy
//!
//! Implementations prioritize *correctness* and *readability* over raw performance optimization,
//! serving as reference implementations for educational purposes.

pub mod algebra;
pub mod algorithmic_information;
#[allow(missing_docs)]
pub mod analysis;
#[allow(missing_docs)]
pub mod differential_geometry;
pub mod elliptic_curves;
#[allow(missing_docs)]
pub mod graph_theory;
pub mod logic;
pub mod number_theory;
pub mod special_functions;
pub mod statistics;
pub mod tensor;
#[allow(missing_docs)]
pub mod vector_calculus;

/// Error types for pure mathematics modules.
pub use crate::error;

// [cite:graph_parameters_rust]

crate::theory_verification!(module = "pure_math", constants = {}, test = {});
