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
//! *   **[Hyperbolic Geometry](hyperbolic)**: Poincaré disk/half-plane models, Möbius isometries, `{p,q}` tilings, and geodesic billiards.
//! *   **[Knot Theory](knot_theory)**: 3D knots, Artin braid closures, Reidemeister moves, and Laurent skein polynomials.
//! *   **[Number Theory](number_theory)**: Primes, Partitions, and Modular Arithmetic.
//! *   **[Statistics](statistics)**: Competitive modeling (Glicko-2), Stochastic processes (Markov), and Data Analysis.
//!
//! ## Philosophy
//!
//! Implementations prioritize *correctness* and *readability* over raw performance optimization,
//! serving as reference implementations for educational purposes.

pub mod algebra;
pub mod algorithmic_information;
/// Submodule providing `analysis` types, traits, and algorithms.
pub mod analysis;
/// Submodule providing `differential_geometry` types, traits, and algorithms.
pub mod differential_geometry;
pub mod elliptic_curves;
/// Submodule providing `graph_theory` types, traits, and algorithms.
pub mod graph_theory;
pub mod hyperbolic;
pub mod knot_theory;
pub mod logic;
pub mod number_theory;
pub mod special_functions;
pub mod statistics;
pub mod tensor;
/// Submodule providing `vector_calculus` types, traits, and algorithms.
pub mod vector_calculus;

/// Error types for pure mathematics modules.
pub use crate::error;

// [cite:graph_parameters_rust]

crate::theory_verification!(
    module = "pure_math",
    constants = {
        MERSENNE_SEVEN = 127.0;
        HARMONIC_OMEGA_SQ = 1.0;
    },
    test = {
        use analysis::ode::OdeModel;
        use scientific_metadata::theory::{MaturityTier, TheoryDescribable};

        assert!(number_theory::is_prime(MERSENNE_SEVEN as u64));
        assert!(!number_theory::is_prime(128));
        let model = OdeModel::new(HARMONIC_OMEGA_SQ, (), ());
        assert_eq!(model.maturity_tier(), MaturityTier::Certified);
        assert!(!model.theory_description().is_empty());
    }
);
