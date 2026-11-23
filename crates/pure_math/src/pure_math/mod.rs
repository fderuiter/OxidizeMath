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
//! *   **[Analysis](analysis)**: Calculus, Complex Riemann Surfaces, Conformal Mapping, Cauchy Integration, and Differential Equations.
//! *   **[Differential Geometry](differential_geometry)**: Parametric manifolds, curvature, Laplace-Beltrami, and zero-allocation flat-buffer geometric flows (`MeanCurvatureFlow`, `HeatEquationSolver`).
//! *   **[Elliptic Curves](elliptic_curves)**: Cryptography primitives and geometric arithmetic.
//! *   **[Graph Theory](graph_theory)**: Network analysis and traversal algorithms.
//! *   **[Hyperbolic Geometry](hyperbolic)**: Poincaré disk/half-plane models, Möbius isometries, `{p,q}` tilings, and geodesic billiards.
//! *   **[Number Theory](number_theory)**: Primes, Partitions, and Modular Arithmetic.
//! *   **[Statistics](statistics)**: Competitive modeling (Glicko-2), Stochastic processes (Markov), and Data Analysis.
//! *   **[Vector Calculus](vector_calculus)**: Curvilinear differential operators, Gauss/Stokes integral theorems, and zero-allocation 3D Helmholtz-Hodge decomposition.
//!
//! ## Curvilinear Stokes & Multi-Sheet Riemann Monodromy
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::nalgebra::Vector3;
//! use math_explorer::num_complex::Complex64;
//! use math_explorer::pure_math::analysis::complex::surfaces::{NthRootSurface, RiemannSurface};
//! use math_explorer::pure_math::vector_calculus::coordinates::Cartesian;
//! use math_explorer::pure_math::vector_calculus::theorems::{StokesPatch, verify_stokes_theorem};
//!
//! let patch = StokesPatch::new(2, 0.0, (-1.0, 1.0), (-1.0, 1.0));
//! let (_, _, diff) = verify_stokes_theorem(&Cartesian, &patch, |p| Vector3::new(-p[1], p[0], 0.0), 16);
//! assert!(diff < 1e-10);
//!
//! let sqrt_surf = NthRootSurface::new(2);
//! assert_eq!(sqrt_surf.next_sheet(0, 1), 1);
//! assert!((sqrt_surf.evaluate(Complex64::new(4.0, 0.0), 1).re + 2.0).abs() < 1e-12);
//! ```
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
pub mod hyperbolic;
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
