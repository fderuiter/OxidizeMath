//! # Tensor Calculus
//!
//! This module provides tools for performing calculations on manifolds equipped with a metric tensor.
//! It supports the computation of fundamental geometric objects such as Christoffel symbols,
//! which are essential for General Relativity and Differential Geometry.
//!
//! ##  Quick Start: Christoffel Symbols
//!
//! Compute the Christoffel symbols for a 2D sphere.
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::tensor::{christoffel_symbols, RiemannianMetric};
//! use math_explorer::nalgebra::{DMatrix, DVector};
//!
//! // 1. Define the Metric for a Sphere (Radius = 1.0)
//! // g_theta_theta = 1, g_phi_phi = sin^2(theta)
//! let metric = RiemannianMetric::new(|p: &DVector<f64>| {
//!     let theta = p[0];
//!     let g11 = 1.0;
//!     let g22 = theta.sin().powi(2);
//!     DMatrix::from_vec(2, 2, vec![g11, 0.0, 0.0, g22])
//! });
//!
//! // 2. Define a point (theta = 45 degrees, phi = 0)
//! let point = DVector::from_vec(vec![std::f64::consts::FRAC_PI_4, 0.0]);
//!
//! // 3. Compute Symbols
//! let gammas = christoffel_symbols(&metric, &point).expect("Singular metric");
//!
//! // Check \Gamma^theta_phi_phi = -sin(theta)cos(theta)
//! // -sin(45)cos(45) = -0.5
//! let gamma_theta_phi_phi = gammas[0][(1, 1)];
//! println!("Gamma^theta_phi_phi: {:.4}", gamma_theta_phi_phi);
//! assert!((gamma_theta_phi_phi + 0.5).abs() < 1e-4);
//! ```

#[allow(missing_docs)]
pub mod christoffel;
#[allow(missing_docs)]
pub mod differentiation;
pub mod geodesic;
pub mod lorentzian;
#[allow(missing_docs)]
pub mod metric;
pub mod static_metric;
pub mod types;

// Re-exports for convenience
pub use christoffel::christoffel_symbols;
pub use geodesic::{GeodesicState, GeodesicStepStatus, GeodesicSystem};
pub use lorentzian::{LorentzianConnection4D, SchwarzschildConnection4D};
pub use metric::{Metric, RiemannianMetric};
pub use static_metric::{
    StaticMetric, StaticRiemannianMetric, TwoSphereMetric, static_christoffel_symbols,
};
pub use types::{ContravariantVector, CovariantVector, TensorError};

// [cite:graph_parameters_rust]
// Note: `papers/tensors.tex` proves SU(2) Clebsch-Gordan coupling of irreducible spherical
// tensor operators rather than Lorentzian metric connections; curved-spacetime Christoffel
// symbols and geodesic equations are derived in `papers/general_relativity_geodesics.tex`.
// [cite:tensors]
// [cite:general_relativity_geodesics]

#[cfg(test)]
mod tests;
