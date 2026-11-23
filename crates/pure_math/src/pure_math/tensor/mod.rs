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
//! use math_explorer::engine_symbolic::Dual;
//! use math_explorer::pure_math::tensor::{christoffel_symbols, DualRiemannianMetric};
//! use math_explorer::nalgebra::{DMatrix, DVector};
//!
//! // 1. Define the Metric for a Sphere (Radius = 1.0)
//! // g_theta_theta = 1, g_phi_phi = sin^2(theta)
//! let metric = DualRiemannianMetric::new(|p: &[Dual<f64>]| {
//!     let theta = p[0];
//!     let sin_t = theta.sin();
//!     DMatrix::from_vec(2, 2, vec![Dual::ONE, Dual::ZERO, Dual::ZERO, sin_t * sin_t])
//! });
//!
//! // 2. Define a point (theta = 45 degrees, phi = 0)
//! let point = DVector::from_vec(vec![std::f64::consts::FRAC_PI_4, 0.0]);
//!
//! // 3. Compute Symbols
//! let gammas = christoffel_symbols(&metric, &point).expect("Singular metric");
//!
//! // Check \Gamma^theta_phi_phi = -sin(theta)cos(theta) = -0.5 to machine precision
//! let gamma_theta_phi_phi = gammas[0][(1, 1)];
//! assert!((gamma_theta_phi_phi + 0.5).abs() < 1e-14);
//! ```

#[allow(missing_docs)]
pub mod christoffel;
#[allow(missing_docs)]
pub mod differentiation;
pub mod geodesic;
#[allow(missing_docs)]
pub mod metric;
pub mod static_metric;
pub mod types;

// Re-exports for convenience
pub use christoffel::{christoffel_symbols, christoffel_symbols_fixed};
pub use geodesic::{GeodesicState, GeodesicStepStatus, GeodesicSystem};
pub use metric::{DualRiemannianMetric, Metric, RiemannianMetric};
pub use static_metric::{
    DualStaticRiemannianMetric, StaticMetric, StaticRiemannianMetric, TwoSphereMetric,
    static_christoffel_symbols,
};
pub use types::{ContravariantVector, CovariantVector, TensorError};

// [cite:graph_parameters_rust]
// [cite:tensors]
// [cite:general_relativity_geodesics]

#[cfg(test)]
mod tests;
