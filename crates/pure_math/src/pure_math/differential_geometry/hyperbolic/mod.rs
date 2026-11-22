//! Intrinsic two-dimensional hyperbolic geometry ($K = -1$) modeled via $\mathrm{SU}(1,1)$
//! Möbius transformations on the conformal Poincaré disk $\mathbb{D}$ and projective
//! Beltrami-Klein disk $\mathbb{K}$.
//!
//! Unlike extrinsic 3D embeddings of constant negative curvature, the pseudosphere encounters
//! Hilbert's singularity at its rim. Working directly inside $\mathbb{D} = \{z \in \mathbb{C} : |z| < 1\}$
//! provides global completeness, exact conformal angles, Fuchsian $\{p, q\}$ tessellations, and
//! closed-form geodesic billiard dynamics.
//!
//! # Example
//!
//! ```
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::num_complex::Complex64;
//! use math_explorer::pure_math::differential_geometry::hyperbolic::{
//!     Su11Matrix, gauss_bonnet_triangle, klein_to_poincare, poincare_distance, poincare_to_klein,
//! };
//!
//! let z = Complex64::new(0.3, -0.2);
//! let w = poincare_to_klein(z);
//! let round_trip = klein_to_poincare(w);
//! assert!((round_trip - z).norm() < 1e-12);
//!
//! let pan = Su11Matrix::hyperbolic_pan(Complex64::new(0.4, 0.25));
//! let a = Complex64::new(0.0, 0.0);
//! let b = Complex64::new(0.35, 0.0);
//! let c = Complex64::new(0.1, 0.32);
//! let d_before = poincare_distance(a, b);
//! let d_after = poincare_distance(pan.apply(a), pan.apply(b));
//! assert!((d_before - d_after).abs() < 1e-12);
//!
//! let gb = gauss_bonnet_triangle(a, b, c);
//! assert!(gb.angle_sum < std::f64::consts::PI);
//! assert!((gb.area - gb.angular_defect).abs() < 1e-12);
//! ```

use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;

/// Exact analytical geodesic billiards and finite-time Lyapunov divergence inside $\{p, q\}$ polygons.
pub mod billiards;
mod billiards_kernel;
/// Intrinsic Poincaré and Klein metrics, geodesic arcs, conformal angles, and Gauss-Bonnet defects.
pub mod metrics;
/// Orientation-preserving $\mathrm{SU}(1,1)$ Möbius disk isometries and model projections.
pub mod mobius;
/// Zero-allocation Fuchsian $\{p, q\}$ hyperbolic tessellations and fundamental-domain re-centering.
pub mod tiling;

#[cfg(test)]
mod tests;

pub use billiards::{
    BilliardParticle, BilliardWall, HyperbolicBilliardConfig, HyperbolicBilliardSystem,
    LyapunovSample, circle_inversion,
};
pub use metrics::{
    GaussBonnetSummary, GeodesicArc, conformal_metric_factor, gauss_bonnet_triangle,
    gaussian_curvature, geodesic_interior_angle, intrinsic_gaussian_curvature, klein_distance,
    poincare_distance, regular_polygon_gauss_bonnet_area,
};
pub use mobius::{
    Su11Matrix, clamp_to_disk, klein_to_poincare, poincare_to_klein, poincare_to_upper_half_plane,
    project_disk_point, unproject_disk_point, upper_half_plane_to_poincare,
};
pub use tiling::{
    HyperbolicPolygon, HyperbolicTilingBuffer, is_hyperbolic_schlafli, regular_polygon_inradius,
    regular_polygon_poincare_radius,
};

/// Strict maximum Euclidean norm permitted inside the open unit disk ($1 - 10^{-9}$).
pub const POINCARE_DISK_RADIUS_MAX: f64 = 1.0 - 1e-9;

/// Constant intrinsic Gaussian curvature of the unit hyperbolic plane $\mathbb{H}^2$.
pub const HYPERBOLIC_CURVATURE: f64 = -1.0;

/// Maximum polygon vertex count supported in inline stack-allocated tile buffers.
pub const MAX_POLYGON_SIDES: usize = 16;

/// Hard upper bound on polygons stored in a [`HyperbolicTilingBuffer`] (well below `16,384`).
pub const MAX_TILING_POLYGONS: usize = 512;

/// Maximum BFS reflection/rotation layer depth for interactive $\{p, q\}$ tilings.
pub const MAX_TILING_DEPTH: usize = 4;

/// Default ring-buffer capacity for geodesic billiard trajectories and Lyapunov traces.
pub const MAX_BILLIARD_HISTORY: usize = 2048;

/// Disk model projection used to visualize and export intrinsic $\mathbb{H}^2$ geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum HyperbolicModel {
    /// Conformal Poincaré disk $\mathbb{D}$: preserves angles; geodesics are orthogonal circular arcs.
    #[default]
    Poincare,
    /// Projective Beltrami-Klein disk $\mathbb{K}$: geodesics are Euclidean straight chords.
    Klein,
}

/// Domain error returned when constructing invalid hyperbolic configurations or matrices.
#[derive(Debug, Clone, PartialEq)]
pub enum HyperbolicError {
    /// Matrix entries violate finiteness or positive $\mathrm{SU}(1,1)$ pseudo-determinant $|a|^2 - |b|^2 > 0$.
    DegenerateSu11Matrix {
        /// Computed $|a|^2 - |b|^2$ determinant value.
        determinant: f64,
    },
    /// Schläfli symbol $\{p, q\}$ does not satisfy the hyperbolic condition $(p - 2)(q - 2) > 4$.
    NonHyperbolicSchlafli {
        /// Polygon side count $p$.
        p: usize,
        /// Vertex valence $q$.
        q: usize,
    },
    /// Point lies outside the open upper half-plane $\operatorname{Im}(\tau) > 0$ or unit disk.
    InvalidDomainPoint {
        /// Explanation of the violated geometric domain constraint.
        reason: &'static str,
    },
    /// Initial billiard state lies outside the fundamental $\{p, q\}$ polygon or has invalid $\epsilon$.
    InvalidBilliardConfig {
        /// Explanation of the invalid billiard parameter.
        reason: &'static str,
    },
}

impl std::fmt::Display for HyperbolicError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DegenerateSu11Matrix { determinant } => {
                write!(
                    f,
                    "degenerate SU(1,1) matrix: |a|^2 - |b|^2 = {determinant} (must be finite and > 0)"
                )
            }
            Self::NonHyperbolicSchlafli { p, q } => {
                write!(
                    f,
                    "Schlafli symbol {{{p}, {q}}} is not hyperbolic: requires 3 <= p,q <= {MAX_POLYGON_SIDES} and (p - 2)*(q - 2) > 4"
                )
            }
            Self::InvalidDomainPoint { reason } => write!(f, "invalid hyperbolic point: {reason}"),
            Self::InvalidBilliardConfig { reason } => {
                write!(f, "invalid hyperbolic billiard configuration: {reason}")
            }
        }
    }
}

impl std::error::Error for HyperbolicError {}

impl Diagnostic for HyperbolicError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "HyperbolicError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

// [cite:modular_polynomials_review]
