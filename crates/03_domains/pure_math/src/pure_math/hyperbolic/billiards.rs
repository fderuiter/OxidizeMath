// [cite:pure_math]
mod physics;
mod twin;

pub use twin::*;

use super::{
    DiskPoint, GeodesicDisk, HyperbolicError, MAX_POLYGON_VERTICES, ideal_polygon_vertices,
    regular_polygon_vertices,
};
use num_complex::Complex64;

/// Maximum number of geodesic segments retained in pre-allocated billiard ring buffers.
pub const MAX_BILLIARD_SEGMENTS: usize = 2048;

/// Convex hyperbolic billiard table geometry in $\overline{\mathbb{D}}$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BilliardDomainKind {
    /// Compact regular `{p, q}` polygon with $(p - 2)(q - 2) > 4$.
    RegularPolygon {
        /// Polygon edge count ($3 \le p \le 16$).
        p: u32,
        /// Vertex valence ($q \ge 3$).
        q: u32,
    },
    /// Ideal regular $n$-gon with vertices on $\partial\mathbb{D}$ ($3 \le n \le 16$).
    IdealPolygon {
        /// Number of ideal sides ($3 \le \text{sides} \le 16$).
        sides: u32,
    },
}

/// Precomputed convex hyperbolic polygon table for zero-allocation ray reflections.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilliardTable {
    kind: BilliardDomainKind,
    vertices: [Complex64; MAX_POLYGON_VERTICES],
    edge_centers: [Complex64; MAX_POLYGON_VERTICES],
    edge_radii: [f64; MAX_POLYGON_VERTICES],
    vertex_count: usize,
}

impl BilliardTable {
    /// Builds a validated [`BilliardTable`] from `kind`.
    #[verified_engine::verified]
    pub fn new(kind: BilliardDomainKind) -> Result<Self, HyperbolicError> {
        let raw_verts = match kind {
            BilliardDomainKind::RegularPolygon { p, q } => {
                if (p as usize) > MAX_POLYGON_VERTICES {
                    return Err(HyperbolicError::InvalidBilliardConfig {
                        reason: "billiard polygon side count exceeds MAX_POLYGON_VERTICES (16)",
                    });
                }
                regular_polygon_vertices(p, q)?
            }
            BilliardDomainKind::IdealPolygon { sides } => ideal_polygon_vertices(sides)?,
        };
        let n = raw_verts.len();
        let mut vertices = [Complex64::new(0.0, 0.0); MAX_POLYGON_VERTICES];
        let mut edge_centers = [Complex64::new(0.0, 0.0); MAX_POLYGON_VERTICES];
        let mut edge_radii = [0.0; MAX_POLYGON_VERTICES];
        for k in 0..n {
            vertices[k] = raw_verts[k];
        }
        for k in 0..n {
            let v0 = vertices[k];
            let v1 = vertices[(k + 1) % n];
            match GeodesicDisk::through_points(v0, v1)? {
                GeodesicDisk::OrthogonalArc { center, radius } => {
                    edge_centers[k] = center;
                    edge_radii[k] = radius;
                }
                GeodesicDisk::Diameter { .. } => {
                    return Err(HyperbolicError::InvalidBilliardConfig {
                        reason: "central billiard table edge unexpectedly passed through origin",
                    });
                }
            }
        }
        assert!((3..=MAX_POLYGON_VERTICES).contains(&n));
        Ok(Self {
            kind,
            vertices,
            edge_centers,
            edge_radii,
            vertex_count: n,
        })
    }

    /// Returns the [`BilliardDomainKind`] of this table.
    #[must_use]
    #[verified_engine::verified]
    pub fn kind(&self) -> BilliardDomainKind {
        debug_assert!(self.vertex_count >= 3);
        self.kind
    }

    /// Fallible companion to [`vertices`](Self::vertices).
    #[verified_engine::verified]
    pub fn try_vertices(
        &self,
    ) -> core::result::Result<&[Complex64], math_commons::error::MathError> {
        if self.vertex_count > MAX_POLYGON_VERTICES {
            return Err(math_commons::error::MathError::NumericalError {
                reason: "vertices: precondition or invariant violated",
            });
        }
        debug_assert!(self.vertex_count <= MAX_POLYGON_VERTICES);
        Ok(&self.vertices[..self.vertex_count])
    }

    /// Returns the active counterclockwise polygon vertices `&[Complex64]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn vertices(&self) -> &[Complex64] {
        self.try_vertices()
            .expect("vertices preconditions must hold")
    }
}

/// Instantaneous state of a unit-speed geodesic billiard particle on $\mathbb{D}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicRay {
    /// Interior position $z \in \mathbb{D}$.
    pub position: DiskPoint,
    /// Unit Euclidean tangent direction $u$ ($|u| = 1$) at `position`.
    pub direction: Complex64,
    /// Index of the most recently struck boundary edge, if any.
    pub last_edge: Option<u8>,
}

impl GeodesicRay {
    /// Constructs a validated [`GeodesicRay`] normalizing `direction`.
    #[verified_engine::verified]
    pub fn new(position: DiskPoint, direction: Complex64) -> Result<Self, HyperbolicError> {
        let norm = direction.re.hypot(direction.im);
        if !norm.is_finite() || norm <= 1.0e-12 {
            return Err(HyperbolicError::InvalidBilliardConfig {
                reason: "ray direction vector must be finite and non-zero",
            });
        }
        let u = Complex64::new(direction.re / norm, direction.im / norm);
        assert!((u.norm_sqr() - 1.0).abs() < 1.0e-10);
        Ok(Self {
            position,
            direction: u,
            last_edge: None,
        })
    }
}

/// Single chordal geodesic flight segment between billiard boundary impacts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicSegment {
    /// Segment start coordinate in $\mathbb{D}$.
    pub start: DiskPoint,
    /// Segment end coordinate in $\mathbb{D}$.
    pub end: DiskPoint,
    /// Struck boundary edge index (`0..vertex_count`).
    pub hit_edge: u8,
    /// Hyperbolic arc length $\Delta s > 0$ traversed along this segment.
    pub hyperbolic_length: f64,
}

impl Default for GeodesicSegment {
    #[verified_engine::verified]
    fn default() -> Self {
        let seg = Self {
            start: DiskPoint::origin(),
            end: DiskPoint::origin(),
            hit_edge: 0,
            hyperbolic_length: 0.0,
        };
        debug_assert_eq!(seg.hyperbolic_length, 0.0);
        seg
    }
}
