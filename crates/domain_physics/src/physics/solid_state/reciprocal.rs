//! Periodic 2D Bravais lattices, Wigner-Seitz Brillouin zones, and zero-allocation 2D Bragg FFTs.
//!
//! Computes primitive and reciprocal basis vectors ($\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi\delta_{ij}$)
//! and exact First Brillouin Zone polygons for all five 2D Bravais nets, alongside a pre-planned
//! `rustfft` 2D diffraction engine and rotational correlation analyzer ($C_4 \dots C_{12}$).
//!
//! # Examples
//!
//! ```
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::solid_state::reciprocal::{BravaisLattice2D, Lattice2DParams};
//! use std::f64::consts::PI;
//!
//! let hex = Lattice2DParams::hexagonal(2.0).expect("valid hexagonal lattice");
//! let [a1, a2] = hex.direct_basis();
//! let [b1, b2] = hex.reciprocal_basis();
//! assert!((a1.dot(&b1) - 2.0 * PI).abs() < 1e-12);
//! assert!(a1.dot(&b2).abs() < 1e-12);
//! let bz = hex.first_brillouin_zone();
//! assert_eq!(bz.len(), 6);
//! ```

// [cite:solid_state]

#[path = "reciprocal/diffraction.rs"]
mod diffraction;

pub use diffraction::{
    BraggDiffractionWorkspace, BraggPeak2D, RotationalSymmetryReport, kinematic_structure_factor,
};

use crate::error::SolidStateError;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI};

const MAX_BZ_VERTICES: usize = 32;

/// All five crystallographic 2D Bravais lattice types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BravaisLattice2D {
    /// General oblique lattice ($a \ne b$, $\gamma \ne 90^\circ$, 2-fold point symmetry $C_2$).
    Oblique,
    /// Primitive rectangular lattice ($a \ne b$, $\gamma = 90^\circ$, $D_2$ point symmetry).
    Rectangular,
    /// Centered rectangular (rhombic primitive) lattice ($|\mathbf{a}_1| = |\mathbf{a}_2|$, $D_2$ symmetry).
    CenteredRectangular,
    /// Square lattice ($a = b$, $\gamma = 90^\circ$, 4-fold point symmetry $D_4$).
    #[default]
    Square,
    /// Hexagonal / triangular lattice ($a = b$, $\gamma = 60^\circ$, 6-fold point symmetry $D_6$).
    Hexagonal,
}

impl BravaisLattice2D {
    /// Human-readable crystallographic net name.
    #[must_use]
    #[verified_engine::verified]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Oblique => "Oblique (p2)",
            Self::Rectangular => "Rectangular (pmm)",
            Self::CenteredRectangular => "Centered Rectangular (cmm)",
            Self::Square => "Square (p4m)",
            Self::Hexagonal => "Hexagonal (p6m)",
        }
    }

    /// Expected crystallographic rotational symmetry order $n \in \{2, 4, 6\}$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn symmetry_order(self) -> usize {
        match self {
            Self::Oblique | Self::Rectangular | Self::CenteredRectangular => 2,
            Self::Square => 4,
            Self::Hexagonal => 6,
        }
    }
}

/// Validated 2D Bravais lattice parameters $(a, b, \gamma)$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Lattice2DParams {
    /// Selected 2D Bravais family.
    pub lattice: BravaisLattice2D,
    /// Primary lattice constant $a > 0$.
    pub a: f64,
    /// Secondary lattice constant $b > 0$ (enforced equal to $a$ for `Square` and `Hexagonal`).
    pub b: f64,
    /// Inter-axial angle $\gamma \in (0, \pi)$ in radians.
    pub gamma_rad: f64,
}

impl Lattice2DParams {
    /// Constructs and normalizes 2D Bravais parameters according to symmetry constraints.
    #[verified_engine::verified]
    pub fn new(
        lattice: BravaisLattice2D,
        a: f64,
        b: f64,
        gamma_rad: f64,
    ) -> Result<Self, SolidStateError> {
        if !a.is_finite() || a <= 0.0 || !b.is_finite() || b <= 0.0 {
            return Err(SolidStateError::InvalidParameter(
                "lattice constants a and b must be finite and positive".into(),
            ));
        }
        if !gamma_rad.is_finite() || gamma_rad <= 0.12 || gamma_rad >= PI - 0.12 {
            return Err(SolidStateError::InvalidParameter(
                "inter-axial angle gamma_rad must lie strictly inside (0.12, pi - 0.12)".into(),
            ));
        }
        let (norm_b, norm_gamma) = match lattice {
            BravaisLattice2D::Square => (a, FRAC_PI_2),
            BravaisLattice2D::Hexagonal => (a, FRAC_PI_3),
            BravaisLattice2D::Rectangular | BravaisLattice2D::CenteredRectangular => (b, FRAC_PI_2),
            BravaisLattice2D::Oblique => (b, gamma_rad),
        };
        Ok(Self {
            lattice,
            a,
            b: norm_b,
            gamma_rad: norm_gamma,
        })
    }

    /// Convenience constructor for a 2D square lattice ($b = a$, $\gamma = \pi/2$).
    #[verified_engine::verified]
    pub fn square(a: f64) -> Result<Self, SolidStateError> {
        Self::new(BravaisLattice2D::Square, a, a, FRAC_PI_2)
    }

    /// Convenience constructor for a 2D hexagonal lattice ($b = a$, $\gamma = \pi/3$).
    #[verified_engine::verified]
    pub fn hexagonal(a: f64) -> Result<Self, SolidStateError> {
        Self::new(BravaisLattice2D::Hexagonal, a, a, FRAC_PI_3)
    }

    /// Primitive direct-space basis vectors $[\mathbf{a}_1, \mathbf{a}_2]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn direct_basis(&self) -> [Vector2<f64>; 2] {
        match self.lattice {
            BravaisLattice2D::Square => [Vector2::new(self.a, 0.0), Vector2::new(0.0, self.a)],
            BravaisLattice2D::Hexagonal => [
                Vector2::new(self.a, 0.0),
                Vector2::new(0.5 * self.a, 0.5 * 3.0_f64.sqrt() * self.a),
            ],
            BravaisLattice2D::Rectangular => [Vector2::new(self.a, 0.0), Vector2::new(0.0, self.b)],
            BravaisLattice2D::CenteredRectangular => [
                Vector2::new(0.5 * self.a, 0.5 * self.b),
                Vector2::new(0.5 * self.a, -0.5 * self.b),
            ],
            BravaisLattice2D::Oblique => [
                Vector2::new(self.a, 0.0),
                Vector2::new(self.b * self.gamma_rad.cos(), self.b * self.gamma_rad.sin()),
            ],
        }
    }

    /// Signed primitive unit-cell determinant $\det(\mathbf{a}_1, \mathbf{a}_2)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn signed_cell_area(&self) -> f64 {
        let [a1, a2] = self.direct_basis();
        a1.x * a2.y - a1.y * a2.x
    }

    /// Primitive unit-cell area $|\Omega| = |\mathbf{a}_1 \times \mathbf{a}_2|$.
    #[must_use]
    #[verified_engine::verified]
    pub fn unit_cell_area(&self) -> f64 {
        self.signed_cell_area().abs()
    }

    /// Reciprocal basis vectors $[\mathbf{b}_1, \mathbf{b}_2]$ satisfying $\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi\delta_{ij}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn reciprocal_basis(&self) -> [Vector2<f64>; 2] {
        let [a1, a2] = self.direct_basis();
        let scale = 2.0 * PI / (a1.x * a2.y - a1.y * a2.x);
        [
            Vector2::new(scale * a2.y, -scale * a2.x),
            Vector2::new(-scale * a1.y, scale * a1.x),
        ]
    }

    /// Computes the exact Wigner-Seitz First Brillouin Zone polygon around $\mathbf{k} = \mathbf{0}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn first_brillouin_zone(&self) -> BrillouinZone2D {
        let [b1, b2] = self.reciprocal_basis();
        let bound = 4.0 * (b1.norm() + b2.norm());
        let mut poly = [[0.0_f64; 2]; MAX_BZ_VERTICES];
        poly[0] = [-bound, -bound];
        poly[1] = [bound, -bound];
        poly[2] = [bound, bound];
        poly[3] = [-bound, bound];
        let mut count = 4_usize;

        for m1 in -3_i32..=3 {
            for m2 in -3_i32..=3 {
                if m1 == 0 && m2 == 0 {
                    continue;
                }
                let g = f64::from(m1) * b1 + f64::from(m2) * b2;
                count = clip_halfplane(&mut poly, count, [g.x, g.y], 0.5 * g.norm_squared());
            }
        }
        count = dedup_polygon(&mut poly, count);
        BrillouinZone2D {
            vertices: poly,
            len: count,
        }
    }

    /// Populates `out` with periodic lattice sites inside `aperture_radius` with zero allocations.
    #[verified_engine::verified]
    pub fn generate_sites_into(&self, aperture_radius: f64, out: &mut Vec<[f64; 2]>) {
        out.clear();
        if !aperture_radius.is_finite() || aperture_radius <= 0.0 {
            return;
        }
        let [a1, a2] = self.direct_basis();
        let det_inv = 1.0 / (a1.x * a2.y - a1.y * a2.x).abs().max(1e-12);
        let m1_bound = ((aperture_radius * a2.norm() * det_inv).ceil() as i32) + 2;
        let m2_bound = ((aperture_radius * a1.norm() * det_inv).ceil() as i32) + 2;
        let r_sq = aperture_radius * aperture_radius;

        for m1 in -m1_bound..=m1_bound {
            for m2 in -m2_bound..=m2_bound {
                let p = f64::from(m1) * a1 + f64::from(m2) * a2;
                if p.norm_squared() <= r_sq && out.len() < out.capacity() {
                    out.push([p.x, p.y]);
                }
            }
        }
    }
}

/// Stack-allocated convex First Brillouin Zone polygon in reciprocal space $(k_x, k_y)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrillouinZone2D {
    vertices: [[f64; 2]; MAX_BZ_VERTICES],
    len: usize,
}

impl BrillouinZone2D {
    /// Active counter-clockwise Brillouin-zone corner coordinates.
    #[must_use]
    #[verified_engine::verified]
    pub fn vertices(&self) -> &[[f64; 2]] {
        &self.vertices[..self.len]
    }

    /// Number of polygon vertices (`4` for rectangular/square, `6` for hexagonal/oblique/centered).
    #[must_use]
    #[verified_engine::verified]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the polygon is empty.
    #[must_use]
    #[verified_engine::verified]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Exact reciprocal-space area $\frac{(2\pi)^2}{|\Omega|}$ computed via the shoelace formula.
    #[must_use]
    #[verified_engine::verified]
    pub fn area(&self) -> f64 {
        if self.len < 3 {
            return 0.0;
        }
        let mut twice_area = 0.0;
        for i in 0..self.len {
            let p1 = self.vertices[i];
            let p2 = self.vertices[(i + 1) % self.len];
            twice_area += p1[0] * p2[1] - p2[0] * p1[1];
        }
        0.5 * twice_area.abs()
    }
}

#[verified_engine::verified]
fn clip_halfplane(
    poly: &mut [[f64; 2]; MAX_BZ_VERTICES],
    count: usize,
    normal: [f64; 2],
    rhs: f64,
) -> usize {
    if count == 0 {
        return 0;
    }
    let mut out = [[0.0_f64; 2]; MAX_BZ_VERTICES];
    let mut out_len = 0_usize;
    for i in 0..count {
        let p1 = poly[i];
        let p2 = poly[(i + 1) % count];
        let d1 = p1[0] * normal[0] + p1[1] * normal[1] - rhs;
        let d2 = p2[0] * normal[0] + p2[1] * normal[1] - rhs;
        if d1 <= 1e-11 {
            if out_len < MAX_BZ_VERTICES {
                out[out_len] = p1;
                out_len += 1;
            }
            if d2 > 1e-11 && out_len < MAX_BZ_VERTICES {
                let t = d1 / (d1 - d2);
                out[out_len] = [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
                out_len += 1;
            }
        } else if d2 <= 1e-11 && out_len < MAX_BZ_VERTICES {
            let t = d1 / (d1 - d2);
            out[out_len] = [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
            out_len += 1;
        }
    }
    *poly = out;
    out_len
}

#[verified_engine::verified]
fn dedup_polygon(poly: &mut [[f64; 2]; MAX_BZ_VERTICES], count: usize) -> usize {
    if count < 2 {
        return count;
    }
    let mut clean = [[0.0_f64; 2]; MAX_BZ_VERTICES];
    let mut len = 0_usize;
    for i in 0..count {
        let p = poly[i];
        let prev = if len == 0 {
            poly[count - 1]
        } else {
            clean[len - 1]
        };
        if (p[0] - prev[0]).powi(2) + (p[1] - prev[1]).powi(2) > 1e-14 {
            clean[len] = p;
            len += 1;
        }
    }
    *poly = clean;
    len
}
