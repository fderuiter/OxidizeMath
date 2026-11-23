//! Zero-allocation Flat Euclidean Dynamical Billiards (`domain_physics::physics::chaos::billiards`).
//!
//! Simulates exact specular ray reflections ($\mathbf{v}' = \mathbf{v} - 2(\mathbf{v}\cdot\hat{\mathbf{n}})\hat{\mathbf{n}}$)
//! inside flat ($K = 0$) planar Euclidean domains and projects collisions onto canonical
//! Birkhoff phase-space coordinates $(s / L, \sin\phi) \in [0, 1) \times [-1, 1]$:
//! - [`CircularBilliard`]: Integrable disk conserving signed angular momentum $R \sin\phi$.
//! - [`EllipticalBilliard`]: Integrable confocal ellipse conserving Berry's focal product $L_1 L_2$.
//! - [`StadiumBilliard`]: Bunimovich defocusing stadium exhibiting K-system ergodicity.
//! - [`SinaiBilliard`]: Dispersing square table with central circular scatterer (hyperbolic chaos).
//!
//! # Example
//!
//! ```
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::chaos::billiards::{BilliardRay, EllipticalBilliard};
//!
//! let ellipse = EllipticalBilliard::new(1.4, 1.0);
//! let mut ray = BilliardRay::from_angle([0.2, 0.15], 0.73).unwrap();
//! let i0 = ellipse.focal_angular_momentum_product(&ray);
//! for _ in 0..100 {
//!     let hit = ellipse.collide(&mut ray).unwrap();
//!     assert!((0.0..1.0).contains(&hit.arc_fraction));
//!     assert!((-1.0..=1.0).contains(&hit.sin_phi));
//! }
//! assert!((ray.speed() - 1.0).abs() < 1e-14);
//! assert!((ellipse.focal_angular_momentum_product(&ray) - i0).abs() < 1e-12);
//! ```

// [cite:chaos_theory]

pub mod ellipse;
pub mod sinai;
pub mod tables;

#[cfg(test)]
mod tests;

pub use ellipse::EllipticalBilliard;
pub use sinai::SinaiBilliard;
pub use tables::{CircularBilliard, StadiumBilliard};

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;
use std::f64::consts::TAU;

pub(crate) const HIT_EPS: f64 = 1e-11;

/// Unit-speed planar Euclidean ray state $(\mathbf{r}, \mathbf{v})$ with $\|\mathbf{v}\| = 1$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BilliardRay {
    /// Planar Cartesian position $[x, y]$.
    pub pos: [f64; 2],
    /// Unit Cartesian velocity $[v_x, v_y]$ ($\|\mathbf{v}\| = 1$).
    pub vel: [f64; 2],
}

impl BilliardRay {
    /// Constructs a unit-speed [`BilliardRay`] by normalizing a non-zero finite velocity vector.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(pos: [f64; 2], vel: [f64; 2]) -> Option<Self> {
        if !pos[0].is_finite() || !pos[1].is_finite() || !vel[0].is_finite() || !vel[1].is_finite()
        {
            return None;
        }
        let norm = vel[0].hypot(vel[1]);
        if norm <= 1e-15 {
            return None;
        }
        Some(Self {
            pos,
            vel: [vel[0] / norm, vel[1] / norm],
        })
    }

    /// Constructs a unit-speed [`BilliardRay`] launched at polar angle `angle` (radians).
    #[must_use]
    #[verified_engine::verified]
    pub fn from_angle(pos: [f64; 2], angle: f64) -> Option<Self> {
        if !pos[0].is_finite() || !pos[1].is_finite() || !angle.is_finite() {
            return None;
        }
        let (s, c) = angle.sin_cos();
        Some(Self { pos, vel: [c, s] })
    }

    /// Euclidean speed $\|\mathbf{v}\|$, preserved at `1.0` across specular collisions.
    #[must_use]
    #[verified_engine::verified]
    pub fn speed(&self) -> f64 {
        self.vel[0].hypot(self.vel[1])
    }
}

/// Exact boundary impact record with canonical Birkhoff coordinates $(s / L, \sin\phi)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BilliardCollision {
    /// Cartesian boundary impact coordinates $[x, y]$.
    pub hit_pos: [f64; 2],
    /// Outgoing unit velocity $[v_x', v_y']$ after specular reflection.
    pub out_vel: [f64; 2],
    /// Free-flight path length (time at unit speed) between the previous state and `hit_pos`.
    pub flight_time: f64,
    /// Normalized counterclockwise perimeter coordinate $s / L \in [0, 1)$.
    pub arc_fraction: f64,
    /// Birkhoff tangential momentum $\sin\phi = \mathbf{v}_{\text{out}} \cdot \hat{\mathbf{t}} \in [-1, 1]$.
    pub sin_phi: f64,
}

#[verified_engine::verified]
pub(crate) fn reflect_unit(vel: [f64; 2], unit_normal: [f64; 2]) -> [f64; 2] {
    let vn = vel[0] * unit_normal[0] + vel[1] * unit_normal[1];
    let rx = vel[0] - 2.0 * vn * unit_normal[0];
    let ry = vel[1] - 2.0 * vn * unit_normal[1];
    let norm = rx.hypot(ry);
    if norm > 1e-15 {
        [rx / norm, ry / norm]
    } else {
        [-vel[0], -vel[1]]
    }
}

/// Polymorphic flat Euclidean billiard table supporting zero-allocation ray dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BilliardTable {
    /// Integrable circular disk billiard.
    Circle(CircularBilliard),
    /// Integrable confocal elliptical billiard.
    Ellipse(EllipticalBilliard),
    /// Ergodic defocusing Bunimovich stadium billiard.
    Stadium(StadiumBilliard),
    /// Hyperbolic dispersing Sinai billiard (square with central circular scatterer).
    Sinai(SinaiBilliard),
}

impl Default for BilliardTable {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::Ellipse(EllipticalBilliard::default())
    }
}

impl BilliardTable {
    /// Returns `true` if `pos` lies inside the free interior of the billiard table.
    #[must_use]
    #[verified_engine::verified]
    pub fn contains(&self, pos: [f64; 2]) -> bool {
        match self {
            Self::Circle(t) => t.contains(pos),
            Self::Ellipse(t) => t.contains(pos),
            Self::Stadium(t) => t.contains(pos),
            Self::Sinai(t) => t.contains(pos),
        }
    }

    /// Total boundary perimeter $L$ used to normalize Birkhoff's $s / L$ coordinate.
    #[must_use]
    #[verified_engine::verified]
    pub fn perimeter(&self) -> f64 {
        match self {
            Self::Circle(t) => t.perimeter(),
            Self::Ellipse(t) => t.perimeter(),
            Self::Stadium(t) => t.perimeter(),
            Self::Sinai(t) => t.perimeter(),
        }
    }

    /// Advances `ray` in-place to its next exact specular boundary collision.
    #[must_use]
    #[verified_engine::verified]
    pub fn collide(&self, ray: &mut BilliardRay) -> Option<BilliardCollision> {
        match self {
            Self::Circle(t) => t.collide(ray),
            Self::Ellipse(t) => t.collide(ray),
            Self::Stadium(t) => t.collide(ray),
            Self::Sinai(t) => t.collide(ray),
        }
    }

    /// Advances `ray` through up to `out.len()` collisions in-place without heap allocation.
    #[verified_engine::verified]
    pub fn advance_bounces(&self, ray: &mut BilliardRay, out: &mut [BilliardCollision]) -> usize {
        let mut count = 0_usize;
        for slot in out.iter_mut() {
            let Some(hit) = self.collide(ray) else {
                break;
            };
            *slot = hit;
            count += 1;
        }
        count
    }

    /// Seeds a symmetric angular fan of unit-speed rays inside the free interior of the table.
    ///
    /// Returns `0` without mutating `out` if `origin` lies outside the valid free table region.
    #[must_use]
    #[verified_engine::verified]
    pub fn seed_ray_fan(
        &self,
        origin: [f64; 2],
        center_angle: f64,
        angular_spread: f64,
        out: &mut [BilliardRay],
    ) -> usize {
        if out.is_empty()
            || !self.contains(origin)
            || !center_angle.is_finite()
            || !angular_spread.is_finite()
        {
            return 0;
        }
        let spread = angular_spread.abs().clamp(0.0, std::f64::consts::PI);
        let n = out.len();
        let mut written = 0_usize;
        for idx in 0..n {
            let delta = if n == 1 {
                0.0
            } else {
                let frac = (idx as f64) / ((n - 1) as f64) - 0.5;
                2.0 * frac * spread
            };
            if let Some(ray) = BilliardRay::from_angle(origin, center_angle + delta) {
                out[written] = ray;
                written += 1;
            }
        }
        written
    }

    /// Samples the outer boundary wall into `out` in counterclockwise order without heap allocation.
    #[verified_engine::verified]
    pub fn sample_outer_boundary(&self, out: &mut [[f64; 2]]) -> usize {
        let n = out.len();
        if n < 4 {
            return 0;
        }
        match self {
            Self::Circle(c) => {
                for (i, slot) in out.iter_mut().enumerate() {
                    let th = TAU * (i as f64) / (n as f64);
                    *slot = [c.radius * th.cos(), c.radius * th.sin()];
                }
                n
            }
            Self::Ellipse(e) => {
                for (i, slot) in out.iter_mut().enumerate() {
                    let th = TAU * (i as f64) / (n as f64);
                    *slot = [e.semi_major * th.cos(), e.semi_minor * th.sin()];
                }
                n
            }
            Self::Stadium(s) => s.sample_boundary(out),
            Self::Sinai(s) => s.sample_outer_square(out),
        }
    }
}

impl TheoryDescribable for BilliardTable {
    fn theory_description(&self) -> String {
        match self {
            Self::Circle(c) => format!(
                "Integrable circular Euclidean billiard (R={:.2}) conserving signed angular momentum.",
                c.radius
            ),
            Self::Ellipse(e) => format!(
                "Integrable confocal elliptical billiard (a={:.2}, b={:.2}) conserving Berry's focal product L₁·L₂.",
                e.semi_major, e.semi_minor
            ),
            Self::Stadium(s) => format!(
                "Bunimovich defocusing stadium billiard (a={:.2}, R={:.2}) exhibiting ergodic mixing.",
                s.half_straight, s.radius
            ),
            Self::Sinai(s) => format!(
                "Sinai dispersing Euclidean billiard (half_side={:.2}, disk r={:.2}) with hyperbolic defocusing.",
                s.half_side, s.scatterer_radius
            ),
        }
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Birkhoff, G. D. (1927). Dynamical Systems; Sinai, Y. G. (1970). Russian Mathematical Surveys, 25(2), 137-189; Bunimovich, L. A. (1979). Communications in Mathematical Physics, 65(3), 295-312; Berry, M. V. (1981). European Journal of Physics, 2(2), 91-102.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Flat Euclidean dynamical billiards and Birkhoff phase-space maps".to_string(),
        );
        map
    }
}
