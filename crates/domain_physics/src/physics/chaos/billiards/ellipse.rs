//! Confocal Elliptical Euclidean Billiard with exact conserved focal angular momentum product.

// [cite:chaos_theory]

use super::{BilliardCollision, BilliardRay, HIT_EPS, reflect_unit};
use std::f64::consts::{FRAC_PI_2, TAU};

const QUADRANT_SUBSTEPS: usize = 4;

const GL4_NODES: [f64; 4] = [
    -0.861_136_311_594_052_6,
    -0.339_981_043_584_856_26,
    0.339_981_043_584_856_26,
    0.861_136_311_594_052_6,
];

const GL4_WEIGHTS: [f64; 4] = [
    0.347_854_845_137_453_85,
    0.652_145_154_862_546_1,
    0.652_145_154_862_546_1,
    0.347_854_845_137_453_85,
];

#[verified_engine::verified]
fn ellipse_arc_segment(a: f64, b: f64, eta0: f64, eta1: f64) -> f64 {
    let half = 0.5 * (eta1 - eta0);
    let mid = 0.5 * (eta0 + eta1);
    let mut sum = 0.0_f64;
    for k in 0..4 {
        let eta = mid + half * GL4_NODES[k];
        let (s, c) = eta.sin_cos();
        let ds = (a * a * s * s + b * b * c * c).sqrt();
        sum += GL4_WEIGHTS[k] * ds;
    }
    half * sum
}

#[verified_engine::verified]
fn ellipse_arc_composite(a: f64, b: f64, eta0: f64, eta1: f64) -> f64 {
    if (eta1 - eta0).abs() <= 1e-15 {
        return 0.0;
    }
    let step = (eta1 - eta0) / (QUADRANT_SUBSTEPS as f64);
    let mut total = 0.0_f64;
    for i in 0..QUADRANT_SUBSTEPS {
        let start = eta0 + (i as f64) * step;
        total += ellipse_arc_segment(a, b, start, start + step);
    }
    total
}

/// Integrable confocal elliptical billiard $\frac{x^2}{a^2} + \frac{y^2}{b^2} \le 1$ ($a \ge b > 0$).
///
/// Exploits fourfold quadrant symmetry paired with composite Gauss-Legendre quadrature so
/// Birkhoff coordinates $(s / L, \sin\phi)$ evaluate in 24 bytes of stack state with zero heap allocations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EllipticalBilliard {
    /// Horizontal semi-major axis $a > 0$.
    pub semi_major: f64,
    /// Vertical semi-minor axis $0 < b \le a$.
    pub semi_minor: f64,
    quadrant_arc: f64,
}

impl Default for EllipticalBilliard {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(1.4, 1.0)
    }
}

impl EllipticalBilliard {
    /// Constructs a confocal elliptical billiard, ordering `(semi_major >= semi_minor > 0)`
    /// and precomputing single-quadrant elliptic arc length on the stack.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(axis_a: f64, axis_b: f64) -> Self {
        let clean_a = if axis_a.is_finite() {
            axis_a.abs().max(1e-4)
        } else {
            1.4
        };
        let clean_b = if axis_b.is_finite() {
            axis_b.abs().max(1e-4)
        } else {
            1.0
        };
        let a = clean_a.max(clean_b);
        let b = clean_a.min(clean_b);
        let quadrant_arc = ellipse_arc_composite(a, b, 0.0, FRAC_PI_2);
        Self {
            semi_major: a,
            semi_minor: b,
            quadrant_arc,
        }
    }

    /// Half-focal separation $c = \sqrt{a^2 - b^2}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn focal_distance(&self) -> f64 {
        (self.semi_major * self.semi_major - self.semi_minor * self.semi_minor)
            .max(0.0)
            .sqrt()
    }

    /// Cartesian coordinates $[[-c, 0], [c, 0]]$ of the two foci.
    #[must_use]
    #[verified_engine::verified]
    pub fn foci(&self) -> [[f64; 2]; 2] {
        let c = self.focal_distance();
        [[-c, 0.0], [c, 0.0]]
    }

    /// Dimensionless orbital eccentricity $e = \sqrt{1 - b^2 / a^2} \in [0, 1)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn eccentricity(&self) -> f64 {
        self.focal_distance() / self.semi_major
    }

    /// Total perimeter $L$ computed via composite Gauss-Legendre quadrature.
    #[must_use]
    #[verified_engine::verified]
    pub fn perimeter(&self) -> f64 {
        4.0 * self.quadrant_arc
    }

    /// Returns `true` if `pos` lies strictly inside the elliptical table.
    #[must_use]
    #[verified_engine::verified]
    pub fn contains(&self, pos: [f64; 2]) -> bool {
        if !pos[0].is_finite() || !pos[1].is_finite() {
            return false;
        }
        let nx = pos[0] / self.semi_major;
        let ny = pos[1] / self.semi_minor;
        nx * nx + ny * ny < 1.0 - 1e-9
    }

    /// Evaluates Berry's exact conserved focal angular momentum product $I = L_1 \cdot L_2$,
    /// where $L_{1,2} = (\mathbf{r} - \mathbf{F}_{1,2}) \times \mathbf{v}$:
    /// - $I > 0$: rotational motion tangent to an inner confocal ellipse caustic,
    /// - $I < 0$: librational motion crossing between foci tangent to a confocal hyperbola,
    /// - $I = 0$: separatrix passing directly through the foci $\mathbf{F}_1, \mathbf{F}_2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn focal_angular_momentum_product(&self, ray: &BilliardRay) -> f64 {
        let c = self.focal_distance();
        let l1 = (ray.pos[0] + c) * ray.vel[1] - ray.pos[1] * ray.vel[0];
        let l2 = (ray.pos[0] - c) * ray.vel[1] - ray.pos[1] * ray.vel[0];
        l1 * l2
    }

    #[verified_engine::verified]
    fn arc_length_at_eta(&self, eta: f64) -> f64 {
        let wrapped = eta.rem_euclid(TAU);
        let quadrant = ((wrapped / FRAC_PI_2).floor() as usize).min(3);
        let base_eta = (quadrant as f64) * FRAC_PI_2;
        (quadrant as f64) * self.quadrant_arc
            + ellipse_arc_composite(self.semi_major, self.semi_minor, base_eta, wrapped)
    }

    /// Advances `ray` to the next exact specular collision on the ellipse boundary.
    #[must_use]
    #[verified_engine::verified]
    pub fn collide(&self, ray: &mut BilliardRay) -> Option<BilliardCollision> {
        let inv_a2 = 1.0 / (self.semi_major * self.semi_major);
        let inv_b2 = 1.0 / (self.semi_minor * self.semi_minor);
        let qa = ray.vel[0] * ray.vel[0] * inv_a2 + ray.vel[1] * ray.vel[1] * inv_b2;
        if qa <= 1e-20 || !qa.is_finite() {
            return None;
        }
        let qb = ray.pos[0] * ray.vel[0] * inv_a2 + ray.pos[1] * ray.vel[1] * inv_b2;
        let qc = ray.pos[0] * ray.pos[0] * inv_a2 + ray.pos[1] * ray.pos[1] * inv_b2 - 1.0;
        let disc = qb * qb - qa * qc;
        if disc < 0.0 {
            return None;
        }
        let t = (-qb + disc.sqrt()) / qa;
        if !t.is_finite() || t <= HIT_EPS {
            return None;
        }

        let raw_x = ray.pos[0] + t * ray.vel[0];
        let raw_y = ray.pos[1] + t * ray.vel[1];
        let rho = ((raw_x / self.semi_major).powi(2) + (raw_y / self.semi_minor).powi(2))
            .sqrt()
            .max(1e-15);
        let hit_pos = [raw_x / rho, raw_y / rho];

        let gx = hit_pos[0] * inv_a2;
        let gy = hit_pos[1] * inv_b2;
        let g_norm = gx.hypot(gy).max(1e-15);
        let n_out = [gx / g_norm, gy / g_norm];
        let tangent_ccw = [-n_out[1], n_out[0]];
        let out_vel = reflect_unit(ray.vel, n_out);

        let eta = (hit_pos[1] / self.semi_minor).atan2(hit_pos[0] / self.semi_major);
        let perim = self.perimeter().max(1e-15);
        let arc_fraction = (self.arc_length_at_eta(eta) / perim).rem_euclid(1.0);
        let sin_phi = (out_vel[0] * tangent_ccw[0] + out_vel[1] * tangent_ccw[1]).clamp(-1.0, 1.0);

        ray.pos = hit_pos;
        ray.vel = out_vel;
        Some(BilliardCollision {
            hit_pos,
            out_vel,
            flight_time: t,
            arc_fraction,
            sin_phi,
        })
    }
}
