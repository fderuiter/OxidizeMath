//! Exact Euclidean ray-collision kernels for Circular and Bunimovich Stadium Billiards.

// [cite:chaos_theory]

use super::{BilliardCollision, BilliardRay, HIT_EPS, reflect_unit};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Integrable circular disk billiard $x^2 + y^2 \le R^2$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularBilliard {
    /// Disk radius $R > 0$.
    pub radius: f64,
}

impl Default for CircularBilliard {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { radius: 1.0 }
    }
}

impl CircularBilliard {
    /// Constructs a circular billiard of radius `radius`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(radius: f64) -> Self {
        let r = if radius.is_finite() {
            radius.abs().max(1e-4)
        } else {
            1.0
        };
        Self { radius: r }
    }

    /// Circumference $L = 2\pi R$.
    #[must_use]
    #[verified_engine::verified]
    pub fn perimeter(&self) -> f64 {
        TAU * self.radius
    }

    /// Returns `true` if `pos` lies strictly inside the disk.
    #[must_use]
    #[verified_engine::verified]
    pub fn contains(&self, pos: [f64; 2]) -> bool {
        pos[0].is_finite()
            && pos[1].is_finite()
            && pos[0].hypot(pos[1]) < self.radius * (1.0 - 1e-9)
    }

    /// Conserved signed angular momentum $L_z = x v_y - y v_x = R \sin\phi$.
    #[must_use]
    #[verified_engine::verified]
    pub fn angular_momentum(&self, ray: &BilliardRay) -> f64 {
        ray.pos[0] * ray.vel[1] - ray.pos[1] * ray.vel[0]
    }

    /// Advances `ray` to the next exact specular collision on $x^2 + y^2 = R^2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn collide(&self, ray: &mut BilliardRay) -> Option<BilliardCollision> {
        let b = ray.pos[0] * ray.vel[0] + ray.pos[1] * ray.vel[1];
        let c = ray.pos[0] * ray.pos[0] + ray.pos[1] * ray.pos[1] - self.radius * self.radius;
        let disc = b * b - c;
        if disc < 0.0 {
            return None;
        }
        let t = -b + disc.sqrt();
        if !t.is_finite() || t <= HIT_EPS {
            return None;
        }
        let rx = ray.pos[0] + t * ray.vel[0];
        let ry = ray.pos[1] + t * ray.vel[1];
        let norm = rx.hypot(ry).max(1e-15);
        let n_out = [rx / norm, ry / norm];
        let hit_pos = [self.radius * n_out[0], self.radius * n_out[1]];
        let tangent_ccw = [-n_out[1], n_out[0]];
        let out_vel = reflect_unit(ray.vel, n_out);
        let theta = hit_pos[1].atan2(hit_pos[0]).rem_euclid(TAU);
        let sin_phi = (out_vel[0] * tangent_ccw[0] + out_vel[1] * tangent_ccw[1]).clamp(-1.0, 1.0);

        ray.pos = hit_pos;
        ray.vel = out_vel;
        Some(BilliardCollision {
            hit_pos,
            out_vel,
            flight_time: t,
            arc_fraction: (theta / TAU).rem_euclid(1.0),
            sin_phi,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct CandidateHit {
    pub(super) t: f64,
    pub(super) pos: [f64; 2],
    pub(super) n_out: [f64; 2],
    pub(super) t_ccw: [f64; 2],
    pub(super) arc_s: f64,
}

#[verified_engine::verified]
pub(super) fn keep_earlier(best: &mut Option<CandidateHit>, cand: CandidateHit) {
    if cand.t.is_finite() && cand.t > HIT_EPS && best.as_ref().is_none_or(|prev| cand.t < prev.t) {
        *best = Some(cand);
    }
}

/// Ergodic Bunimovich stadium billiard formed by two straight segments $x \in [-a, a], y = \pm R$
/// and two semicircular caps of radius $R$ centered at $(\pm a, 0)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StadiumBilliard {
    /// Half-length $a \ge 0$ of the parallel straight walls.
    pub half_straight: f64,
    /// Semicircular cap radius $R > 0$.
    pub radius: f64,
}

impl Default for StadiumBilliard {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(0.8, 1.0)
    }
}

impl StadiumBilliard {
    /// Constructs a Bunimovich stadium with straight half-length `half_straight` and cap `radius`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(half_straight: f64, radius: f64) -> Self {
        let a = if half_straight.is_finite() {
            half_straight.abs()
        } else {
            0.8
        };
        let r = if radius.is_finite() {
            radius.abs().max(1e-4)
        } else {
            1.0
        };
        Self {
            half_straight: a,
            radius: r,
        }
    }

    /// Total perimeter $L = 4a + 2\pi R$.
    #[must_use]
    #[verified_engine::verified]
    pub fn perimeter(&self) -> f64 {
        4.0 * self.half_straight + TAU * self.radius
    }

    /// Returns `true` if `pos` lies strictly inside the stadium.
    #[must_use]
    #[verified_engine::verified]
    pub fn contains(&self, pos: [f64; 2]) -> bool {
        if !pos[0].is_finite() || !pos[1].is_finite() {
            return false;
        }
        let a = self.half_straight;
        let r = self.radius * (1.0 - 1e-9);
        if pos[0].abs() <= a {
            pos[1].abs() < r
        } else {
            let cx = if pos[0] > 0.0 { a } else { -a };
            (pos[0] - cx).hypot(pos[1]) < r
        }
    }

    #[verified_engine::verified]
    fn check_straight_walls(&self, ray: &BilliardRay, best: &mut Option<CandidateHit>) {
        let a = self.half_straight;
        let r = self.radius;
        if a <= 0.0 {
            return;
        }
        if ray.vel[1] < -1e-14 {
            let t = (-r - ray.pos[1]) / ray.vel[1];
            let x = ray.pos[0] + t * ray.vel[0];
            if t > HIT_EPS && x.abs() <= a + 1e-11 {
                let xc = x.clamp(-a, a);
                keep_earlier(
                    best,
                    CandidateHit {
                        t,
                        pos: [xc, -r],
                        n_out: [0.0, -1.0],
                        t_ccw: [1.0, 0.0],
                        arc_s: xc + a,
                    },
                );
            }
        }
        if ray.vel[1] > 1e-14 {
            let t = (r - ray.pos[1]) / ray.vel[1];
            let x = ray.pos[0] + t * ray.vel[0];
            if t > HIT_EPS && x.abs() <= a + 1e-11 {
                let xc = x.clamp(-a, a);
                keep_earlier(
                    best,
                    CandidateHit {
                        t,
                        pos: [xc, r],
                        n_out: [0.0, 1.0],
                        t_ccw: [-1.0, 0.0],
                        arc_s: 2.0 * a + PI * r + (a - xc),
                    },
                );
            }
        }
    }

    #[verified_engine::verified]
    fn check_cap(&self, ray: &BilliardRay, right_cap: bool, best: &mut Option<CandidateHit>) {
        let a = self.half_straight;
        let r = self.radius;
        let cx = if right_cap { a } else { -a };
        let dx = ray.pos[0] - cx;
        let dy = ray.pos[1];
        let b = dx * ray.vel[0] + dy * ray.vel[1];
        let c = dx * dx + dy * dy - r * r;
        let disc = b * b - c;
        if disc < 0.0 {
            return;
        }
        let t = -b + disc.sqrt();
        if t <= HIT_EPS {
            return;
        }
        let hx = ray.pos[0] + t * ray.vel[0];
        let hy = ray.pos[1] + t * ray.vel[1];
        let on_half = if right_cap {
            hx >= a - 1e-11
        } else {
            hx <= -a + 1e-11
        };
        if !on_half {
            return;
        }
        let lx = hx - cx;
        let norm = lx.hypot(hy).max(1e-15);
        let n_out = [lx / norm, hy / norm];
        if ray.vel[0] * n_out[0] + ray.vel[1] * n_out[1] <= 1e-14 {
            return;
        }
        let pos = [cx + r * n_out[0], r * n_out[1]];
        let t_ccw = [-n_out[1], n_out[0]];
        let arc_s = if right_cap {
            let alpha = n_out[1].atan2(n_out[0]).clamp(-FRAC_PI_2, FRAC_PI_2);
            2.0 * a + r * (alpha + FRAC_PI_2)
        } else {
            let mut alpha = n_out[1].atan2(n_out[0]);
            if alpha < 0.0 {
                alpha += TAU;
            }
            4.0 * a + PI * r + r * (alpha - FRAC_PI_2).clamp(0.0, PI)
        };
        keep_earlier(
            best,
            CandidateHit {
                t,
                pos,
                n_out,
                t_ccw,
                arc_s,
            },
        );
    }

    /// Advances `ray` to the next exact specular collision on the stadium boundary.
    #[must_use]
    #[verified_engine::verified]
    pub fn collide(&self, ray: &mut BilliardRay) -> Option<BilliardCollision> {
        let mut best: Option<CandidateHit> = None;
        self.check_straight_walls(ray, &mut best);
        self.check_cap(ray, true, &mut best);
        self.check_cap(ray, false, &mut best);

        let hit = best?;
        let out_vel = reflect_unit(ray.vel, hit.n_out);
        let perim = self.perimeter().max(1e-15);
        let sin_phi = (out_vel[0] * hit.t_ccw[0] + out_vel[1] * hit.t_ccw[1]).clamp(-1.0, 1.0);
        ray.pos = hit.pos;
        ray.vel = out_vel;
        Some(BilliardCollision {
            hit_pos: hit.pos,
            out_vel,
            flight_time: hit.t,
            arc_fraction: (hit.arc_s / perim).rem_euclid(1.0),
            sin_phi,
        })
    }

    #[verified_engine::verified]
    pub(super) fn sample_boundary(&self, out: &mut [[f64; 2]]) -> usize {
        let n = out.len();
        if n < 8 {
            return 0;
        }
        let a = self.half_straight;
        let r = self.radius;
        let perim = self.perimeter();
        for (idx, slot) in out.iter_mut().enumerate() {
            let s = perim * (idx as f64) / (n as f64);
            *slot = if s < 2.0 * a {
                [-a + s, -r]
            } else if s < 2.0 * a + PI * r {
                let ang = -FRAC_PI_2 + (s - 2.0 * a) / r;
                [a + r * ang.cos(), r * ang.sin()]
            } else if s < 4.0 * a + PI * r {
                [a - (s - 2.0 * a - PI * r), r]
            } else {
                let ang = FRAC_PI_2 + (s - 4.0 * a - PI * r) / r;
                [-a + r * ang.cos(), r * ang.sin()]
            };
        }
        n
    }
}
