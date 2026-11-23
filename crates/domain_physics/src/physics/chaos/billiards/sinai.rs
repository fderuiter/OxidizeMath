//! Exact Euclidean ray-collision kernel for the dispersing Sinai Billiard.

// [cite:chaos_theory]

use super::tables::{CandidateHit, keep_earlier};
use super::{BilliardCollision, BilliardRay, HIT_EPS, reflect_unit};
use std::f64::consts::TAU;

/// Hyperbolic dispersing Sinai billiard: square $[-h, h]^2$ with central disk scatterer $x^2 + y^2 \le r^2$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinaiBilliard {
    /// Outer square half-side length $h > 0$.
    pub half_side: f64,
    /// Central dispersing disk radius $0 < r < h$.
    pub scatterer_radius: f64,
}

impl Default for SinaiBilliard {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(1.0, 0.38)
    }
}

impl SinaiBilliard {
    /// Constructs a Sinai billiard clamping `0 < scatterer_radius < 0.9 * half_side`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(half_side: f64, scatterer_radius: f64) -> Self {
        let h = if half_side.is_finite() {
            half_side.abs().max(1e-3)
        } else {
            1.0
        };
        let r = if scatterer_radius.is_finite() {
            scatterer_radius.abs().clamp(1e-4, 0.9 * h)
        } else {
            0.38 * h
        };
        Self {
            half_side: h,
            scatterer_radius: r,
        }
    }

    /// Combined perimeter $L = 8h + 2\pi r$ across outer square ($[0, 8h)$) and scatterer ($[8h, L)$).
    #[must_use]
    #[verified_engine::verified]
    pub fn perimeter(&self) -> f64 {
        8.0 * self.half_side + TAU * self.scatterer_radius
    }

    /// Returns `true` if `pos` lies inside $(-h, h)^2$ and strictly outside the central disk.
    #[must_use]
    #[verified_engine::verified]
    pub fn contains(&self, pos: [f64; 2]) -> bool {
        if !pos[0].is_finite() || !pos[1].is_finite() {
            return false;
        }
        let margin = self.half_side * (1.0 - 1e-9);
        pos[0].abs() < margin
            && pos[1].abs() < margin
            && pos[0].hypot(pos[1]) > self.scatterer_radius * (1.0 + 1e-9)
    }

    #[verified_engine::verified]
    fn check_outer_box(&self, ray: &BilliardRay, best: &mut Option<CandidateHit>) {
        let h = self.half_side;
        if ray.vel[1] < -1e-14 {
            let t = (-h - ray.pos[1]) / ray.vel[1];
            let x = ray.pos[0] + t * ray.vel[0];
            if t > HIT_EPS && x.abs() <= h + 1e-11 {
                let xc = x.clamp(-h, h);
                keep_earlier(
                    best,
                    CandidateHit {
                        t,
                        pos: [xc, -h],
                        n_out: [0.0, -1.0],
                        t_ccw: [1.0, 0.0],
                        arc_s: xc + h,
                    },
                );
            }
        }
        if ray.vel[0] > 1e-14 {
            let t = (h - ray.pos[0]) / ray.vel[0];
            let y = ray.pos[1] + t * ray.vel[1];
            if t > HIT_EPS && y.abs() <= h + 1e-11 {
                let yc = y.clamp(-h, h);
                keep_earlier(
                    best,
                    CandidateHit {
                        t,
                        pos: [h, yc],
                        n_out: [1.0, 0.0],
                        t_ccw: [0.0, 1.0],
                        arc_s: 2.0 * h + (yc + h),
                    },
                );
            }
        }
        if ray.vel[1] > 1e-14 {
            let t = (h - ray.pos[1]) / ray.vel[1];
            let x = ray.pos[0] + t * ray.vel[0];
            if t > HIT_EPS && x.abs() <= h + 1e-11 {
                let xc = x.clamp(-h, h);
                keep_earlier(
                    best,
                    CandidateHit {
                        t,
                        pos: [xc, h],
                        n_out: [0.0, 1.0],
                        t_ccw: [-1.0, 0.0],
                        arc_s: 4.0 * h + (h - xc),
                    },
                );
            }
        }
        if ray.vel[0] < -1e-14 {
            let t = (-h - ray.pos[0]) / ray.vel[0];
            let y = ray.pos[1] + t * ray.vel[1];
            if t > HIT_EPS && y.abs() <= h + 1e-11 {
                let yc = y.clamp(-h, h);
                keep_earlier(
                    best,
                    CandidateHit {
                        t,
                        pos: [-h, yc],
                        n_out: [-1.0, 0.0],
                        t_ccw: [0.0, -1.0],
                        arc_s: 6.0 * h + (h - yc),
                    },
                );
            }
        }
    }

    #[verified_engine::verified]
    fn check_central_scatterer(&self, ray: &BilliardRay, best: &mut Option<CandidateHit>) {
        let r = self.scatterer_radius;
        let b = ray.pos[0] * ray.vel[0] + ray.pos[1] * ray.vel[1];
        let c = ray.pos[0] * ray.pos[0] + ray.pos[1] * ray.pos[1] - r * r;
        let disc = b * b - c;
        if disc <= 1e-24 || b >= 0.0 {
            return;
        }
        let t = -b - disc.sqrt();
        if t <= HIT_EPS {
            return;
        }
        let hx = ray.pos[0] + t * ray.vel[0];
        let hy = ray.pos[1] + t * ray.vel[1];
        let norm = hx.hypot(hy).max(1e-15);
        let n_into_disk = [-hx / norm, -hy / norm];
        if ray.vel[0] * n_into_disk[0] + ray.vel[1] * n_into_disk[1] <= 1e-14 {
            return;
        }
        let pos = [r * hx / norm, r * hy / norm];
        let t_ccw = [-n_into_disk[1], n_into_disk[0]];
        let theta = pos[1].atan2(pos[0]).rem_euclid(TAU);
        keep_earlier(
            best,
            CandidateHit {
                t,
                pos,
                n_out: n_into_disk,
                t_ccw,
                arc_s: 8.0 * self.half_side + r * theta,
            },
        );
    }

    /// Advances `ray` to the next exact specular collision on the square box or central disk.
    #[must_use]
    #[verified_engine::verified]
    pub fn collide(&self, ray: &mut BilliardRay) -> Option<BilliardCollision> {
        let mut best: Option<CandidateHit> = None;
        self.check_central_scatterer(ray, &mut best);
        self.check_outer_box(ray, &mut best);

        let hit = best?;
        let h = self.half_side;
        let on_outer_corner = hit.arc_s < 8.0 * h - 1e-12
            && (hit.pos[0].abs() - h).abs() < 1e-11
            && (hit.pos[1].abs() - h).abs() < 1e-11;
        let out_vel = if on_outer_corner {
            [-ray.vel[0], -ray.vel[1]]
        } else {
            reflect_unit(ray.vel, hit.n_out)
        };
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
    pub(super) fn sample_outer_square(&self, out: &mut [[f64; 2]]) -> usize {
        if out.len() < 4 {
            return 0;
        }
        let h = self.half_side;
        out[0] = [-h, -h];
        out[1] = [h, -h];
        out[2] = [h, h];
        out[3] = [-h, h];
        4
    }
}
