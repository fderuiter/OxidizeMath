//! Analytical geodesic ray-wall intersection and circle-inversion reflection helpers.

use super::billiards::{BilliardParticle, BilliardWall, circle_inversion};
use super::{GeodesicArc, Su11Matrix, clamp_to_disk};
use num_complex::Complex64;

const MAX_BOUNCES_PER_STEP: usize = 8;

#[verified_engine::verified]
pub(super) fn propagate_particle(
    walls: &[BilliardWall],
    mut particle: BilliardParticle,
    mut last_wall: Option<usize>,
    ds: f64,
) -> (BilliardParticle, Option<usize>, u64) {
    let mut remaining = ds;
    let mut bounces = 0_u64;
    let mut iter = 0;
    while remaining > 1e-12 && iter < MAX_BOUNCES_PER_STEP {
        if let Some((wall_idx, hit_sigma)) = nearest_wall_hit(walls, particle, last_wall)
            && hit_sigma <= remaining + 1e-12
        {
            let travel = hit_sigma.clamp(0.0, remaining);
            particle = advance_geodesic(particle, travel);
            particle = reflect_on_wall(&walls[wall_idx], particle);
            remaining = (remaining - travel).max(0.0);
            last_wall = Some(wall_idx);
            bounces += 1;
            iter += 1;
            continue;
        }
        particle = advance_geodesic(particle, remaining);
        particle = enforce_interior(walls, particle);
        break;
    }
    (particle, last_wall, bounces)
}

#[verified_engine::verified]
fn advance_geodesic(particle: BilliardParticle, sigma: f64) -> BilliardParticle {
    if sigma <= 1e-15 {
        return particle;
    }
    let dir_norm = particle.dir.norm().max(1e-15);
    let u = particle.dir / dir_norm;
    let w = u * (0.5 * sigma).tanh();
    let boost = Su11Matrix::translation_from_origin(particle.pos);
    BilliardParticle {
        pos: boost.apply(w),
        dir: boost.apply_direction(w, u),
    }
}

#[verified_engine::verified]
fn nearest_wall_hit(
    walls: &[BilliardWall],
    particle: BilliardParticle,
    last_wall: Option<usize>,
) -> Option<(usize, f64)> {
    let to_origin = Su11Matrix::translation_to_origin(particle.pos);
    let u = to_origin.apply_direction(particle.pos, particle.dir);
    let mut best: Option<(usize, f64)> = None;
    for (idx, wall) in walls.iter().enumerate() {
        if last_wall == Some(idx) {
            continue;
        }
        if let Some(sigma) = ray_wall_sigma(&to_origin, u, wall)
            && best.is_none_or(|(_, b_sig)| sigma < b_sig)
        {
            best = Some((idx, sigma));
        }
    }
    best
}

#[verified_engine::verified]
fn ray_wall_sigma(to_origin: &Su11Matrix, u: Complex64, wall: &BilliardWall) -> Option<f64> {
    let w0 = to_origin.apply(wall.v0);
    let w1 = to_origin.apply(wall.v1);
    let c0 = w0.re * u.im - w0.im * u.re;
    let c1 = u.re * w1.im - u.im * w1.re;
    if c0 < -1e-11 || c1 < -1e-11 {
        return None;
    }
    let arc = GeodesicArc::through_points(w0, w1);
    if arc.is_diameter {
        return None;
    }
    let b = u.re * arc.circle_center.re + u.im * arc.circle_center.im;
    if b <= 1.0 {
        return None;
    }
    let t = (1.0 / (b + (b * b - 1.0).max(0.0).sqrt())).clamp(0.0, 1.0 - 1e-12);
    Some(2.0 * t.atanh())
}

#[verified_engine::verified]
fn reflect_on_wall(wall: &BilliardWall, particle: BilliardParticle) -> BilliardParticle {
    let center = wall.arc.circle_center;
    let radius = wall.arc.circle_radius;
    let radial = particle.pos - center;
    let rn = radial.norm();
    if rn <= 1e-15 || radius <= 1e-15 {
        return BilliardParticle {
            pos: particle.pos,
            dir: -particle.dir,
        };
    }
    let inward_normal = radial / rn;
    let snapped = clamp_to_disk(center + inward_normal * (radius + 1e-11));
    let dot = particle.dir.re * inward_normal.re + particle.dir.im * inward_normal.im;
    let mut reflected = particle.dir - inward_normal * (2.0 * dot);
    let refl_dot = reflected.re * inward_normal.re + reflected.im * inward_normal.im;
    if refl_dot < 0.0 {
        reflected -= inward_normal * (2.0 * refl_dot);
    }
    let norm = reflected.norm().max(1e-15);
    BilliardParticle {
        pos: snapped,
        dir: reflected / norm,
    }
}

#[verified_engine::verified]
fn enforce_interior(walls: &[BilliardWall], mut particle: BilliardParticle) -> BilliardParticle {
    for wall in walls {
        if !wall.arc.is_diameter {
            let diff = particle.pos - wall.arc.circle_center;
            if diff.norm_sqr() < wall.arc.circle_radius * wall.arc.circle_radius {
                particle.pos =
                    circle_inversion(particle.pos, wall.arc.circle_center, wall.arc.circle_radius);
                particle = reflect_on_wall(wall, particle);
            }
        }
    }
    particle
}

// [cite:modular_polynomials_review]
