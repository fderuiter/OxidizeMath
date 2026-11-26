//! Discrete optical bench surface tracers and RK4 continuous GRIN steppers.

// [cite:optics]

use pure_math::pure_math::analysis::ode::{ArrayState, OdeSystem, RungeKutta4, Solver};

use super::{
    EikonalRaySystem, GrinMedium, OpticalBenchScene, PolarizationMode, RayFanConfig, TracedRayMeta,
};
use crate::physics::optics::refraction::{
    SnellOutcome, cauchy_refractive_index, fresnel_polarized, intersect_ray_circle_arc,
    intersect_ray_segment, normalize_2d, reflect_vector_2d, snell_vector_2d,
};

pub(super) struct RayStepCtx<'a> {
    pub(super) cfg: &'a RayFanConfig,
    pub(super) origin: [f64; 2],
    pub(super) dir0: [f64; 2],
    pub(super) wl_nm: f64,
}

fn f_transmittance(n1: f64, n2: f64, cos_i: f64, pol: PolarizationMode) -> f64 {
    let f = fresnel_polarized(n1, n2, cos_i.clamp(-1.0, 1.0).acos());
    match pol {
        PolarizationMode::Unpolarized => f.transmittance_unpolarized,
        PolarizationMode::SPolarized => f.transmittance_s,
        PolarizationMode::PPolarized => f.transmittance_p,
    }
}

pub(super) fn trace_single_ray(
    ctx: &RayStepCtx<'_>,
    rk4: &mut RungeKutta4<ArrayState<4>>,
    pts: &mut [[f64; 2]],
    meta: &mut TracedRayMeta,
) {
    let ext = ctx.cfg.bench_half_extent.max(1.0);
    match ctx.cfg.scene {
        OpticalBenchScene::CoffeeCupMirror { radius } => {
            trace_coffee_cup(ctx.origin, ctx.dir0, radius, ext, pts, meta);
        }
        OpticalBenchScene::SphericalBiconvexLens {
            radius_curv,
            half_thickness,
            aperture_radius,
            glass,
        } => {
            let n_glass = cauchy_refractive_index(glass.cauchy(), ctx.wl_nm);
            trace_biconvex_lens(
                ctx,
                [radius_curv, half_thickness, aperture_radius, n_glass],
                ext,
                pts,
                meta,
            );
        }
        OpticalBenchScene::DispersivePrism {
            apex_angle_rad,
            side_length,
            tilt_rad,
            glass,
        } => {
            let n_glass = cauchy_refractive_index(glass.cauchy(), ctx.wl_nm);
            trace_prism(
                ctx,
                [apex_angle_rad, side_length, tilt_rad, n_glass],
                ext,
                pts,
                meta,
            );
        }
        OpticalBenchScene::PlanarDielectric {
            interface_x,
            n1,
            n2,
        } => {
            trace_planar(ctx, [interface_x, n1, n2], ext, pts, meta);
        }
        OpticalBenchScene::ContinuousGrin {
            medium,
            step_tau,
            steps,
        } => trace_grin(
            ctx.origin,
            ctx.dir0,
            medium,
            (step_tau, steps, ext),
            rk4,
            pts,
            meta,
        ),
    }
}

fn trace_coffee_cup(
    origin: [f64; 2],
    dir0: [f64; 2],
    radius: f64,
    ext: f64,
    pts: &mut [[f64; 2]],
    meta: &mut TracedRayMeta,
) {
    pts[0] = origin;
    if let Some((_, hit, out_n)) =
        intersect_ray_circle_arc(origin, dir0, [0.0, 0.0], radius, Some(1.0), radius)
    {
        let refl = reflect_vector_2d(dir0, [-out_n[0], -out_n[1]]);
        pts[1] = hit;
        pts[2] = [
            hit[0] + 2.2 * radius * refl[0],
            hit[1] + 2.2 * radius * refl[1],
        ];
        meta.point_len = 3;
        meta.exit_origin = hit;
        meta.exit_dir = refl;
    } else {
        pts[1] = [
            origin[0] + 2.5 * ext * dir0[0],
            origin[1] + 2.5 * ext * dir0[1],
        ];
        meta.point_len = 2;
    }
}

fn trace_biconvex_lens(
    ctx: &RayStepCtx<'_>,
    lens: [f64; 4],
    ext: f64,
    pts: &mut [[f64; 2]],
    meta: &mut TracedRayMeta,
) {
    let [r_curv, half_t, ap_r, n_glass] = lens;
    let pol = ctx.cfg.polarization;
    pts[0] = ctx.origin;
    let c1 = [r_curv - half_t, 0.0];
    let c2 = [half_t - r_curv, 0.0];
    let Some((_, hit1, n1_out)) =
        intersect_ray_circle_arc(ctx.origin, ctx.dir0, c1, r_curv, Some(-1.0), ap_r)
    else {
        pts[1] = [
            ctx.origin[0] + 2.5 * ext * ctx.dir0[0],
            ctx.origin[1] + 2.5 * ext * ctx.dir0[1],
        ];
        meta.point_len = 2;
        return;
    };
    pts[1] = hit1;
    let SnellOutcome::Refracted {
        direction: dir1,
        cos_theta_i: ci1,
        ..
    } = snell_vector_2d(ctx.dir0, n1_out, 1.0, n_glass)
    else {
        let refl = reflect_vector_2d(ctx.dir0, n1_out);
        pts[2] = [hit1[0] + 2.0 * ext * refl[0], hit1[1] + 2.0 * ext * refl[1]];
        meta.point_len = 3;
        meta.total_internal_reflection = true;
        return;
    };
    meta.power *= f_transmittance(1.0, n_glass, ci1, pol);
    let Some((_, hit2, n2_out)) = intersect_ray_circle_arc(hit1, dir1, c2, r_curv, Some(1.0), ap_r)
    else {
        pts[2] = [hit1[0] + 2.0 * ext * dir1[0], hit1[1] + 2.0 * ext * dir1[1]];
        meta.point_len = 3;
        meta.exit_origin = hit1;
        meta.exit_dir = dir1;
        return;
    };
    pts[2] = hit2;
    finish_exit_refraction(hit2, dir1, n2_out, (n_glass, 1.0, ext), pol, pts, meta);
}

fn finish_exit_refraction(
    hit2: [f64; 2],
    dir1: [f64; 2],
    n2_out: [f64; 2],
    media_ext: (f64, f64, f64),
    pol: PolarizationMode,
    pts: &mut [[f64; 2]],
    meta: &mut TracedRayMeta,
) {
    let (n1, n2, ext) = media_ext;
    match snell_vector_2d(dir1, n2_out, n1, n2) {
        SnellOutcome::Refracted {
            direction: dir2,
            cos_theta_i: ci2,
            ..
        } => {
            meta.power *= f_transmittance(n1, n2, ci2, pol);
            pts[3] = [hit2[0] + 2.5 * ext * dir2[0], hit2[1] + 2.5 * ext * dir2[1]];
            meta.point_len = 4;
            meta.exit_origin = hit2;
            meta.exit_dir = dir2;
        }
        SnellOutcome::TotalInternalReflection {
            reflected_direction,
            ..
        } => {
            pts[3] = [
                hit2[0] + 2.0 * ext * reflected_direction[0],
                hit2[1] + 2.0 * ext * reflected_direction[1],
            ];
            meta.point_len = 4;
            meta.exit_origin = hit2;
            meta.exit_dir = reflected_direction;
            meta.total_internal_reflection = true;
        }
    }
}

fn trace_prism(
    ctx: &RayStepCtx<'_>,
    prism: [f64; 4],
    ext: f64,
    pts: &mut [[f64; 2]],
    meta: &mut TracedRayMeta,
) {
    let [apex_angle, side, tilt, n_glass] = prism;
    let half_a = 0.5 * apex_angle;
    let (h, w) = (side * half_a.cos(), side * half_a.sin());
    let raw_v = [[0.0, 0.6 * h], [-w, -0.4 * h], [w, -0.4 * h]];
    let (st, ct) = tilt.sin_cos();
    let verts = raw_v.map(|v| [v[0] * ct - v[1] * st, v[0] * st + v[1] * ct]);
    let edges = [
        (verts[0], verts[1]),
        (verts[1], verts[2]),
        (verts[2], verts[0]),
    ];

    pts[0] = ctx.origin;
    let (mut cur_o, mut cur_d, mut inside, mut v_count) = (ctx.origin, ctx.dir0, false, 1usize);
    for _ in 0..2 {
        let mut best: Option<(f64, [f64; 2], [f64; 2])> = None;
        for &(a, b) in &edges {
            if let Some(hit) = intersect_ray_segment(cur_o, cur_d, a, b)
                && best.is_none_or(|prev| hit.0 < prev.0)
            {
                best = Some(hit);
            }
        }
        let Some((_, hit_pt, n_seg)) = best else {
            break;
        };
        pts[v_count] = hit_pt;
        v_count += 1;
        let (n1, n2) = if inside {
            (n_glass, 1.0)
        } else {
            (1.0, n_glass)
        };
        match snell_vector_2d(cur_d, n_seg, n1, n2) {
            SnellOutcome::Refracted {
                direction,
                cos_theta_i,
                ..
            } => {
                meta.power *= f_transmittance(n1, n2, cos_theta_i, ctx.cfg.polarization);
                cur_o = hit_pt;
                cur_d = direction;
                inside = !inside;
            }
            SnellOutcome::TotalInternalReflection {
                reflected_direction,
                ..
            } => {
                cur_o = hit_pt;
                cur_d = reflected_direction;
                meta.total_internal_reflection = true;
            }
        }
    }
    pts[v_count] = [
        cur_o[0] + 2.5 * ext * cur_d[0],
        cur_o[1] + 2.5 * ext * cur_d[1],
    ];
    meta.point_len = v_count + 1;
    meta.exit_origin = cur_o;
    meta.exit_dir = cur_d;
}

fn trace_planar(
    ctx: &RayStepCtx<'_>,
    planar: [f64; 3],
    ext: f64,
    pts: &mut [[f64; 2]],
    meta: &mut TracedRayMeta,
) {
    let [x0, n1, n2] = planar;
    pts[0] = ctx.origin;
    if let Some((_, hit, n_seg)) =
        intersect_ray_segment(ctx.origin, ctx.dir0, [x0, -2.0 * ext], [x0, 2.0 * ext])
    {
        pts[1] = hit;
        match snell_vector_2d(ctx.dir0, n_seg, n1, n2) {
            SnellOutcome::Refracted {
                direction,
                cos_theta_i,
                ..
            } => {
                meta.power *= f_transmittance(n1, n2, cos_theta_i, ctx.cfg.polarization);
                pts[2] = [
                    hit[0] + 2.2 * ext * direction[0],
                    hit[1] + 2.2 * ext * direction[1],
                ];
                meta.point_len = 3;
                meta.exit_origin = hit;
                meta.exit_dir = direction;
            }
            SnellOutcome::TotalInternalReflection {
                reflected_direction,
                ..
            } => {
                pts[2] = [
                    hit[0] + 2.2 * ext * reflected_direction[0],
                    hit[1] + 2.2 * ext * reflected_direction[1],
                ];
                meta.point_len = 3;
                meta.exit_origin = hit;
                meta.exit_dir = reflected_direction;
                meta.total_internal_reflection = true;
            }
        }
    } else {
        pts[1] = [
            ctx.origin[0] + 2.2 * ext * ctx.dir0[0],
            ctx.origin[1] + 2.2 * ext * ctx.dir0[1],
        ];
        meta.point_len = 2;
    }
}

struct LuneburgInterior {
    inv_r2: f64,
}

impl OdeSystem<ArrayState<4>> for LuneburgInterior {
    fn derivative(&self, _tau: f64, s: &ArrayState<4>) -> ArrayState<4> {
        ArrayState([s.0[2], s.0[3], -s.0[0] * self.inv_r2, -s.0[1] * self.inv_r2])
    }

    fn derivative_in_place(&self, _tau: f64, s: &ArrayState<4>, out: &mut ArrayState<4>) {
        out.0 = [s.0[2], s.0[3], -s.0[0] * self.inv_r2, -s.0[1] * self.inv_r2];
    }
}

fn trace_luneburg_exterior(
    origin: [f64; 2],
    dir0: [f64; 2],
    radius: f64,
    steps_ext: (usize, f64),
    rk4: &mut RungeKutta4<ArrayState<4>>,
    pts: &mut [[f64; 2]],
    meta: &mut TracedRayMeta,
) {
    let (steps, ext) = steps_ext;
    pts[0] = origin;
    let Some((_, entry, _)) =
        intersect_ray_circle_arc(origin, dir0, [0.0, 0.0], radius, None, radius)
    else {
        pts[1] = [
            origin[0] + 2.5 * ext * dir0[0],
            origin[1] + 2.5 * ext * dir0[1],
        ];
        meta.point_len = 2;
        return;
    };
    pts[1] = entry;
    let m = steps.min(pts.len().saturating_sub(3)).max(16);
    let dt = (std::f64::consts::FRAC_PI_2 * radius) / (m as f64);
    let sys = LuneburgInterior {
        inv_r2: 1.0 / (radius * radius).max(1.0e-12),
    };
    let mut state = ArrayState([entry[0], entry[1], dir0[0], dir0[1]]);
    for s in 0..m {
        rk4.step(&sys, (s as f64) * dt, &mut state, dt);
        pts[2 + s] = [state.0[0], state.0[1]];
    }
    let exit_pt = pts[1 + m];
    let exit_dir = normalize_2d([state.0[2], state.0[3]]).unwrap_or(dir0);
    pts[2 + m] = [
        exit_pt[0] + 2.2 * ext * exit_dir[0],
        exit_pt[1] + 2.2 * ext * exit_dir[1],
    ];
    meta.point_len = 3 + m;
    meta.exit_origin = exit_pt;
    meta.exit_dir = exit_dir;
}

fn trace_grin(
    origin: [f64; 2],
    dir0: [f64; 2],
    medium: GrinMedium,
    step_cfg: (f64, usize, f64),
    rk4: &mut RungeKutta4<ArrayState<4>>,
    pts: &mut [[f64; 2]],
    meta: &mut TracedRayMeta,
) {
    let (step_tau, steps, ext) = step_cfg;
    if let GrinMedium::Luneburg { radius } = medium
        && origin[0] * origin[0] + origin[1] * origin[1] > radius * radius + 1.0e-12
    {
        trace_luneburg_exterior(origin, dir0, radius, (steps, ext), rk4, pts, meta);
        return;
    }
    let sys = EikonalRaySystem { medium };
    let n_init = medium.refractive_index_at(origin[0], origin[1]);
    let mut state = ArrayState([origin[0], origin[1], n_init * dir0[0], n_init * dir0[1]]);
    pts[0] = origin;
    let max_steps = steps.min(pts.len().saturating_sub(1)).max(1);
    let dt = step_tau.clamp(1.0e-4, 0.25);
    let mut len = 1usize;
    for step in 0..max_steps {
        rk4.step(&sys, (step as f64) * dt, &mut state, dt);
        pts[len] = [state.0[0], state.0[1]];
        len += 1;
        if state.0[0].abs() > 1.8 * ext || state.0[1].abs() > 1.8 * ext {
            break;
        }
    }
    meta.point_len = len;
    meta.exit_origin = pts[len - 1];
    meta.exit_dir = normalize_2d([state.0[2], state.0[3]]).unwrap_or(dir0);
}
