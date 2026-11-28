//! Relativistic worldline samples, initial 4-velocities, Doppler three-velocities,
//! and exact non-circular Kerr/Schwarzschild proper-time clock dilation.

use std::f64::consts::FRAC_PI_2;

use math_commons::ast::CompiledExpr;
use math_commons::constants::C;
use nalgebra::SVector;
use pure_math::pure_math::analysis::ode::ArrayState;

use super::custom_metric::MIN_REGULAR_LAPSE;
use super::kerr::KerrEquatorialMetric;
use super::schwarzschild::{
    GeodesicKind, SchwarzschildBlackHole, combined_clock_rate, flamm_paraboloid_z,
};
use super::wormhole::MorrisThorneWormhole;

/// Fixed upper bound on RK4 steps per trajectory (`samples` reserves `MAX_TRAJECTORY_SAMPLES + 8`).
pub const MAX_TRAJECTORY_SAMPLES: usize = 720;
/// Maximum simultaneous geodesics integrated in `PhotonFan` mode.
pub const MAX_BUNDLE_RAYS: usize = 16;

/// Pre-allocated trajectory storage for a single ray in a geodesic bundle.
#[derive(Debug, Clone)]
pub struct BundleRay {
    /// Sampled worldline points up to [`MAX_TRAJECTORY_SAMPLES`].
    pub samples: Vec<TrajectorySample>,
    /// Whether this ray halted at the horizon singularity guard.
    pub halted: bool,
    /// Impact parameter $b$ or angular momentum $\ell$ assigned to this ray.
    pub impact_b: f64,
}

impl BundleRay {
    /// Pre-allocates sample buffer for zero-allocation stepping.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            samples: Vec::with_capacity(cap),
            halted: false,
            impact_b: 0.0,
        }
    }
}

/// Single integrated worldline sample with 3D embedding coordinates, Kerr clocks, and Doppler velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectorySample {
    /// Coordinate time $ct$.
    pub coord_ct: f64,
    /// Observer proper parameter $\tau$ (or affine $\lambda$ for null rays).
    pub proper_tau: f64,
    /// Radial coordinate $r$ (or proper throat coordinate $l$ for wormholes).
    pub radial_coord: f64,
    /// Azimuthal angle $\varphi$.
    pub phi: f64,
    /// Radial derivative $dr/d\lambda$ or $dl/d\lambda$.
    pub radial_vel: f64,
    /// Coordinate angular velocity $\Omega = d\varphi / d(ct)$.
    pub dphi_dct: f64,
    /// Local spatial three-velocity $\boldsymbol{\beta} = \mathbf{v} / c$ for Doppler beaming.
    pub velocity_beta: [f64; 3],
    /// Embedded 3D point $[x, y, z]$ on Flamm's paraboloid or Morris-Thorne surface.
    pub embed_xyz: [f64; 3],
    /// Quadratic metric norm $g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu$.
    pub metric_norm: f64,
    /// Instantaneous proper-to-coordinate clock rate $d\tau / dt$.
    pub clock_rate: f64,
    /// Instantaneous static/ZAMO gravitational lapse dilation factor.
    pub grav_dilation: f64,
    /// Instantaneous local Lorentz factor $\gamma(v)$.
    pub lorentz_gamma: f64,
}

/// Converts coordinate radial and angular velocities into a bounded planar three-velocity $\boldsymbol{\beta} = \mathbf{v} / c$.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn planar_velocity_beta(r: f64, phi: f64, dr_dct: f64, dphi_dct: f64) -> [f64; 3] {
    let (sin_p, cos_p) = phi.sin_cos();
    let v_rad = dr_dct.clamp(-0.92, 0.92);
    let v_tan = (r * dphi_dct).clamp(-0.92, 0.92);
    let vx = v_rad * cos_p - v_tan * sin_p;
    let vy = v_rad * sin_p + v_tan * cos_p;
    let mag_sq = vx * vx + vy * vy;
    let out = if mag_sq > 0.90 {
        let s = 0.948 / mag_sq.sqrt();
        [vx * s, vy * s, 0.0]
    } else {
        [vx, vy, 0.0]
    };
    debug_assert!(out[0].is_finite() && out[1].is_finite());
    out
}

/// Computes normalized initial 4D position, 4-velocity, and specific energy $e$ in equatorial Kerr/Schwarzschild coordinates.
#[must_use]
#[verified_engine::verified]
pub fn initial_kerr_4d_state(
    kind: GeodesicKind,
    rs: f64,
    spin_a: f64,
    r0: f64,
    phi0: f64,
    impact_val: f64,
    radial_bias: f64,
) -> (SVector<f64, 4>, SVector<f64, 4>, f64) {
    let delta = (r0 * r0 - rs * r0 + spin_a * spin_a).max(1e-6);
    let sqrt_delta = delta.sqrt();
    let g03 = -(spin_a * rs) / r0;
    let g11 = (r0 * r0) / delta;
    let g33 = r0 * r0 + spin_a * spin_a + (spin_a * spin_a * rs) / r0;

    match kind {
        GeodesicKind::Null => {
            let mut ell = impact_val;
            if sqrt_delta - g03 > 1e-9 {
                ell = ell.min(g33 / (sqrt_delta - g03));
            }
            if sqrt_delta + g03 > 1e-9 {
                ell = ell.max(-g33 / (sqrt_delta + g03));
            }
            let ut = (g33 - (spin_a * rs / r0) * ell) / delta;
            let uphi = ((spin_a * rs / r0) + (1.0 - rs / r0) * ell) / delta;
            let ur_sq = ((delta / (r0 * r0)) * (ut - ell * uphi)).max(0.0);
            let sign = if radial_bias > 0.0 { 1.0 } else { -1.0 };
            debug_assert!(ut.is_finite() && uphi.is_finite());
            (
                SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, phi0),
                SVector::<f64, 4>::new(ut.max(1e-4), sign * ur_sq.sqrt(), 0.0, uphi),
                1.0,
            )
        }
        GeodesicKind::Timelike => {
            let ell = impact_val;
            let ur = radial_bias * 0.45;
            let omega0 = (spin_a * rs) / (r0 * g33);
            let ut = ((g33 / delta) * (1.0 + (ell * ell) / g33 + g11 * ur * ur))
                .max(1e-6)
                .sqrt();
            let e = omega0 * ell + (delta / g33) * ut;
            let uphi = ((spin_a * rs / r0) * e + (1.0 - rs / r0) * ell) / delta;
            debug_assert!(ut > 0.0 && e.is_finite());
            (
                SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, phi0),
                SVector::<f64, 4>::new(ut, ur, 0.0, uphi),
                e,
            )
        }
    }
}

/// Computes normalized initial 4D position, 4-velocity, and specific energy $e$ for a diagonal static radial-lapse metric.
#[must_use]
#[verified_engine::verified]
pub fn initial_diagonal_4d_state(
    kind: GeodesicKind,
    r0: f64,
    phi0: f64,
    f0: f64,
    impact_val: f64,
    radial_bias: f64,
) -> (SVector<f64, 4>, SVector<f64, 4>, f64) {
    let safe_f = f0.max(MIN_REGULAR_LAPSE);
    match kind {
        GeodesicKind::Null => {
            let e = 1.0;
            let max_b = r0 / safe_f.sqrt();
            let ell = impact_val.clamp(-max_b, max_b);
            let ut = e / safe_f;
            let uphi = ell / (r0 * r0);
            let ur_sq = (e * e - safe_f * (ell * ell) / (r0 * r0)).max(0.0);
            let sign = if radial_bias > 0.0 { 1.0 } else { -1.0 };
            debug_assert!(ut > 0.0);
            (
                SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, phi0),
                SVector::<f64, 4>::new(ut, sign * ur_sq.sqrt(), 0.0, uphi),
                e,
            )
        }
        GeodesicKind::Timelike => {
            let ell = impact_val;
            let uphi = ell / (r0 * r0);
            let ur = radial_bias * 0.45;
            let e_sq = ur * ur + safe_f * (1.0 + (ell * ell) / (r0 * r0));
            let e = e_sq.max(1e-6).sqrt();
            let ut = e / safe_f;
            debug_assert!(ut > 0.0 && e.is_finite());
            (
                SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, phi0),
                SVector::<f64, 4>::new(ut, ur, 0.0, uphi),
                e,
            )
        }
    }
}

/// Computes normalized initial 3-velocity $(u^t, u^l, u^\varphi, e)$ on a traversable wormhole surface of squared radius `r2`.
#[must_use]
#[verified_engine::verified]
pub fn wormhole_initial_vel(
    kind: GeodesicKind,
    r2: f64,
    impact_val: f64,
    radial_bias: f64,
) -> (f64, f64, f64, f64) {
    let safe_r2 = r2.max(0.01);
    let sign = if radial_bias > 0.0 { 1.0 } else { -1.0 };
    let out = match kind {
        GeodesicKind::Null => {
            let max_b = safe_r2.sqrt();
            let ell = impact_val.clamp(-max_b, max_b);
            let uphi = ell / safe_r2;
            let ul = sign * (1.0 - (ell * ell) / safe_r2).max(0.0).sqrt();
            (1.0, ul, uphi, 1.0)
        }
        GeodesicKind::Timelike => {
            let uphi = impact_val / safe_r2;
            let e = (1.0 + (impact_val * impact_val) / safe_r2 + radial_bias * radial_bias).sqrt();
            (e, sign * radial_bias.abs(), uphi, e)
        }
    };
    debug_assert!(out.0 > 0.0 && out.3.is_finite());
    out
}

/// Evaluates gravitational lapse, total proper-to-coordinate clock rate $d\tau / dt$, and local Lorentz factor $\gamma$.
///
/// Includes the radial coordinate velocity contribution $-\frac{r^2}{\Delta}(dr/d(ct))^2$ via
/// [`KerrEquatorialMetric::worldline_clock_dilation`] so eccentric and radially infalling Kerr
/// geodesics report exact proper-time dilation matching $1 / u^t$.
#[must_use]
#[verified_engine::verified]
pub fn compute_4d_clock_factors(
    bh: Option<&SchwarzschildBlackHole>,
    throat_r: f64,
    spin_a: f64,
    r: f64,
    dtau_dt: f64,
    dr_dct: f64,
    dphi_dct: f64,
) -> (f64, f64, f64) {
    if let Some(schwarz) = bh {
        let r_safe = r.max(1.001 * throat_r);
        let g = schwarz.gravitational_time_dilation(r_safe).unwrap_or(0.0);
        let beta = if g > 1e-6 && dtau_dt > 0.0 {
            (1.0 - (dtau_dt / g).clamp(0.0, 1.0).powi(2))
                .max(0.0)
                .sqrt()
                .min(0.999)
        } else {
            0.0
        };
        let rt = combined_clock_rate(schwarz, r_safe, beta * C).unwrap_or(dtau_dt);
        let gam = if beta > 0.0 {
            1.0 / (1.0 - beta * beta).sqrt()
        } else {
            1.0
        };
        debug_assert!(gam >= 1.0);
        (g, rt, gam)
    } else {
        let rs = (throat_r * throat_r + spin_a * spin_a) / throat_r.max(1e-6);
        let kerr = KerrEquatorialMetric::new(rs, spin_a).ok();
        let r_safe = r.max(1.005 * throat_r);
        let zamo_lapse = kerr
            .and_then(|k| k.zamo_clock_dilation(r_safe).ok())
            .unwrap_or(0.0);
        let rate = if dtau_dt > 0.0 {
            kerr.and_then(|k| k.worldline_clock_dilation(r_safe, dr_dct, dphi_dct).ok())
                .unwrap_or(dtau_dt)
        } else {
            0.0
        };
        let gam = if zamo_lapse > 1e-6 && rate > 0.0 {
            (zamo_lapse / rate).max(1.0)
        } else {
            1.0
        };
        debug_assert!(gam >= 1.0);
        (zamo_lapse, rate, gam)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Sample4dMeta<'a> {
    pub(crate) kind: GeodesicKind,
    pub(crate) bh: Option<&'a SchwarzschildBlackHole>,
    pub(crate) throat_r: f64,
    pub(crate) spin_a: f64,
}

#[verified_engine::verified]
pub(crate) fn push_4d_worldline_sample(
    buf: &mut Vec<TrajectorySample>,
    meta: Sample4dMeta<'_>,
    proper_tau: f64,
    state: &ArrayState<8>,
    metric_norm: f64,
) {
    if buf.len() >= buf.capacity() || buf.len() > MAX_TRAJECTORY_SAMPLES {
        return;
    }
    let r = state.0[1].max(meta.throat_r);
    let phi = state.0[3];
    let z = flamm_paraboloid_z(meta.throat_r, r).unwrap_or(0.0);
    let (sin_p, cos_p) = phi.sin_cos();
    let ut = state.0[4].max(1e-6);
    let dr_dct = state.0[5] / ut;
    let dphi_dct = state.0[7] / ut;
    let velocity_beta = planar_velocity_beta(r, phi, dr_dct, dphi_dct);
    let dtau_dt = match meta.kind {
        GeodesicKind::Null => 0.0,
        GeodesicKind::Timelike => (1.0 / ut).clamp(0.0, 1.0),
    };
    let (grav, rate, gamma) = compute_4d_clock_factors(
        meta.bh,
        meta.throat_r,
        meta.spin_a,
        r,
        dtau_dt,
        dr_dct,
        dphi_dct,
    );
    buf.push(TrajectorySample {
        coord_ct: state.0[0],
        proper_tau,
        radial_coord: r,
        phi,
        radial_vel: state.0[5],
        dphi_dct,
        velocity_beta,
        embed_xyz: [r * cos_p, r * sin_p, z],
        metric_norm,
        clock_rate: rate,
        grav_dilation: grav,
        lorentz_gamma: gamma,
    });
}

#[verified_engine::verified]
pub(crate) fn push_wormhole_worldline_sample(
    buf: &mut Vec<TrajectorySample>,
    kind: GeodesicKind,
    wh: &MorrisThorneWormhole,
    proper_tau: f64,
    state: &ArrayState<6>,
    metric_norm: f64,
) {
    if buf.len() >= buf.capacity() || buf.len() > MAX_TRAJECTORY_SAMPLES {
        return;
    }
    let l = state.0[1];
    let phi = state.0[2];
    let xyz = wh.embedding_xyz(l, phi).unwrap_or([0.0, 0.0, 0.0]);
    let ut = state.0[3].max(1.0);
    let dphi_dct = state.0[5] / ut;
    let r_areal = wh.areal_radius(l).unwrap_or(wh.throat_radius);
    let velocity_beta = planar_velocity_beta(r_areal, phi, state.0[4] / ut, dphi_dct);
    let clock_rate = match kind {
        GeodesicKind::Null => 0.0,
        GeodesicKind::Timelike => (1.0 / ut).clamp(0.0, 1.0),
    };
    buf.push(TrajectorySample {
        coord_ct: state.0[0],
        proper_tau,
        radial_coord: l,
        phi,
        radial_vel: state.0[4],
        dphi_dct,
        velocity_beta,
        embed_xyz: xyz,
        metric_norm,
        clock_rate,
        grav_dilation: 1.0,
        lorentz_gamma: ut,
    });
}

#[derive(Clone, Copy)]
pub(crate) struct CustomWorldlineCtx<'a> {
    pub(crate) kind: GeodesicKind,
    pub(crate) compiled: &'a CompiledExpr,
    pub(crate) scale: f64,
    pub(crate) params: &'a [f64],
}

#[verified_engine::verified]
pub(crate) fn push_custom_lapse_worldline_sample(
    buf: &mut Vec<TrajectorySample>,
    ctx: CustomWorldlineCtx<'_>,
    proper_tau: f64,
    state: &ArrayState<8>,
    metric_norm: f64,
) {
    if buf.len() >= buf.capacity() || buf.len() > MAX_TRAJECTORY_SAMPLES {
        return;
    }
    let r = state.0[1].max(0.08);
    let phi = state.0[3];
    let f = ctx.compiled.eval_2d(r, ctx.scale, ctx.params).max(0.0);
    let r_eff = (r * (1.0 - f)).clamp(0.0, r);
    let z = 2.0 * (r_eff * (r - r_eff).max(0.0)).sqrt();
    let (sin_p, cos_p) = phi.sin_cos();
    let grav = f.sqrt();
    let ut = state.0[4].max(1e-6);
    let dphi_dct = state.0[7] / ut;
    let velocity_beta = planar_velocity_beta(r, phi, state.0[5] / ut, dphi_dct);
    let dtau_dt = match ctx.kind {
        GeodesicKind::Null => 0.0,
        GeodesicKind::Timelike => (1.0 / ut).clamp(0.0, 1.0),
    };
    let gamma = if grav > 1e-6 && dtau_dt > 0.0 {
        (grav / dtau_dt).max(1.0)
    } else {
        1.0
    };
    buf.push(TrajectorySample {
        coord_ct: state.0[0],
        proper_tau,
        radial_coord: r,
        phi,
        radial_vel: state.0[5],
        dphi_dct,
        velocity_beta,
        embed_xyz: [r * cos_p, r * sin_p, z],
        metric_norm,
        clock_rate: dtau_dt,
        grav_dilation: grav,
        lorentz_gamma: gamma,
    });
}

#[verified_engine::verified]
pub(crate) fn push_custom_wormhole_worldline_sample(
    buf: &mut Vec<TrajectorySample>,
    ctx: CustomWorldlineCtx<'_>,
    proper_tau: f64,
    state: &ArrayState<6>,
    metric_norm: f64,
) {
    if buf.len() >= buf.capacity() || buf.len() > MAX_TRAJECTORY_SAMPLES {
        return;
    }
    let l = state.0[1];
    let phi = state.0[2];
    let b0 = ctx.scale.max(0.1);
    let r = ctx.compiled.eval_2d(l, ctx.scale, ctx.params).max(0.1);
    let z = b0 * (l / b0).asinh();
    let (sin_p, cos_p) = phi.sin_cos();
    let ut = state.0[3].max(1.0);
    let dphi_dct = state.0[5] / ut;
    let velocity_beta = planar_velocity_beta(r, phi, state.0[4] / ut, dphi_dct);
    let clock_rate = match ctx.kind {
        GeodesicKind::Null => 0.0,
        GeodesicKind::Timelike => (1.0 / ut).clamp(0.0, 1.0),
    };
    buf.push(TrajectorySample {
        coord_ct: state.0[0],
        proper_tau,
        radial_coord: l,
        phi,
        radial_vel: state.0[4],
        dphi_dct,
        velocity_beta,
        embed_xyz: [r * cos_p, r * sin_p, z],
        metric_norm,
        clock_rate,
        grav_dilation: 1.0,
        lorentz_gamma: ut,
    });
}

// [cite:general_relativity_geodesics]
