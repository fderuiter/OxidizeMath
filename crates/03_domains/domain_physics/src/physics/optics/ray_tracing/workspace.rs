//! Arena-backed `RayFanWorkspace` and zero-allocation ray-fan bench integration.

// [cite:optics]

use super::surfaces::{RayStepCtx, trace_single_ray};
use super::{RayFanConfig, TracedRayMeta};
use math_memory::{ArenaError, ArenaLayout, ArenaRegion, FlagRegion, OwnedByteArena};
use pure_math::pure_math::analysis::ode::{ArrayState, RungeKutta4};

const POLY_BANDS_NM: [f64; 5] = [420.0, 480.0, 540.0, 600.0, 670.0];

/// Pre-allocated workspace for zero-allocation ray-fan integration and surface tracing.
pub struct RayFanWorkspace {
    rk4: RungeKutta4<ArrayState<4>>,
    arena: OwnedByteArena,
    points: ArenaRegion<[f64; 2]>,
    spans: ArenaRegion<[usize; 2]>,
    kinematics: ArenaRegion<[f64; 6]>,
    spectral_bands: ArenaRegion<u8>,
    tir_flags: FlagRegion,
    max_rays: usize,
    active_rays: usize,
    max_points_per_ray: usize,
}

impl RayFanWorkspace {
    /// Fallibly allocates a single-arena workspace capable of storing `max_rays` polylines with `max_points_per_ray` vertices each.
    pub fn try_new(max_rays: usize, max_points_per_ray: usize) -> Result<Self, ArenaError> {
        let caps_r = max_rays.max(1);
        let caps_p = max_points_per_ray.max(4);
        let total_pts = caps_r
            .checked_mul(caps_p)
            .ok_or(ArenaError::ArithmeticOverflow)?;
        let mut layout = ArenaLayout::new();
        let points = layout.reserve::<[f64; 2]>(total_pts)?;
        let spans = layout.reserve::<[usize; 2]>(caps_r)?;
        let kinematics = layout.reserve::<[f64; 6]>(caps_r)?;
        let spectral_bands = layout.reserve::<u8>(caps_r)?;
        let tir_flags = layout.reserve_flags(caps_r)?;
        Ok(Self {
            rk4: RungeKutta4::new(&ArrayState([0.0; 4])),
            arena: OwnedByteArena::try_from_layout(&layout)?,
            points,
            spans,
            kinematics,
            spectral_bands,
            tir_flags,
            max_rays: caps_r,
            active_rays: 0,
            max_points_per_ray: caps_p,
        })
    }

    /// Allocates a single-arena workspace capable of storing `max_rays` polylines with `max_points_per_ray` vertices each.
    #[must_use]
    pub fn new(max_rays: usize, max_points_per_ray: usize) -> Self {
        Self::try_new(max_rays, max_points_per_ray).expect("RayFanWorkspace::new failed")
    }

    /// Number of active traced rays from the most recent [`trace_ray_fan`] call.
    #[must_use]
    pub const fn ray_count(&self) -> usize {
        self.active_rays
    }

    /// Fallibly reconstructs the metadata record for traced ray `index` (`0..ray_count()`).
    pub fn try_ray_meta(&self, index: usize) -> Result<TracedRayMeta, ArenaError> {
        if index >= self.active_rays {
            return Err(ArenaError::IndexOutOfBounds {
                index,
                len: self.active_rays,
            });
        }
        let [point_start, point_len] = self.arena.region_slice(self.spans)[index];
        let [wavelength_nm, power, ox, oy, dx, dy] =
            self.arena.region_slice(self.kinematics)[index];
        let spectral_band = self.arena.region_slice(self.spectral_bands)[index];
        let total_internal_reflection = self.arena.flag_get(self.tir_flags, index);
        Ok(TracedRayMeta {
            point_start,
            point_len,
            wavelength_nm,
            power,
            exit_origin: [ox, oy],
            exit_dir: [dx, dy],
            spectral_band,
            total_internal_reflection,
        })
    }

    /// Reconstructs the metadata record for traced ray `index` (`0..ray_count()`).
    #[must_use]
    pub fn ray_meta(&self, index: usize) -> TracedRayMeta {
        self.try_ray_meta(index).expect("ray index out of bounds")
    }

    /// Iterates over active [`TracedRayMeta`] records without heap allocation.
    pub fn iter_rays(&self) -> impl ExactSizeIterator<Item = TracedRayMeta> + '_ {
        (0..self.active_rays).map(|idx| self.ray_meta(idx))
    }

    /// Fallibly returns the vertex slice for traced ray `index`.
    pub fn try_ray_vertices(&self, index: usize) -> Result<&[[f64; 2]], ArenaError> {
        if index >= self.active_rays {
            return Err(ArenaError::IndexOutOfBounds {
                index,
                len: self.active_rays,
            });
        }
        let [point_start, point_len] = self.arena.region_slice(self.spans)[index];
        Ok(&self.arena.region_slice(self.points)[point_start..point_start + point_len])
    }

    /// Returns the vertex slice for traced ray `index`.
    #[must_use]
    pub fn ray_vertices(&self, index: usize) -> &[[f64; 2]] {
        self.try_ray_vertices(index)
            .expect("ray index out of bounds")
    }
}

/// Traces a ray fan across the configured optical bench scene into `workspace` with zero heap allocations.
#[verified_engine::verified]
pub fn trace_ray_fan(config: &RayFanConfig, workspace: &mut RayFanWorkspace) -> usize {
    let bands: &[f64] = if config.polychromatic {
        &POLY_BANDS_NM
    } else {
        std::slice::from_ref(&config.wavelength_nm)
    };
    let max_rays = workspace.max_rays;
    let max_pts = workspace.max_points_per_ray;
    let spatial_n = (config.ray_count.max(1))
        .min(max_rays / bands.len().max(1))
        .max(1);
    let (sin_a, cos_a) = config.launch_angle_rad.sin_cos();
    let dir0 = [cos_a, sin_a];
    let normal_perp = [-sin_a, cos_a];

    let mut cur = workspace.arena.raw_cursor();
    let points_buf = cur.region_mut(workspace.points);
    let spans_buf = cur.region_mut(workspace.spans);
    let kin_buf = cur.region_mut(workspace.kinematics);
    let band_buf = cur.region_mut(workspace.spectral_bands);
    let mut tir_buf = cur.flags_mut(workspace.tir_flags);

    let mut out_idx = 0usize;
    for (b_idx, &wl_nm) in bands.iter().enumerate() {
        for r_idx in 0..spatial_n {
            if out_idx >= max_rays {
                break;
            }
            let u = if spatial_n <= 1 {
                0.0
            } else {
                2.0 * (r_idx as f64) / ((spatial_n - 1) as f64) - 1.0
            };
            let offset = u * config.beam_half_width;
            let origin = [
                config.source_x + offset * normal_perp[0],
                config.source_y + offset * normal_perp[1],
            ];
            let p_start = out_idx * max_pts;
            let p_slice = &mut points_buf[p_start..p_start + max_pts];
            let mut meta = TracedRayMeta {
                point_start: p_start,
                point_len: 0,
                wavelength_nm: wl_nm,
                power: 1.0,
                exit_origin: origin,
                exit_dir: dir0,
                spectral_band: b_idx as u8,
                total_internal_reflection: false,
            };
            let ctx = RayStepCtx {
                cfg: config,
                origin,
                dir0,
                wl_nm,
            };
            trace_single_ray(&ctx, &mut workspace.rk4, p_slice, &mut meta);
            spans_buf[out_idx] = [meta.point_start, meta.point_len];
            kin_buf[out_idx] = [
                meta.wavelength_nm,
                meta.power,
                meta.exit_origin[0],
                meta.exit_origin[1],
                meta.exit_dir[0],
                meta.exit_dir[1],
            ];
            band_buf[out_idx] = meta.spectral_band;
            tir_buf.set(out_idx, meta.total_internal_reflection);
            out_idx += 1;
        }
    }
    workspace.active_rays = out_idx;
    out_idx
}
