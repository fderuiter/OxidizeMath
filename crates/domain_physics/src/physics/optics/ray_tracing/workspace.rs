//! Arena-backed workspace for zero-allocation geometric and GRIN ray-fan tracing.

// [cite:optics]

use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};
use pure_math::pure_math::analysis::ode::{ArrayState, RungeKutta4};

/// Per-ray metadata indexing polyline slices returned by [`RayFanWorkspace::ray_vertices`].
#[derive(Debug, Clone, Copy, Default)]
pub struct TracedRayMeta {
    /// Starting index in `RayFanWorkspace` vertex storage.
    pub point_start: usize,
    /// Number of valid vertices along this ray polyline.
    pub point_len: usize,
    /// Vacuum wavelength in nanometers.
    pub wavelength_nm: f64,
    /// Remaining normalized optical power in $[0, 1]$.
    pub power: f64,
    /// Final straight-segment origin for caustic intersection.
    pub exit_origin: [f64; 2],
    /// Final unit propagation direction.
    pub exit_dir: [f64; 2],
    /// Spectral channel index (`0..5`) so caustic pairing stays monochromatic.
    pub spectral_band: u8,
    /// True if ray underwent total internal reflection.
    pub total_internal_reflection: bool,
}

/// Pre-allocated workspace for zero-allocation ray-fan integration and surface tracing.
pub struct RayFanWorkspace {
    pub(super) rk4: RungeKutta4<ArrayState<4>>,
    pub(super) arena: OwnedByteArena,
    pub(super) points: ArenaRegion<[f64; 2]>,
    pub(super) ray_f64: ArenaRegion<[f64; 6]>,
    pub(super) ray_spans: ArenaRegion<[usize; 2]>,
    pub(super) ray_flags: ArenaRegion<[u8; 2]>,
    pub(super) max_rays: usize,
    pub(super) active_rays: usize,
    pub(super) max_points_per_ray: usize,
}

impl RayFanWorkspace {
    /// Allocates a workspace capable of storing `max_rays` polylines with `max_points_per_ray` vertices each.
    #[must_use]
    pub fn new(max_rays: usize, max_points_per_ray: usize) -> Self {
        let caps_r = max_rays.max(1);
        let caps_p = max_points_per_ray.max(4);
        let mut layout = ArenaLayout::new();
        let points = layout
            .add_slice::<[f64; 2]>(
                caps_r
                    .checked_mul(caps_p)
                    .expect("ray vertex count overflow"),
            )
            .expect("ray points layout");
        let ray_f64 = layout
            .add_slice::<[f64; 6]>(caps_r)
            .expect("ray f64 layout");
        let ray_spans = layout
            .add_slice::<[usize; 2]>(caps_r)
            .expect("ray spans layout");
        let ray_flags = layout
            .add_slice::<[u8; 2]>(caps_r)
            .expect("ray flags layout");
        let arena = OwnedByteArena::with_capacity(layout.total_bytes());
        Self {
            rk4: RungeKutta4::new(&ArrayState([0.0; 4])),
            arena,
            points,
            ray_f64,
            ray_spans,
            ray_flags,
            max_rays: caps_r,
            active_rays: 0,
            max_points_per_ray: caps_p,
        }
    }

    /// Number of active traced rays from the most recent `trace_ray_fan` call.
    #[must_use]
    pub const fn ray_count(&self) -> usize {
        self.active_rays
    }

    /// Reconstructs the metadata record for traced ray `index`.
    #[must_use]
    pub fn ray(&self, index: usize) -> TracedRayMeta {
        assert!(index < self.active_rays, "ray index out of bounds");
        self.read_meta_unchecked(index)
    }

    pub(super) fn read_meta_unchecked(&self, index: usize) -> TracedRayMeta {
        let span = self.arena.region(self.ray_spans)[index];
        let vals = self.arena.region(self.ray_f64)[index];
        let flags = self.arena.region(self.ray_flags)[index];
        TracedRayMeta {
            point_start: span[0],
            point_len: span[1],
            wavelength_nm: vals[0],
            power: vals[1],
            exit_origin: [vals[2], vals[3]],
            exit_dir: [vals[4], vals[5]],
            spectral_band: flags[0],
            total_internal_reflection: flags[1] != 0,
        }
    }

    pub(super) fn write_meta(&mut self, index: usize, meta: &TracedRayMeta) {
        self.arena.region_mut(self.ray_spans)[index] = [meta.point_start, meta.point_len];
        self.arena.region_mut(self.ray_f64)[index] = [
            meta.wavelength_nm,
            meta.power,
            meta.exit_origin[0],
            meta.exit_origin[1],
            meta.exit_dir[0],
            meta.exit_dir[1],
        ];
        self.arena.region_mut(self.ray_flags)[index] =
            [meta.spectral_band, u8::from(meta.total_internal_reflection)];
    }

    /// Exact-size iterator over active traced ray metadata records from the most recent `trace_ray_fan` call.
    pub fn rays(&self) -> impl ExactSizeIterator<Item = TracedRayMeta> + '_ {
        (0..self.active_rays).map(|idx| self.read_meta_unchecked(idx))
    }

    /// Returns the vertex slice for traced ray `index`.
    #[must_use]
    pub fn ray_vertices(&self, index: usize) -> &[[f64; 2]] {
        assert!(index < self.active_rays, "ray index out of bounds");
        let span = self.arena.region(self.ray_spans)[index];
        &self.arena.region(self.points)[span[0]..span[0] + span[1]]
    }
}
