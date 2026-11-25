//! Arena-backed workspace for zero-allocation geometric and GRIN ray-fan tracing.

// [cite:optics]

use math_memory::{ArenaError, ArenaLayout, ArenaRegion, OwnedByteArena};
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
    ///
    /// Panics only if layout capacity overflows or allocation fails; see [`Self::try_new`].
    #[must_use]
    pub fn new(max_rays: usize, max_points_per_ray: usize) -> Self {
        Self::try_new(max_rays, max_points_per_ray).expect("RayFanWorkspace::new allocation failed")
    }

    /// Fallibly allocates a workspace capable of storing `max_rays` polylines with `max_points_per_ray` vertices each.
    pub fn try_new(max_rays: usize, max_points_per_ray: usize) -> Result<Self, ArenaError> {
        let caps_r = max_rays.max(1);
        let caps_p = max_points_per_ray.max(4);
        let total_points = caps_r
            .checked_mul(caps_p)
            .ok_or(ArenaError::CapacityOverflow {
                elem_size: core::mem::size_of::<[f64; 2]>(),
                count: caps_r,
            })?;
        let mut layout = ArenaLayout::new();
        let points = layout.add_slice::<[f64; 2]>(total_points)?;
        let ray_f64 = layout.add_slice::<[f64; 6]>(caps_r)?;
        let ray_spans = layout.add_slice::<[usize; 2]>(caps_r)?;
        let ray_flags = layout.add_slice::<[u8; 2]>(caps_r)?;
        let arena = OwnedByteArena::try_with_capacity(layout.total_bytes())?;
        Ok(Self {
            rk4: RungeKutta4::new(&ArrayState([0.0; 4])),
            arena,
            points,
            ray_f64,
            ray_spans,
            ray_flags,
            max_rays: caps_r,
            active_rays: 0,
            max_points_per_ray: caps_p,
        })
    }

    /// Number of active traced rays from the most recent `trace_ray_fan` call.
    #[must_use]
    pub const fn ray_count(&self) -> usize {
        self.active_rays
    }

    /// Reconstructs the metadata record for traced ray `index`.
    ///
    /// Panics if `index >= self.ray_count()`; see [`Self::checked_ray`].
    #[must_use]
    pub fn ray(&self, index: usize) -> TracedRayMeta {
        self.checked_ray(index).expect("ray index out of bounds")
    }

    /// Bounds-checked accessor for the metadata record of traced ray `index`.
    pub fn checked_ray(&self, index: usize) -> Result<TracedRayMeta, ArenaError> {
        if index >= self.active_rays {
            return Err(ArenaError::InsufficientCapacity {
                requested: index.saturating_add(1),
                available: self.active_rays,
            });
        }
        Ok(self.read_meta_unchecked(index))
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
    ///
    /// Panics if `index >= self.ray_count()`; see [`Self::checked_ray_vertices`].
    #[must_use]
    pub fn ray_vertices(&self, index: usize) -> &[[f64; 2]] {
        self.checked_ray_vertices(index)
            .expect("ray index out of bounds")
    }

    /// Bounds-checked accessor for the vertex slice of traced ray `index`.
    pub fn checked_ray_vertices(&self, index: usize) -> Result<&[[f64; 2]], ArenaError> {
        if index >= self.active_rays {
            return Err(ArenaError::InsufficientCapacity {
                requested: index.saturating_add(1),
                available: self.active_rays,
            });
        }
        let span = self.arena.region(self.ray_spans)[index];
        Ok(&self.arena.region(self.points)[span[0]..span[0] + span[1]])
    }
}
