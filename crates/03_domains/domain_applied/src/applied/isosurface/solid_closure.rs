//! Watertight 2-manifold solid-closure extraction for [`MarchingCubes`] via a virtual scalar collar.
// [cite:isosurface_extraction]

use super::edge_cache::{EMPTY_EDGE_VERTEX, SliceEdgeCache};
use super::gradients::{CentralDifferenceEstimator, GradientEstimator};
use super::marching_cubes::MarchingCubes;
use super::tables::{CUBE_EDGE_FLAGS, EDGE_CONNECTION, TRIANGLE_CONNECTION_TABLE, VERTEX_OFFSET};
use super::types::{Mesh, Point3D, VoxelGrid};
use super::workspace::IsosurfaceWorkspace;
use crate::error::IsosurfaceError;
use math_commons::math_kernel::types::flatten_3d_index;

/// Selects which side of the scalar level-set threshold forms the interior of a closed 3D solid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SolidPolarity {
    /// Values strictly below `threshold` (`f(x, y, z) < threshold`) lie inside the solid (SDF convention).
    #[default]
    Sublevel,
    /// Values strictly above `threshold` (`f(x, y, z) > threshold`) lie inside the solid (density/lobe convention).
    Superlevel,
}

/// Axis-aligned half-space cutting plane retaining `coord <= max_coord` and sealing the cut cross-section.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CuttingPlane {
    /// Coordinate axis index (`0 = X`, `1 = Y`, `2 = Z`).
    pub axis: usize,
    /// Maximum world-space coordinate retained along `axis`; `coord > max_coord` is clipped and capped.
    pub max_coord: f32,
}

/// Configuration for watertight 2-manifold isosurface extraction with boundary and cutting-plane caps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolidClosureConfig {
    /// Interior sign convention relative to `threshold`.
    pub polarity: SolidPolarity,
    /// Optional axis-aligned cutting plane sealed with outward-oriented planar cap triangles.
    pub cutting_plane: Option<CuttingPlane>,
}

impl SolidClosureConfig {
    /// Closed sublevel solid (`f(x, y, z) < threshold` inside) without an interior cutting plane.
    #[must_use]
    #[verified_engine::verified]
    pub const fn sublevel() -> Self {
        Self {
            polarity: SolidPolarity::Sublevel,
            cutting_plane: None,
        }
    }

    /// Closed superlevel solid (`f(x, y, z) > threshold` inside) without an interior cutting plane.
    #[must_use]
    #[verified_engine::verified]
    pub const fn superlevel() -> Self {
        Self {
            polarity: SolidPolarity::Superlevel,
            cutting_plane: None,
        }
    }

    /// Attaches an optional half-space [`CuttingPlane`] whose cross-section is capped into a solid.
    #[must_use]
    #[verified_engine::verified]
    pub const fn with_cutting_plane(mut self, cutting_plane: Option<CuttingPlane>) -> Self {
        self.cutting_plane = cutting_plane;
        self
    }
}

#[derive(Clone, Copy)]
struct AxisCollarSpec {
    orig: f32,
    step: f32,
    pad: f32,
    limit: usize,
    cap_coord: f32,
}

impl AxisCollarSpec {
    fn new(n_real: usize, orig: f32, step: f32, cut_max: Option<f32>) -> Self {
        let pad = (0.02 * step).max(1.0e-4);
        let max_span = orig + ((n_real - 1) as f32) * step;
        let Some(max_c) = cut_max else {
            return Self {
                orig,
                step,
                pad,
                limit: n_real,
                cap_coord: max_span + pad,
            };
        };
        if !max_c.is_finite() || max_c >= max_span {
            return Self {
                orig,
                step,
                pad,
                limit: n_real,
                cap_coord: max_span + pad,
            };
        }
        if max_c < orig {
            return Self {
                orig,
                step,
                pad,
                limit: 0,
                cap_coord: orig,
            };
        }
        let mut k = (((max_c - orig) / step).floor().max(0.0) as usize).min(n_real);
        while k < n_real && orig + (k as f32) * step <= max_c {
            k += 1;
        }
        while k > 0 && orig + ((k - 1) as f32) * step > max_c {
            k -= 1;
        }
        let last_kept = if k == 0 {
            orig - pad
        } else {
            orig + ((k - 1) as f32) * step
        };
        Self {
            orig,
            step,
            pad,
            limit: k,
            cap_coord: max_c.max(last_kept + pad),
        }
    }

    #[inline(always)]
    fn world_coord(&self, g: usize) -> f32 {
        if g == 0 {
            self.orig - self.pad
        } else if g <= self.limit {
            self.orig + ((g - 1) as f32) * self.step
        } else {
            self.cap_coord + ((g - self.limit - 1) as f32) * self.pad
        }
    }
}

struct SolidCubeContext {
    values: [f32; 8],
    is_collar: [bool; 8],
    positions: [Point3D; 8],
    coords: [usize; 3],
    case_index: usize,
    tau: f32,
    sign: f32,
}

#[inline]
fn interpolate_point(p1: Point3D, v1: f32, p2: Point3D, v2: f32, tau: f32) -> Point3D {
    let denom = v2 - v1;
    if denom.abs() < Point3D::EPSILON {
        return (p1 + p2) * 0.5;
    }
    let t = ((tau - v1) / denom).clamp(1.0e-4, 1.0 - 1.0e-4);
    p1 + (p2 - p1) * t
}

#[inline]
fn unit_or_fallback(v: Point3D, fallback: Point3D) -> Point3D {
    let n = v.normalize();
    if n.is_finite() && n.magnitude() > 0.5 {
        n
    } else {
        fallback
    }
}

impl<'a, G: GradientEstimator> MarchingCubes<'a, G> {
    /// Extracts a watertight, outward-oriented 2-manifold [`Mesh`] sealed at outer bounds and `closure.cutting_plane`.
    #[verified_engine::verified]
    pub fn extract_solid(
        &self,
        threshold: f32,
        closure: SolidClosureConfig,
    ) -> Result<Mesh, IsosurfaceError> {
        let mut workspace =
            IsosurfaceWorkspace::new_lazy(self.grid.width() + 2, self.grid.height() + 2);
        let mut mesh = Mesh::new();
        self.extract_solid_into(threshold, closure, &mut workspace, &mut mesh)?;
        Ok(mesh)
    }

    /// Extracts a watertight 2-manifold [`Mesh`] into `out_mesh` reusing `workspace` with zero heap allocations.
    ///
    /// When `workspace` is constructed via [`IsosurfaceWorkspace::with_locked_capacity`] with at least
    /// `(grid.width() + 2, grid.height() + 2)` slice dimensions, this method allocates zero heap bytes.
    #[verified_engine::verified]
    pub fn extract_solid_into(
        &self,
        threshold: f32,
        closure: SolidClosureConfig,
        workspace: &mut IsosurfaceWorkspace,
        out_mesh: &mut Mesh,
    ) -> Result<(), IsosurfaceError> {
        self.validate_grid()?;
        out_mesh.clear();
        if !threshold.is_finite() {
            return Ok(());
        }
        let (w, h) = (self.grid.width(), self.grid.height());
        let axes = self.build_collar_axes(closure.cutting_plane);
        if axes[0].limit == 0 || axes[1].limit == 0 || axes[2].limit == 0 {
            return Ok(());
        }
        workspace.edge_cache.reset_for_dimensions(w + 2, h + 2);
        let sign = match closure.polarity {
            SolidPolarity::Sublevel => 1.0_f32,
            SolidPolarity::Superlevel => -1.0_f32,
        };
        let tau = sign * threshold;
        let collar_val = tau + (tau.abs() * 1.0e-3).max(1.0e-3);
        let tie_nudge = (tau.abs() * 2.0e-6).max(2.0e-6);

        for gz in 0..=axes[2].limit {
            if gz > 0 {
                workspace.edge_cache.advance_z();
            }
            for gy in 0..=axes[1].limit {
                if gy > 0 {
                    workspace.edge_cache.advance_y();
                }
                for gx in 0..=axes[0].limit {
                    if !self.process_solid_cube(
                        [gx, gy, gz],
                        &axes,
                        (sign, tau, collar_val, tie_nudge),
                        workspace,
                        out_mesh,
                    ) {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    fn build_collar_axes(&self, cut: Option<CuttingPlane>) -> [AxisCollarSpec; 3] {
        let orig = self.grid.origin();
        let vs = self.grid.voxel_size();
        let cut_for = |ax: usize| cut.filter(|c| c.axis.min(2) == ax).map(|c| c.max_coord);
        [
            AxisCollarSpec::new(self.grid.width(), orig.x, vs.x, cut_for(0)),
            AxisCollarSpec::new(self.grid.height(), orig.y, vs.y, cut_for(1)),
            AxisCollarSpec::new(self.grid.depth(), orig.z, vs.z, cut_for(2)),
        ]
    }

    #[inline]
    fn process_solid_cube(
        &self,
        coords: [usize; 3],
        axes: &[AxisCollarSpec; 3],
        scalars: (f32, f32, f32, f32),
        workspace: &mut IsosurfaceWorkspace,
        out_mesh: &mut Mesh,
    ) -> bool {
        let (sign, tau, collar_val, tie_nudge) = scalars;
        let (w, h) = (self.grid.width(), self.grid.height());
        let data = self.grid.data();
        let mut values = [collar_val; 8];
        let mut is_collar = [true; 8];
        let mut case_index = 0usize;

        for (corner, &[ox, oy, oz]) in VERTEX_OFFSET.iter().enumerate() {
            let (vx, vy, vz) = (coords[0] + ox, coords[1] + oy, coords[2] + oz);
            if vx > 0
                && vx <= axes[0].limit
                && vy > 0
                && vy <= axes[1].limit
                && vz > 0
                && vz <= axes[2].limit
            {
                is_collar[corner] = false;
                let idx = flatten_3d_index(vx - 1, vy - 1, vz - 1, w, h);
                let mut v = sign * data[idx];
                if !v.is_finite() {
                    v = collar_val;
                } else if (v - tau).abs() < tie_nudge {
                    v = tau + tie_nudge;
                }
                values[corner] = v;
                if v < tau {
                    case_index |= 1 << corner;
                }
            }
        }
        if CUBE_EDGE_FLAGS[case_index] == 0 {
            return true;
        }
        if workspace.is_capacity_locked()
            && (out_mesh.vertices.len() + 12 > out_mesh.vertices.capacity()
                || out_mesh.normals.len() + 12 > out_mesh.normals.capacity()
                || out_mesh.indices.len() + 15 > out_mesh.indices.capacity())
        {
            return false;
        }
        let positions = Self::solid_cube_positions(coords, axes);
        let ctx = SolidCubeContext {
            values,
            is_collar,
            positions,
            coords,
            case_index,
            tau,
            sign,
        };
        self.emit_solid_cube_triangles(&ctx, out_mesh, &mut workspace.edge_cache);
        true
    }

    #[inline]
    fn solid_cube_positions(coords: [usize; 3], axes: &[AxisCollarSpec; 3]) -> [Point3D; 8] {
        let (x0, x1) = (
            axes[0].world_coord(coords[0]),
            axes[0].world_coord(coords[0] + 1),
        );
        let (y0, y1) = (
            axes[1].world_coord(coords[1]),
            axes[1].world_coord(coords[1] + 1),
        );
        let (z0, z1) = (
            axes[2].world_coord(coords[2]),
            axes[2].world_coord(coords[2] + 1),
        );
        [
            Point3D::new(x0, y0, z0),
            Point3D::new(x1, y0, z0),
            Point3D::new(x1, y1, z0),
            Point3D::new(x0, y1, z0),
            Point3D::new(x0, y0, z1),
            Point3D::new(x1, y0, z1),
            Point3D::new(x1, y1, z1),
            Point3D::new(x0, y1, z1),
        ]
    }

    #[inline]
    fn emit_solid_cube_triangles(
        &self,
        ctx: &SolidCubeContext,
        out_mesh: &mut Mesh,
        edge_cache: &mut SliceEdgeCache,
    ) {
        let edge_flags = CUBE_EDGE_FLAGS[ctx.case_index];
        let mut edge_verts = [0usize; 12];
        for (edge_idx, &[c1, c2]) in EDGE_CONNECTION.iter().enumerate() {
            if (edge_flags & (1 << edge_idx)) == 0 {
                continue;
            }
            let slot = edge_cache.slot_mut(ctx.coords[0], ctx.coords[1], edge_idx);
            if *slot != EMPTY_EDGE_VERTEX {
                edge_verts[edge_idx] = *slot as usize;
            } else {
                let pos = interpolate_point(
                    ctx.positions[c1],
                    ctx.values[c1],
                    ctx.positions[c2],
                    ctx.values[c2],
                    ctx.tau,
                );
                let normal = self.solid_edge_normal(ctx, c1, c2);
                let new_idx = out_mesh.vertices.len();
                out_mesh.vertices.push(pos);
                out_mesh.normals.push(normal);
                *slot = new_idx as u32;
                edge_verts[edge_idx] = new_idx;
            }
        }
        let table_row = &TRIANGLE_CONNECTION_TABLE[ctx.case_index];
        let mut k = 0usize;
        while table_row[k] != -1 {
            let i0 = edge_verts[table_row[k] as usize];
            let i1 = edge_verts[table_row[k + 1] as usize];
            let i2 = edge_verts[table_row[k + 2] as usize];
            out_mesh.indices.push(i0);
            out_mesh.indices.push(i2);
            out_mesh.indices.push(i1);
            k += 3;
        }
    }

    #[inline]
    fn solid_edge_normal(&self, ctx: &SolidCubeContext, c1: usize, c2: usize) -> Point3D {
        let fallback = Point3D::new(0.0, 0.0, 1.0);
        if ctx.is_collar[c1] && !ctx.is_collar[c2] {
            return unit_or_fallback(ctx.positions[c1] - ctx.positions[c2], fallback);
        }
        if ctx.is_collar[c2] && !ctx.is_collar[c1] {
            return unit_or_fallback(ctx.positions[c2] - ctx.positions[c1], fallback);
        }
        let n1 = self.world_outward_gradient(ctx.coords, c1, ctx.sign);
        let n2 = self.world_outward_gradient(ctx.coords, c2, ctx.sign);
        let denom = ctx.values[c2] - ctx.values[c1];
        let t = if denom.abs() > Point3D::EPSILON {
            ((ctx.tau - ctx.values[c1]) / denom).clamp(0.0, 1.0)
        } else {
            0.5
        };
        unit_or_fallback(n1 + (n2 - n1) * t, unit_or_fallback(n1, fallback))
    }

    #[inline]
    fn world_outward_gradient(&self, cube_coords: [usize; 3], corner: usize, sign: f32) -> Point3D {
        let [ox, oy, oz] = VERTEX_OFFSET[corner];
        let (rx, ry, rz) = (
            (cube_coords[0] + ox).saturating_sub(1),
            (cube_coords[1] + oy).saturating_sub(1),
            (cube_coords[2] + oz).saturating_sub(1),
        );
        let raw = self.estimator.gradient(self.grid, rx, ry, rz);
        let vs = self.grid.voxel_size();
        Point3D::new(
            (sign * raw.x) / vs.x.max(Point3D::EPSILON),
            (sign * raw.y) / vs.y.max(Point3D::EPSILON),
            (sign * raw.z) / vs.z.max(Point3D::EPSILON),
        )
    }
}

/// Extracts a watertight, outward-oriented 2-manifold [`Mesh`] using [`CentralDifferenceEstimator`].
#[verified_engine::verified]
pub fn extract_solid_isosurface(
    grid: &VoxelGrid,
    threshold: f32,
    closure: SolidClosureConfig,
) -> Result<Mesh, IsosurfaceError> {
    MarchingCubes::new(grid, CentralDifferenceEstimator).extract_solid(threshold, closure)
}

/// Extracts a watertight 2-manifold [`Mesh`] into `out_mesh` reusing `workspace` with zero heap allocations.
#[verified_engine::verified]
pub fn extract_solid_isosurface_into(
    grid: &VoxelGrid,
    threshold: f32,
    closure: SolidClosureConfig,
    workspace: &mut IsosurfaceWorkspace,
    out_mesh: &mut Mesh,
) -> Result<(), IsosurfaceError> {
    MarchingCubes::new(grid, CentralDifferenceEstimator)
        .extract_solid_into(threshold, closure, workspace, out_mesh)
}
