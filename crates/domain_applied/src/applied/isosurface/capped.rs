//! Boundary-capped, outward-wound [`MarchingCubes`] solid extraction inside a 1-voxel apron.
// [cite:isosurface_extraction]

use super::gradients::CentralDifferenceEstimator;
use super::marching_cubes::MarchingCubes;
use super::types::{Mesh, Point3D, VoxelGrid};
use super::workspace::IsosurfaceWorkspace;
use crate::error::IsosurfaceError;

const DEADBAND: f32 = 1.0e-4;
const MAX_FIELD_MAG: f32 = 1.0e6;

/// Half-space cutting plane sealing volumetric solids at `coord[axis] <= max_coord`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutPlaneCap {
    /// Coordinate axis index (`0 = X`, `1 = Y`, `2 = Z`).
    pub axis: usize,
    /// Maximum retained world-space coordinate along `axis`.
    pub max_coord: f32,
}

/// Axis-aligned bounding box clamping boundary-capped [`MarchingCubes`] vertices onto planar faces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CappedBoxBounds {
    /// Minimum world-space corner `(x_min, y_min, z_min)`.
    pub min: Point3D,
    /// Maximum world-space corner `(x_max, y_max, z_max)`.
    pub max: Point3D,
}

#[inline]
#[verified_engine::verified]
fn stabilize_signed_sample(raw_s: f32) -> f32 {
    if !raw_s.is_finite() {
        return DEADBAND;
    }
    let clamped = raw_s.clamp(-MAX_FIELD_MAG, MAX_FIELD_MAG);
    if clamped <= 0.0 {
        clamped.min(-DEADBAND)
    } else {
        clamped.max(DEADBAND)
    }
}

#[inline]
#[verified_engine::verified]
fn axis_comp(pt: Point3D, axis: usize) -> f32 {
    match axis {
        0 => pt.x,
        1 => pt.y,
        _ => pt.z,
    }
}

#[inline]
#[verified_engine::verified]
fn set_axis_comp(pt: &mut Point3D, axis: usize, val: f32) {
    match axis {
        0 => pt.x = val,
        1 => pt.y = val,
        _ => pt.z = val,
    }
}

#[verified_engine::verified]
fn validate_source_grid(grid: &VoxelGrid) -> Result<(usize, usize, usize), IsosurfaceError> {
    let (nx, ny, nz) = (grid.width(), grid.height(), grid.depth());
    if nx < 2 || ny < 2 || nz < 2 {
        return Err(IsosurfaceError::InvalidGrid(
            "Source grid dimensions must be at least 2x2x2".to_string(),
        ));
    }
    let expected = nx
        .checked_mul(ny)
        .and_then(|wh| wh.checked_mul(nz))
        .ok_or_else(|| {
            IsosurfaceError::InvalidGrid("Source grid dimensions overflow".to_string())
        })?;
    if grid.data().len() < expected {
        return Err(IsosurfaceError::InvalidGrid(
            "Source grid buffer shorter than width * height * depth".to_string(),
        ));
    }
    Ok((nx, ny, nz))
}

#[verified_engine::verified]
fn populate_padded_interior(
    source_grid: &VoxelGrid,
    threshold: f32,
    inside_below: bool,
    padded_grid: &mut VoxelGrid,
) {
    let (nx, ny, nz) = (
        source_grid.width(),
        source_grid.height(),
        source_grid.depth(),
    );
    let (pw, ph) = (nx + 2, ny + 2);
    let src = source_grid.data();
    let dst = padded_grid.data_mut();
    dst.fill(DEADBAND);
    for z in 0..nz {
        for y in 0..ny {
            let src_row = (z * ny + y) * nx;
            let dst_row = ((z + 1) * ph + (y + 1)) * pw + 1;
            for x in 0..nx {
                let val = src[src_row + x];
                let raw_s = if inside_below {
                    val - threshold
                } else {
                    threshold - val
                };
                dst[dst_row + x] = stabilize_signed_sample(raw_s);
            }
        }
    }
}

#[inline]
#[verified_engine::verified]
fn padded_index_for_axis(
    axis: usize,
    slice: usize,
    u: usize,
    v: usize,
    pw: usize,
    ph: usize,
) -> usize {
    let (px, py, pz) = match axis {
        0 => (slice, u, v),
        1 => (u, slice, v),
        _ => (u, v, slice),
    };
    (pz * ph + py) * pw + px
}

#[verified_engine::verified]
fn apply_cut_plane_apron(
    padded_grid: &mut VoxelGrid,
    dims: (usize, usize, usize),
    box_min: Point3D,
    vs: Point3D,
    cut: CutPlaneCap,
) {
    let ax = cut.axis.min(2);
    let (nx, ny, nz) = dims;
    let (pw, ph) = (nx + 2, ny + 2);
    let (n_ax, n_u, n_v) = match ax {
        0 => (nx, ny, nz),
        1 => (ny, nx, nz),
        _ => (nz, nx, ny),
    };
    let step_a = axis_comp(vs, ax);
    let min_a = axis_comp(box_min, ax);
    let rel = ((cut.max_coord - min_a) / step_a).floor();
    let m = (rel.max(0.0) as usize + 1).min(n_ax);
    if m >= n_ax {
        return;
    }
    let a_m = min_a + ((m - 1) as f32) * step_a;
    let alpha = ((cut.max_coord - a_m) / step_a).clamp(0.0, 1.0);
    let buf = padded_grid.data_mut();
    for v in 1..=n_v {
        for u in 1..=n_u {
            let idx_m = padded_index_for_axis(ax, m, u, v, pw, ph);
            let idx_next = padded_index_for_axis(ax, m + 1, u, v, pw, ph);
            let v_m = buf[idx_m];
            let v_next = buf[idx_next];
            buf[idx_next] = if v_m < 0.0 {
                let v_cut = (1.0 - alpha) * v_m + alpha * v_next;
                if v_cut > DEADBAND {
                    v_next.max(DEADBAND)
                } else if alpha > 1.0e-3 {
                    (-v_m * (1.0 - alpha) / alpha).clamp(DEADBAND, MAX_FIELD_MAG)
                } else {
                    DEADBAND
                }
            } else {
                v_next.max(DEADBAND)
            };
            for p in (m + 2)..=(n_ax + 1) {
                let idx_p = padded_index_for_axis(ax, p, u, v, pw, ph);
                buf[idx_p] = DEADBAND;
            }
        }
    }
}

#[verified_engine::verified]
fn orient_and_clamp_mesh(mesh: &mut Mesh, bounds: CappedBoxBounds, vs: Point3D) {
    for tri in mesh.indices.as_chunks_mut::<3>().0 {
        tri.swap(1, 2);
    }
    let tol = Point3D::new(1.0e-4 * vs.x, 1.0e-4 * vs.y, 1.0e-4 * vs.z);
    for (v, n) in mesh.vertices.iter_mut().zip(mesh.normals.iter_mut()) {
        let mut cap_n = Point3D::new(0.0, 0.0, 0.0);
        if v.x <= bounds.min.x + tol.x {
            if v.x < bounds.min.x {
                v.x = bounds.min.x;
            }
            cap_n.x = -1.0;
        } else if v.x >= bounds.max.x - tol.x {
            if v.x > bounds.max.x {
                v.x = bounds.max.x;
            }
            cap_n.x = 1.0;
        }
        if v.y <= bounds.min.y + tol.y {
            if v.y < bounds.min.y {
                v.y = bounds.min.y;
            }
            cap_n.y = -1.0;
        } else if v.y >= bounds.max.y - tol.y {
            if v.y > bounds.max.y {
                v.y = bounds.max.y;
            }
            cap_n.y = 1.0;
        }
        if v.z <= bounds.min.z + tol.z {
            if v.z < bounds.min.z {
                v.z = bounds.min.z;
            }
            cap_n.z = -1.0;
        } else if v.z >= bounds.max.z - tol.z {
            if v.z > bounds.max.z {
                v.z = bounds.max.z;
            }
            cap_n.z = 1.0;
        }
        *n = resolve_outward_vertex_normal(*n, cap_n, vs);
    }
}

#[inline]
#[verified_engine::verified]
fn resolve_outward_vertex_normal(raw_n: Point3D, cap_n: Point3D, vs: Point3D) -> Point3D {
    if cap_n.magnitude() > 0.5 {
        return cap_n.normalize();
    }
    let scaled = if raw_n.is_finite() {
        Point3D::new(
            raw_n.x / vs.x.max(Point3D::EPSILON),
            raw_n.y / vs.y.max(Point3D::EPSILON),
            raw_n.z / vs.z.max(Point3D::EPSILON),
        )
    } else {
        Point3D::new(0.0, 0.0, 1.0)
    };
    let unit = scaled.normalize();
    if unit.is_finite() && unit.magnitude() > 0.5 {
        unit
    } else {
        Point3D::new(0.0, 0.0, 1.0)
    }
}

/// Extracts a closed, outward-wound 2-manifold solid from `source_grid` into `out_mesh` with zero
/// heap allocations when `padded_grid`, `workspace`, and `out_mesh` have sufficient capacity.
///
/// When `inside_below` is `true`, regions with `value <= threshold` form the solid interior;
/// when `false`, regions with `value >= threshold` form the solid interior.
#[verified_engine::verified]
pub fn extract_capped_isosurface_into(
    source_grid: &VoxelGrid,
    threshold: f32,
    inside_below: bool,
    cut_plane: Option<CutPlaneCap>,
    padded_grid: &mut VoxelGrid,
    workspace: &mut IsosurfaceWorkspace,
    out_mesh: &mut Mesh,
) -> Result<(), IsosurfaceError> {
    let (nx, ny, nz) = validate_source_grid(source_grid)?;
    out_mesh.clear();
    let vs = *source_grid.voxel_size();
    let box_min = *source_grid.origin();
    let mut box_max = Point3D::new(
        box_min.x + ((nx - 1) as f32) * vs.x,
        box_min.y + ((ny - 1) as f32) * vs.y,
        box_min.z + ((nz - 1) as f32) * vs.z,
    );
    if let Some(cut) = cut_plane {
        let ax = cut.axis.min(2);
        let min_a = axis_comp(box_min, ax);
        let max_a = axis_comp(box_max, ax);
        if !cut.max_coord.is_finite() || cut.max_coord <= min_a + 1.0e-4 * axis_comp(vs, ax) {
            return Ok(());
        }
        if cut.max_coord < max_a {
            set_axis_comp(&mut box_max, ax, cut.max_coord);
        }
    }
    padded_grid
        .reconfigure_in_place((nx + 2, ny + 2, nz + 2), vs, box_min - vs)
        .map_err(|e| IsosurfaceError::InvalidGrid(e.to_string()))?;
    populate_padded_interior(source_grid, threshold, inside_below, padded_grid);
    if let Some(cut) = cut_plane {
        apply_cut_plane_apron(padded_grid, (nx, ny, nz), box_min, vs, cut);
    }
    MarchingCubes::new(padded_grid, CentralDifferenceEstimator)
        .extract_into(0.0, workspace, out_mesh)?;
    orient_and_clamp_mesh(
        out_mesh,
        CappedBoxBounds {
            min: box_min,
            max: box_max,
        },
        vs,
    );
    Ok(())
}

/// Allocates a temporary 1-voxel apron grid and workspace to extract a closed, outward-wound solid [`Mesh`].
#[verified_engine::verified]
pub fn extract_capped_isosurface(
    source_grid: &VoxelGrid,
    threshold: f32,
    inside_below: bool,
    cut_plane: Option<CutPlaneCap>,
) -> Result<Mesh, IsosurfaceError> {
    let (nx, ny, nz) = validate_source_grid(source_grid)?;
    let vs = *source_grid.voxel_size();
    let mut padded_grid = VoxelGrid::builder()
        .dimensions(nx + 2, ny + 2, nz + 2)
        .data(vec![DEADBAND; (nx + 2) * (ny + 2) * (nz + 2)])
        .voxel_size(vs)
        .origin(*source_grid.origin() - vs)
        .build()
        .map_err(|e| IsosurfaceError::InvalidGrid(e.to_string()))?;
    let mut ws = IsosurfaceWorkspace::new_lazy(nx + 2, ny + 2);
    let mut mesh = Mesh::new();
    extract_capped_isosurface_into(
        source_grid,
        threshold,
        inside_below,
        cut_plane,
        &mut padded_grid,
        &mut ws,
        &mut mesh,
    )?;
    Ok(mesh)
}
