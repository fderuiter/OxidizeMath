//! Zero-allocation reusable workspace for [`MarchingCubes`] isosurface extraction (ADR 0025).
// [cite:isosurface_extraction]

use super::edge_cache::SliceEdgeCache;
use super::gradients::{CentralDifferenceEstimator, GradientEstimator};
use super::marching_cubes::{CubeData, MarchingCubes};
use super::tables::CUBE_EDGE_FLAGS;
use super::types::{Mesh, Point3D, VoxelGrid};
use crate::error::IsosurfaceError;

/// Reusable rolling-slice cache workspace enabling zero-allocation per-frame [`MarchingCubes`] extraction.
pub struct IsosurfaceWorkspace {
    pub(super) edge_cache: SliceEdgeCache,
    lock_mesh_capacity: bool,
}

impl IsosurfaceWorkspace {
    /// Creates an unallocated workspace that lazily allocates slice buffers on first active surface cube.
    #[verified_engine::verified]
    pub fn new_lazy(width: usize, height: usize) -> Self {
        Self {
            edge_cache: SliceEdgeCache::new(width.max(2), height.max(2)),
            lock_mesh_capacity: false,
        }
    }

    /// Pre-allocates rolling slice edge buffers up to `max_width x max_height` for hot-path reuse.
    #[verified_engine::verified]
    pub fn with_max_dimensions(max_width: usize, max_height: usize) -> Self {
        Self {
            edge_cache: SliceEdgeCache::with_capacity(max_width, max_height),
            lock_mesh_capacity: false,
        }
    }

    /// Pre-allocates rolling slice buffers and prevents `Mesh` growth past its reserved capacity.
    ///
    /// Pair with [`Mesh::with_capacity`] inside `VerifiedAllocator`-locked simulation frames.
    #[verified_engine::verified]
    pub fn with_locked_capacity(max_width: usize, max_height: usize) -> Self {
        Self {
            edge_cache: SliceEdgeCache::with_capacity(max_width, max_height),
            lock_mesh_capacity: true,
        }
    }
}

impl<'a, G: GradientEstimator> MarchingCubes<'a, G> {
    /// Extracts the `threshold` isosurface into `out_mesh` reusing `workspace` buffers in-place.
    #[verified_engine::verified]
    pub fn extract_into(
        &self,
        threshold: f32,
        workspace: &mut IsosurfaceWorkspace,
        out_mesh: &mut Mesh,
    ) -> Result<(), IsosurfaceError> {
        self.validate_grid()?;
        out_mesh.clear();
        let width = self.grid.width();
        let height = self.grid.height();
        let depth = self.grid.depth();
        workspace.edge_cache.reset_for_dimensions(width, height);

        let stride_y = width;
        let stride_z = width * height;

        for z in 0..depth - 1 {
            if z > 0 {
                workspace.edge_cache.advance_z();
            }
            let z_interior = z > 0 && z + 2 < depth;
            for y in 0..height - 1 {
                if y > 0 {
                    workspace.edge_cache.advance_y();
                }
                let row_is_interior = z_interior && y > 0 && y + 2 < height;
                let mut cached_gradients: Option<[Point3D; 4]> = None;

                for x in 0..width - 1 {
                    let base_idx = z * stride_z + y * stride_y + x;
                    let (values, index) =
                        self.get_cube_values_fast(base_idx, stride_y, stride_z, threshold);
                    if CUBE_EDGE_FLAGS[index] == 0 {
                        cached_gradients = None;
                        continue;
                    }
                    if workspace.lock_mesh_capacity
                        && (out_mesh.vertices.len() + 12 > out_mesh.vertices.capacity()
                            || out_mesh.normals.len() + 12 > out_mesh.normals.capacity()
                            || out_mesh.indices.len() + 15 > out_mesh.indices.capacity())
                    {
                        return Ok(());
                    }
                    let positions = self.get_cube_positions(x, y, z);
                    let (cube_normals, right_face) = self.compute_gradients_fast(
                        (x, y, z),
                        base_idx,
                        (stride_y, stride_z),
                        row_is_interior,
                        cached_gradients,
                    );
                    cached_gradients = Some(right_face);
                    let cube_data = CubeData {
                        values,
                        positions,
                        normals: cube_normals,
                        index,
                        coords: (x, y, z),
                    };
                    self.triangulate_cube(
                        &cube_data,
                        threshold,
                        &mut out_mesh.vertices,
                        &mut out_mesh.normals,
                        &mut out_mesh.indices,
                        &mut workspace.edge_cache,
                    );
                }
            }
        }
        Ok(())
    }
}

/// Extracts the `threshold` isosurface into `out_mesh` using [`CentralDifferenceEstimator`] and `workspace`.
#[verified_engine::verified]
pub fn extract_isosurface_into(
    grid: &VoxelGrid,
    threshold: f32,
    workspace: &mut IsosurfaceWorkspace,
    out_mesh: &mut Mesh,
) -> Result<(), IsosurfaceError> {
    let extractor = MarchingCubes::new(grid, CentralDifferenceEstimator);
    extractor.extract_into(threshold, workspace, out_mesh)
}
