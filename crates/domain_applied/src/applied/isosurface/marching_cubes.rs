use super::edge_cache::{EMPTY_EDGE_VERTEX, SliceEdgeCache};
use super::gradients::{
    CentralDifferenceEstimator, GradientEstimator, ensure_valid_triangle_normals,
    interpolate_normal, interpolate_vertex,
};
use super::tables::{CUBE_EDGE_FLAGS, EDGE_CONNECTION, TRIANGLE_CONNECTION_TABLE};
use super::types::{Mesh, Point3D, VoxelGrid};
use crate::error::IsosurfaceError;

/// Data for a single cube in the grid.
struct CubeData {
    /// Values at the 8 corners.
    values: [f32; 8],
    /// Positions of the 8 corners.
    positions: [Point3D; 8],
    /// Normals at the 8 corners.
    normals: [Point3D; 8],
    /// The case index (0-255).
    index: usize,
    /// Coordinate of the cube in the grid (x, y, z).
    coords: (usize, usize, usize),
}

/// Marching Cubes algorithm implementation.
///
/// Converts a scalar field (voxel grid) into a polygonal mesh.
pub struct MarchingCubes<'a, G: GradientEstimator> {
    grid: &'a VoxelGrid,
    estimator: G,
}

impl<'a, G: GradientEstimator> MarchingCubes<'a, G> {
    /// Creates a new Marching Cubes extractor.
    #[verified_engine::verified]
    pub fn new(grid: &'a VoxelGrid, estimator: G) -> Self {
        Self { grid, estimator }
    }

    /// Extracts the isosurface for the given threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The scalar value at which to extract the surface. Values in the grid
    ///   less than this threshold are considered "inside" the volume, and values greater
    ///   than or equal to the threshold are considered "outside".
    ///
    /// # Returns
    ///
    /// Returns `Ok(Mesh)` containing the generated triangles on success.
    ///
    /// # Errors
    ///
    /// Returns an [`IsosurfaceError`] if the associated grid dimensions are invalid
    /// (e.g., less than 2x2x2) or if data length mismatches (via [`IsosurfaceError::Math`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_applied::__doctest_facade as math_explorer;
    /// use math_explorer::applied::isosurface::{CentralDifferenceEstimator, VoxelGrid, Point3D};
    /// use math_explorer::applied::isosurface::marching_cubes::MarchingCubes;
    ///
    /// // Define a minimal 2x2x2 grid.
    /// let grid = VoxelGrid::builder()
    /// .dimensions(2, 2, 2)
    /// .data(vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    /// .voxel_size(Point3D::new(1.0, 1.0, 1.0))
    /// .origin(Point3D::new(0.0, 0.0, 0.0))
    /// .build()
    /// .unwrap();
    ///
    /// // Instantiate the custom extractor.
    /// let extractor = MarchingCubes::new(&grid, CentralDifferenceEstimator);
    ///
    /// // Extract the surface at threshold 0.0.
    /// let mesh = extractor.extract(0.0).expect("Extraction failed");
    /// assert!(!mesh.indices.is_empty(), "Expected a generated mesh");
    /// ```
    #[verified_engine::verified]
    pub fn extract(&self, threshold: f32) -> Result<Mesh, IsosurfaceError> {
        let mut mesh = Mesh::default();
        let mut edge_cache = SliceEdgeCache::new(self.grid.width(), self.grid.height());
        self.extract_into(threshold, &mut edge_cache, &mut mesh)?;
        Ok(mesh)
    }

    /// Extracts the isosurface into pre-allocated [`SliceEdgeCache`] and [`Mesh`] buffers
    /// so inner loops execute with zero dynamic heap allocations when capacity suffices.
    #[verified_engine::verified]
    pub fn extract_into(
        &self,
        threshold: f32,
        edge_cache: &mut SliceEdgeCache,
        mesh: &mut Mesh,
    ) -> Result<(), IsosurfaceError> {
        self.validate_grid()?;
        mesh.vertices.clear();
        mesh.normals.clear();
        mesh.indices.clear();
        edge_cache.prepare(self.grid.width(), self.grid.height());

        let stride_y = self.grid.width();
        let stride_z = self.grid.width() * self.grid.height();
        let spacing = self.grid.voxel_size();
        let inv_voxel_size = Point3D::new(1.0 / spacing.x, 1.0 / spacing.y, 1.0 / spacing.z);

        for z in 0..self.grid.depth() - 1 {
            if z > 0 {
                edge_cache.advance_z();
            }
            let z_interior = z > 0 && z < self.grid.depth() - 2;

            for y in 0..self.grid.height() - 1 {
                if y > 0 {
                    edge_cache.advance_y();
                }
                let row_is_interior = z_interior && y > 0 && y < self.grid.height() - 2;
                let mut cached_gradients: Option<[Point3D; 4]> = None;

                for x in 0..self.grid.width() - 1 {
                    let base_idx = z * stride_z + y * stride_y + x;
                    let (values, index) =
                        self.get_cube_values_fast(base_idx, stride_y, stride_z, threshold);

                    if CUBE_EDGE_FLAGS[index] == 0 {
                        cached_gradients = None;
                        continue;
                    }

                    let positions = self.get_cube_positions(x, y, z);
                    let (cube_normals, right_face) = self.compute_gradients_fast(
                        (x, y, z),
                        base_idx,
                        (stride_y, stride_z),
                        inv_voxel_size,
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

                    self.triangulate_cube(&cube_data, threshold, mesh, edge_cache);
                }
            }
        }
        Ok(())
    }

    #[verified_engine::verified]
    fn validate_grid(&self) -> Result<(), IsosurfaceError> {
        if self.grid.width() < 2 || self.grid.height() < 2 || self.grid.depth() < 2 {
            return Err(IsosurfaceError::InvalidGrid(
                "Grid dimensions must be at least 2x2x2".to_string(),
            ));
        }

        let expected_len = self
            .grid
            .width()
            .checked_mul(self.grid.height())
            .and_then(|wh| wh.checked_mul(self.grid.depth()))
            .ok_or_else(|| {
                IsosurfaceError::InvalidGrid("Grid dimensions cause integer overflow".to_string())
            })?;

        if self.grid.data().len() < expected_len {
            return Err(IsosurfaceError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(expected_len),
                    actual: math_commons::math_kernel::types::Dimension(self.grid.data().len()),
                },
            ));
        }
        Ok(())
    }

    /// Fetches the scalar values for the 8 corners of a cube and computes its case index.
    ///
    /// This method is highly optimized for the inner loop of the marching cubes algorithm.
    /// It uses unchecked indexing and bitwise operations to compute the active edges.
    ///
    #[inline(always)]
    #[verified_engine::verified]
    fn get_cube_values_fast(
        &self,
        base_idx: usize,
        stride_y: usize,
        stride_z: usize,
        threshold: f32,
    ) -> ([f32; 8], usize) {
        let data = &self.grid.data();
        let mut values = [0.0; 8];
        let mut index = 0;

        // Helper macro to fetch and update index
        macro_rules! fetch {
            ($i:expr, $offset:expr, $bit:expr) => {
                let v = data.get(base_idx + $offset).copied().unwrap_or(0.0);
                values[$i] = v;
                if v < threshold {
                    index |= $bit;
                }
            };
        }

        fetch!(0, 0, 1);
        fetch!(1, 1, 2);
        fetch!(2, 1 + stride_y, 4);
        fetch!(3, stride_y, 8);
        fetch!(4, stride_z, 16);
        fetch!(5, 1 + stride_z, 32);
        fetch!(6, 1 + stride_y + stride_z, 64);
        fetch!(7, stride_y + stride_z, 128);

        (values, index)
    }

    #[inline]
    #[verified_engine::verified]
    fn get_cube_positions(&self, x: usize, y: usize, z: usize) -> [Point3D; 8] {
        let x_pos = self.grid.origin().x + (x as f32) * self.grid.voxel_size().x;
        let y_pos = self.grid.origin().y + (y as f32) * self.grid.voxel_size().y;
        let z_pos = self.grid.origin().z + (z as f32) * self.grid.voxel_size().z;

        let next_x = x_pos + self.grid.voxel_size().x;
        let next_y = y_pos + self.grid.voxel_size().y;
        let next_z = z_pos + self.grid.voxel_size().z;

        [
            Point3D::new(x_pos, y_pos, z_pos),    // 0
            Point3D::new(next_x, y_pos, z_pos),   // 1
            Point3D::new(next_x, next_y, z_pos),  // 2
            Point3D::new(x_pos, next_y, z_pos),   // 3
            Point3D::new(x_pos, y_pos, next_z),   // 4
            Point3D::new(next_x, y_pos, next_z),  // 5
            Point3D::new(next_x, next_y, next_z), // 6
            Point3D::new(x_pos, next_y, next_z),  // 7
        ]
    }

    /// Computes physical gradient vectors for the 8 corners of a cube with X-axis face caching.
    #[inline(always)]
    #[verified_engine::verified]
    fn compute_gradients_fast(
        &self,
        coords: (usize, usize, usize),
        base_idx: usize,
        strides: (usize, usize),
        inv_voxel_size: Point3D,
        row_is_interior: bool,
        cached_left_face: Option<[Point3D; 4]>,
    ) -> ([Point3D; 8], [Point3D; 4]) {
        let (x, y, z) = coords;
        let (stride_y, stride_z) = strides;
        let data = self.grid.data();
        let can_use_fast_path = row_is_interior && x > 0 && x < self.grid.width() - 2;

        let left_face = if let Some(grads) = cached_left_face {
            grads
        } else if can_use_fast_path {
            self.face_gradients_fast(data, base_idx, stride_y, stride_z, inv_voxel_size)
        } else {
            self.face_gradients_checked(x, y, z)
        };

        let right_face = if can_use_fast_path {
            self.face_gradients_fast(data, base_idx + 1, stride_y, stride_z, inv_voxel_size)
        } else {
            self.face_gradients_checked(x + 1, y, z)
        };

        let normals = [
            left_face[0],
            right_face[0],
            right_face[1],
            left_face[1],
            left_face[2],
            right_face[2],
            right_face[3],
            left_face[3],
        ];
        (normals, right_face)
    }

    #[inline(always)]
    #[verified_engine::verified]
    fn face_gradients_fast(
        &self,
        data: &[f32],
        x_idx: usize,
        stride_y: usize,
        stride_z: usize,
        inv_voxel_size: Point3D,
    ) -> [Point3D; 4] {
        [
            self.estimator
                .gradient_fast_scaled(data, x_idx, stride_y, stride_z, inv_voxel_size),
            self.estimator.gradient_fast_scaled(
                data,
                x_idx + stride_y,
                stride_y,
                stride_z,
                inv_voxel_size,
            ),
            self.estimator.gradient_fast_scaled(
                data,
                x_idx + stride_z,
                stride_y,
                stride_z,
                inv_voxel_size,
            ),
            self.estimator.gradient_fast_scaled(
                data,
                x_idx + stride_y + stride_z,
                stride_y,
                stride_z,
                inv_voxel_size,
            ),
        ]
    }

    #[inline(always)]
    #[verified_engine::verified]
    fn face_gradients_checked(&self, x: usize, y: usize, z: usize) -> [Point3D; 4] {
        [
            self.estimator.gradient(self.grid, x, y, z),
            self.estimator.gradient(self.grid, x, y + 1, z),
            self.estimator.gradient(self.grid, x, y, z + 1),
            self.estimator.gradient(self.grid, x, y + 1, z + 1),
        ]
    }

    #[inline]
    #[verified_engine::verified]
    fn triangulate_cube(
        &self,
        cube: &CubeData,
        threshold: f32,
        mesh: &mut Mesh,
        edge_cache: &mut SliceEdgeCache,
    ) {
        let edge_flags = CUBE_EDGE_FLAGS[cube.index];
        let mut edge_vertex_indices = [0usize; 12];
        let (x, y, _z) = cube.coords;

        for i in 0..12 {
            if (edge_flags & (1 << i)) != 0 {
                let slot = edge_cache.slot_mut(x, y, i);
                let cached = *slot;

                if cached != EMPTY_EDGE_VERTEX {
                    edge_vertex_indices[i] = cached as usize;
                } else {
                    let v1_idx = EDGE_CONNECTION[i][0];
                    let v2_idx = EDGE_CONNECTION[i][1];

                    let v = interpolate_vertex(
                        cube.positions[v1_idx],
                        cube.values[v1_idx],
                        cube.positions[v2_idx],
                        cube.values[v2_idx],
                        threshold,
                    );
                    let n = interpolate_normal(
                        cube.normals[v1_idx],
                        cube.values[v1_idx],
                        cube.normals[v2_idx],
                        cube.values[v2_idx],
                        threshold,
                    );

                    let new_idx = mesh.vertices.len();
                    mesh.vertices.push(v);
                    mesh.normals.push(n);
                    *slot = new_idx as u32;
                    edge_vertex_indices[i] = new_idx;
                }
            }
        }

        // Emit outward counter-clockwise triangles for {V < threshold} solids.
        let mut i = 0;
        loop {
            let v1_lookup = TRIANGLE_CONNECTION_TABLE[cube.index][i];
            if v1_lookup == -1 {
                break;
            }
            let v2_lookup = TRIANGLE_CONNECTION_TABLE[cube.index][i + 1];
            let v3_lookup = TRIANGLE_CONNECTION_TABLE[cube.index][i + 2];

            let i0 = edge_vertex_indices[v1_lookup as usize];
            let i1 = edge_vertex_indices[v3_lookup as usize];
            let i2 = edge_vertex_indices[v2_lookup as usize];
            mesh.indices.push(i0);
            mesh.indices.push(i1);
            mesh.indices.push(i2);
            ensure_valid_triangle_normals(mesh, i0, i1, i2);

            i += 3;
        }
    }
}

/// Extracts the isosurface for the given threshold using the Marching Cubes algorithm.
///
/// Converts a scalar field (voxel grid) into a polygonal mesh using a central
/// difference estimator for normal calculation.
///
/// # Arguments
///
/// * `grid` - A reference to the [`VoxelGrid`] representing the volumetric data.
/// * `threshold` - The scalar value at which to extract the surface.
///
/// # Returns
///
/// Returns `Ok(Mesh)` containing the generated triangles on success, or
/// an `IsosurfaceError` (e.g. `IsosurfaceError::Math`) if the grid is invalid or data length mismatches.
///
/// # Examples
///
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::isosurface::{extract_isosurface, VoxelGrid, Point3D};
///
/// // Create a tiny 2x2x2 grid representing a corner
/// let grid = VoxelGrid::builder()
/// .dimensions(2, 2, 2)
/// .data(vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
/// .voxel_size(Point3D::new(1.0, 1.0, 1.0))
/// .origin(Point3D::new(0.0, 0.0, 0.0))
/// .build()
/// .unwrap();
///
/// // Extract the surface where value == 0.0
/// let mesh = extract_isosurface(&grid, 0.0).expect("Failed to extract isosurface");
///
/// // The simple gradient will yield triangles spanning across the middle.
/// assert!(!mesh.indices.is_empty(), "Mesh should contain triangles");
/// ```
#[verified_engine::verified]
pub fn extract_isosurface(grid: &VoxelGrid, threshold: f32) -> Result<Mesh, IsosurfaceError> {
    let extractor = MarchingCubes::new(grid, CentralDifferenceEstimator);
    extractor.extract(threshold)
}
