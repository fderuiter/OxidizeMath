use super::types::{Point3D, VoxelGrid};

/// Strategy trait for estimating gradients in a voxel grid.
pub trait GradientEstimator {
    /// Computes the gradient at a specific 3D grid coordinate.
    ///
    /// This method safely computes the gradient by checking boundary conditions.
    /// If the coordinate is on the boundary of the grid, a forward or backward difference
    /// is used instead of a central difference.
    ///
    /// # Arguments
    ///
    /// * `grid` - The [`VoxelGrid`] containing the scalar field data.
    /// * `x` - The x-coordinate in the grid.
    /// * `y` - The y-coordinate in the grid.
    /// * `z` - The z-coordinate in the grid.
    ///
    /// # Returns
    ///
    /// A [`Point3D`] representing the gradient vector $( \partial f / \partial x, \partial f / \partial y, \partial f / \partial z )$.
    ///
    /// # Examples
    ///
    /// ```
    /// # use domain_applied::__doctest_facade as math_explorer;
    /// use math_explorer::applied::isosurface::types::{VoxelGrid, Point3D};
    /// use math_explorer::applied::isosurface::gradients::{GradientEstimator, CentralDifferenceEstimator};
    ///
    /// let data = vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0];
    /// let grid = VoxelGrid::builder()
    ///     .dimensions(3, 3, 1)
    ///     .data(data)
    ///     .voxel_size(Point3D::new(1.0, 1.0, 1.0))
    ///     .origin(Point3D::new(0.0, 0.0, 0.0))
    ///     .build()
    ///     .unwrap();
    /// let estimator = CentralDifferenceEstimator;
    ///
    /// let grad = estimator.gradient(&grid, 1, 1, 0);
    /// assert_eq!(grad.x, 1.0); // (grid[2, 1, 0] - grid[0, 1, 0]) / 2.0 = (3.0 - 1.0) / 2.0
    /// assert_eq!(grad.y, 1.0); // (grid[1, 2, 0] - grid[1, 0, 0]) / 2.0 = (3.0 - 1.0) / 2.0
    /// assert_eq!(grad.z, -2.0); // Boundary condition fallback: grid[1, 1, 1] - grid[1, 1, 0] = 0.0 - 2.0 = -2.0
    /// ```
    #[verified_engine::verified]
    fn gradient(&self, grid: &VoxelGrid, x: usize, y: usize, z: usize) -> Point3D;

    /// Computes the gradient at a specific linear index using direct memory access.
    ///
    /// This is a high-performance, unchecked variant intended for use in inner loops
    /// where bounds checking has already been performed or is guaranteed not to fail
    /// due to structural padding.
    ///
    /// # Arguments
    ///
    /// * `data` - A flat slice representing the 3D grid data.
    /// * `idx` - The 1D linear index corresponding to the $(x, y, z)$ coordinate.
    /// * `stride_y` - The stride to advance one unit in the y-direction (typically `width`).
    /// * `stride_z` - The stride to advance one unit in the z-direction (typically `width * height`).
    ///
    /// # Returns
    ///
    /// A [`Point3D`] representing the gradient vector.
    ///
    /// # Panics
    ///
    /// This method will panic if the bounds check fails.
    #[verified_engine::verified]
    fn gradient_fast(&self, data: &[f32], idx: usize, stride_y: usize, stride_z: usize) -> Point3D;
}

/// Estimates gradients using the Central Difference method.
///
/// $ \nabla f \approx \left( \frac{f(x+1) - f(x-1)}{2}, \frac{f(y+1) - f(y-1)}{2}, \frac{f(z+1) - f(z-1)}{2} \right) $
pub struct CentralDifferenceEstimator;

impl GradientEstimator for CentralDifferenceEstimator {
    #[inline]
    #[verified_engine::verified]
    fn gradient(&self, grid: &VoxelGrid, x: usize, y: usize, z: usize) -> Point3D {
        let dx = if x == 0 {
            grid.get(x + 1, y, z) - grid.get(x, y, z)
        } else if x == grid.width() - 1 {
            grid.get(x, y, z) - grid.get(x - 1, y, z)
        } else {
            (grid.get(x + 1, y, z) - grid.get(x - 1, y, z)) / 2.0
        };

        let dy = if y == 0 {
            grid.get(x, y + 1, z) - grid.get(x, y, z)
        } else if y == grid.height() - 1 {
            grid.get(x, y, z) - grid.get(x, y - 1, z)
        } else {
            (grid.get(x, y + 1, z) - grid.get(x, y - 1, z)) / 2.0
        };

        let dz = if z == 0 {
            grid.get(x, y, z + 1) - grid.get(x, y, z)
        } else if z == grid.depth() - 1 {
            grid.get(x, y, z) - grid.get(x, y, z - 1)
        } else {
            (grid.get(x, y, z + 1) - grid.get(x, y, z - 1)) / 2.0
        };

        Point3D::new(dx, dy, dz)
    }

    #[inline(always)]
    #[verified_engine::verified]
    fn gradient_fast(&self, data: &[f32], idx: usize, stride_y: usize, stride_z: usize) -> Point3D {
        let dx = (data
            .get(idx.checked_add(1).unwrap_or(idx))
            .copied()
            .unwrap_or(0.0)
            - data
                .get(idx.checked_sub(1).unwrap_or(idx))
                .copied()
                .unwrap_or(0.0))
            * 0.5;
        let dy = (data
            .get(idx.checked_add(stride_y).unwrap_or(idx))
            .copied()
            .unwrap_or(0.0)
            - data
                .get(idx.checked_sub(stride_y).unwrap_or(idx))
                .copied()
                .unwrap_or(0.0))
            * 0.5;
        let dz = (data
            .get(idx.checked_add(stride_z).unwrap_or(idx))
            .copied()
            .unwrap_or(0.0)
            - data
                .get(idx.checked_sub(stride_z).unwrap_or(idx))
                .copied()
                .unwrap_or(0.0))
            * 0.5;
        Point3D::new(dx, dy, dz)
    }
}
