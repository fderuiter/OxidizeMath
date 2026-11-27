use super::geometry::GeometryStrategy;
use math_commons::math_kernel::types::GridIndex;
use oxidize_core::boundary::{BoundaryCondition, NeumannBoundary as CoreNeumann};

/// Strategy resolving cardinal 2D neighbor grid indices at domain boundaries.
pub trait BoundaryStrategy {
    /// Resolves `(left, right, up, down)` flattened neighbor indices for cell `(x, y)`.
    #[verified_engine::verified]
    fn neighbors<G: GeometryStrategy>(
        &self,
        x: usize,
        y: usize,
        geom: &G,
    ) -> (GridIndex, GridIndex, GridIndex, GridIndex);
}

/// Zero-flux homogeneous Neumann boundary condition ($\partial_n u = 0$) on 2D grids.
#[derive(Debug, Clone, Copy, Default)]
pub struct NeumannBoundary;

impl BoundaryStrategy for NeumannBoundary {
    #[inline(always)]
    #[verified_engine::verified]
    fn neighbors<G: GeometryStrategy>(
        &self,
        x: usize,
        y: usize,
        geom: &G,
    ) -> (GridIndex, GridIndex, GridIndex, GridIndex) {
        let width = *geom.width();
        let height = *geom.height();

        let bc = CoreNeumann;
        // Neighbors: left, right, up, down
        let idx_l = bc
            .resolve(x as isize - 1, y as isize, width, height)
            .unwrap();
        let idx_r = bc
            .resolve(x as isize + 1, y as isize, width, height)
            .unwrap();
        let idx_u = bc
            .resolve(x as isize, y as isize - 1, width, height)
            .unwrap();
        let idx_d = bc
            .resolve(x as isize, y as isize + 1, width, height)
            .unwrap();

        (
            GridIndex(idx_l),
            GridIndex(idx_r),
            GridIndex(idx_u),
            GridIndex(idx_d),
        )
    }
}
