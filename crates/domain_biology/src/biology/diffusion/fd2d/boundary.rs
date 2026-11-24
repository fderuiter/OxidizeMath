use super::geometry::GeometryStrategy;
use math_commons::math_kernel::types::{GridIndex, flatten_2d_index};
use oxidize_core::boundary::{BoundaryCondition, NeumannBoundary as CoreNeumann};

/// Spatial boundary condition mode for 2D Cartesian reaction-diffusion stencils.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryMode {
    /// Zero-flux reflecting boundaries ($\nabla u \cdot \mathbf{n} = 0$).
    #[default]
    Neumann,
    /// Toroidal wrap-around boundaries ($u(x + W, y) = u(x, y + H) = u(x, y)$).
    Periodic,
}

#[allow(missing_docs)]
pub trait BoundaryStrategy {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn neighbors<G: GeometryStrategy>(
        &self,
        x: usize,
        y: usize,
        geom: &G,
    ) -> (GridIndex, GridIndex, GridIndex, GridIndex);
}

#[derive(Debug, Clone, Copy, Default)]
#[allow(missing_docs)]
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

/// Toroidal 2D periodic boundary strategy wrapping opposite grid edges.
#[derive(Debug, Clone, Copy, Default)]
pub struct PeriodicBoundary;

impl BoundaryStrategy for PeriodicBoundary {
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
        let x_l = if x > 0 {
            x - 1
        } else {
            width.saturating_sub(1)
        };
        let x_r = if x + 1 < width { x + 1 } else { 0 };
        let y_u = if y > 0 {
            y - 1
        } else {
            height.saturating_sub(1)
        };
        let y_d = if y + 1 < height { y + 1 } else { 0 };
        (
            GridIndex(flatten_2d_index(x_l, y, width)),
            GridIndex(flatten_2d_index(x_r, y, width)),
            GridIndex(flatten_2d_index(x, y_u, width)),
            GridIndex(flatten_2d_index(x, y_d, width)),
        )
    }
}

impl BoundaryStrategy for BoundaryMode {
    #[inline(always)]
    #[verified_engine::verified]
    fn neighbors<G: GeometryStrategy>(
        &self,
        x: usize,
        y: usize,
        geom: &G,
    ) -> (GridIndex, GridIndex, GridIndex, GridIndex) {
        match self {
            Self::Neumann => NeumannBoundary.neighbors(x, y, geom),
            Self::Periodic => PeriodicBoundary.neighbors(x, y, geom),
        }
    }
}
