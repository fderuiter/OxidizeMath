use super::geometry::GeometryStrategy;
use math_commons::math_kernel::types::{GridIndex, flatten_2d_index};
use oxidize_core::boundary::{BoundaryCondition, NeumannBoundary as CoreNeumann};

/// Strategy resolving 2D cardinal stencil neighbor indices on boundary cells.
pub trait BoundaryStrategy {
    /// Returns `(left, right, up, down)` flat grid indices for cell `(x, y)`.
    #[verified_engine::verified]
    fn neighbors<G: GeometryStrategy>(
        &self,
        x: usize,
        y: usize,
        geom: &G,
    ) -> (GridIndex, GridIndex, GridIndex, GridIndex);
}

/// Zero-flux Neumann boundary clamping out-of-bounds indices to the nearest edge cell.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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

/// Periodic (toroidal) boundary condition wrapping opposite grid edges seamlessly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
        if width == 0 || height == 0 {
            return (GridIndex(0), GridIndex(0), GridIndex(0), GridIndex(0));
        }

        let x_l = if x > 0 { x - 1 } else { width - 1 };
        let x_r = if x + 1 < width { x + 1 } else { 0 };
        let y_u = if y > 0 { y - 1 } else { height - 1 };
        let y_d = if y + 1 < height { y + 1 } else { 0 };

        (
            GridIndex(flatten_2d_index(x_l, y, width)),
            GridIndex(flatten_2d_index(x_r, y, width)),
            GridIndex(flatten_2d_index(x, y_u, width)),
            GridIndex(flatten_2d_index(x, y_d, width)),
        )
    }
}

/// Runtime-selectable 2D boundary mode for [`super::FiniteDifference2D`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Zero-flux clamped Neumann boundary.
    #[default]
    Neumann,
    /// Seamless toroidal wrap across opposite grid edges.
    Periodic,
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
