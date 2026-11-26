use super::boundary::BoundaryStrategy;
use super::geometry::GeometryStrategy;
use math_commons::math_kernel::types::{GridIndex, flatten_2d_index};
use oxidize_core::iteration::IterationPattern;

/// Strategy traversing 2D grid cells and supplying 5-point stencil indices `(center, left, right, up, down)`.
pub trait IterationStrategy {
    /// Invokes `op(idx, idx_l, idx_r, idx_u, idx_d)` across every cell in `geom`.
    #[verified_engine::verified]
    fn iterate<G, B, F>(&self, geom: &G, boundary: &B, op: F)
    where
        G: GeometryStrategy,
        B: BoundaryStrategy,
        F: FnMut(GridIndex, GridIndex, GridIndex, GridIndex, GridIndex);
}

/// Loop-splitting 2D traversal separating perimeter boundary resolution from branchless interior sweeps.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoopSplittingIteration;

impl IterationStrategy for LoopSplittingIteration {
    #[inline(always)]
    #[verified_engine::verified]
    fn iterate<G, B, F>(&self, geom: &G, boundary: &B, mut op: F)
    where
        G: GeometryStrategy,
        B: BoundaryStrategy,
        F: FnMut(GridIndex, GridIndex, GridIndex, GridIndex, GridIndex),
    {
        let width = *geom.width();
        let height = *geom.height();

        // Boundary points
        IterationPattern::for_each_boundary(width, height, |x, y| {
            let idx = GridIndex(flatten_2d_index(x, y, width));
            let (idx_l, idx_r, idx_u, idx_d) = boundary.neighbors(x, y, geom);
            op(idx, idx_l, idx_r, idx_u, idx_d);
        });

        // Interior points
        IterationPattern::for_each_interior(width, height, |x, y| {
            let idx = flatten_2d_index(x, y, width);
            let idx_l = idx - 1;
            let idx_r = idx + 1;
            let idx_u = idx - width;
            let idx_d = idx + width;
            op(
                GridIndex(idx),
                GridIndex(idx_l),
                GridIndex(idx_r),
                GridIndex(idx_u),
                GridIndex(idx_d),
            );
        });
    }
}
