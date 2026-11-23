use super::boundary::BoundaryStrategy;
use super::geometry::GeometryStrategy;
use math_commons::math_kernel::types::{GridIndex, flatten_2d_index};
use oxidize_core::iteration::IterationPattern;

#[allow(missing_docs)]
pub trait IterationStrategy {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn iterate<G, B, F>(&self, geom: &G, boundary: &B, op: F)
    where
        G: GeometryStrategy,
        B: BoundaryStrategy,
        F: FnMut(GridIndex, GridIndex, GridIndex, GridIndex, GridIndex);
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(missing_docs)]
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
