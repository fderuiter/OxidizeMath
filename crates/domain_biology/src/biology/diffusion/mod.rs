//! Spatial Diffusion Strategies
//!
//! This module provides strategies for computing the spatial diffusion term $D \nabla^2 u$
//! in reaction-diffusion systems.
//!
//! ## Performance Optimization
//!
//! Implementations like `FiniteDifference2D` use **Loop Splitting** to separate the "hot path" (interior points)
//! from boundary handling. This allows the compiler to:
//! 1. Vectorize the interior loop without conditional checks.
//! 2. Unroll loops for better instruction pipelining.
//! 3. Use `unsafe` indexing (with rigorous safety proofs) to eliminate bounds checks.

#[allow(missing_docs)]
pub mod fd1d;
#[allow(missing_docs)]
pub mod fd2d;
pub mod fd3d;

pub use fd1d::FiniteDifference1D;
pub use fd2d::{CellMask2D, FiniteDifference2D, MaskedFiniteDifference2D, ReactionCellRole};
pub use fd3d::FiniteDifference3D;
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;

/// Defines a strategy for computing spatial diffusion.
pub trait SpatialDiffusion<const N: usize> {
    /// Computes diffusion terms for each point and calls the closure.
    /// Internal iteration allows for optimization (loop fusion, SIMD).
    ///
    /// The closure `op` is called with `(index, current_vals, diff_vals)` where:
    /// * `index`: The linear index of the point.
    /// * `current_vals`: Current values of species at index.
    /// * `diff_vals`: The diffusion terms ($D \nabla^2 u$).
    #[verified_engine::verified]
    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]);

    /// Returns the core FusedStencilStepper configured for this spatial grid.
    fn stepper(&self) -> FusedStencilStepper;

    /// Fused step using the core math engine.
    #[verified_engine::verified]
    fn step_fused<K: crate::biology::morphogenesis::reaction::ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    );

    /// Applies the diffusion operator to the state vectors.
    ///
    /// Computes $D \nabla^2 u$ and stores the result in `out`.
    ///
    /// # Arguments
    /// * `state` - Input concentration slices.
    /// * `out` - Output buffers for diffusion terms.
    /// * `coeffs` - Diffusion coefficients.
    #[verified_engine::verified]
    fn apply(&self, state: [&[f64]; N], out: [&mut [f64]; N], coeffs: [f64; N]) {
        // Default implementation: calculate diffusion and write to buffer
        self.map_diffusion(state, coeffs, |i, _, diffs| {
            for s in 0..N {
                if i < out[s].len() {
                    out[s][i] = diffs[s];
                }
            }
        });
    }
}

// [cite:graph_parameters_rust]
