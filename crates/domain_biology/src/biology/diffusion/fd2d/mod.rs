use super::SpatialDiffusion;

#[allow(missing_docs)]
pub mod boundary;
#[allow(missing_docs)]
pub mod geometry;
#[allow(missing_docs)]
pub mod iteration;
#[cfg(test)]
mod tests;

use boundary::{BoundaryMode, NeumannBoundary};
use geometry::{Cartesian2D, GeometryStrategy};
use iteration::{IterationStrategy, LoopSplittingIteration};
use math_commons::math_kernel::types::{Dimension, StepSize};
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;

/// A 2D Finite Difference implementation using a 5-point stencil.
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifference2D {
    #[allow(missing_docs)]
    pub geometry: Cartesian2D,
    #[allow(missing_docs)]
    pub boundary: NeumannBoundary,
    /// Active boundary condition mode (`Neumann` by default, or `Periodic` for toroidal wrap).
    pub boundary_mode: BoundaryMode,
    #[allow(missing_docs)]
    pub iteration: LoopSplittingIteration,
}

impl FiniteDifference2D {
    /// Creates a new 2D finite-difference operator with zero-flux Neumann boundaries.
    #[verified_engine::verified]
    pub fn new(width: Dimension, height: Dimension, dx: StepSize, dy: StepSize) -> Self {
        Self {
            geometry: Cartesian2D {
                width,
                height,
                dx,
                dy,
            },
            boundary: NeumannBoundary,
            boundary_mode: BoundaryMode::Neumann,
            iteration: LoopSplittingIteration,
        }
    }

    /// Creates a new 2D finite-difference operator with periodic (toroidal) boundaries.
    #[verified_engine::verified]
    pub fn new_periodic(width: Dimension, height: Dimension, dx: StepSize, dy: StepSize) -> Self {
        Self::new(width, height, dx, dy).with_boundary_mode(BoundaryMode::Periodic)
    }

    /// Returns a copy configured with the specified [`BoundaryMode`].
    #[verified_engine::verified]
    pub fn with_boundary_mode(mut self, mode: BoundaryMode) -> Self {
        self.boundary_mode = mode;
        self
    }

    /// Returns `true` when opposite edges wrap toroidally via [`BoundaryMode::Periodic`].
    #[verified_engine::verified]
    pub fn is_periodic(&self) -> bool {
        self.boundary_mode == BoundaryMode::Periodic
    }
}

impl crate::biology::reaction_diffusion::DiffusionModel for FiniteDifference2D {
    #[verified_engine::verified]
    fn apply(
        &self,
        state: &crate::biology::reaction_diffusion::ChemicalState,
        out: &mut crate::biology::reaction_diffusion::ChemicalState,
        coeffs: &[f64],
    ) {
        let n_species = state.num_species();
        if n_species == 0 {
            return;
        }

        let n_grid = self.geometry.size();
        assert_eq!(
            state.grid_size(),
            n_grid,
            "ChemicalState grid size mismatch with FiniteDifference2D"
        );
        assert_eq!(out.grid_size(), n_grid, "Output state grid size mismatch");
        assert_eq!(
            coeffs.len(),
            n_species,
            "Diffusion coefficients count mismatch"
        );

        let inv_dx_sq = 1.0 / (*self.geometry.dx() * *self.geometry.dx());
        let inv_dy_sq = 1.0 / (*self.geometry.dy() * *self.geometry.dy());

        for (s, coeff) in coeffs.iter().enumerate().take(n_species) {
            let src = state.species(s);
            let dst = out.species_mut(s);
            let cx = *coeff * inv_dx_sq;
            let cy = *coeff * inv_dy_sq;
            let c_center = -2.0 * (cx + cy);

            self.iteration.iterate(
                &self.geometry,
                &self.boundary_mode,
                |idx, idx_l, idx_r, idx_u, idx_d| {
                    dst[*idx] = (src[*idx_r] + src[*idx_l]) * cx
                        + (src[*idx_d] + src[*idx_u]) * cy
                        + src[*idx] * c_center;
                },
            );
        }
    }
}

impl<const N: usize> SpatialDiffusion<N> for FiniteDifference2D {
    fn stepper(&self) -> FusedStencilStepper {
        FusedStencilStepper::new_2d(self.geometry.dx, self.geometry.dy)
    }

    #[verified_engine::verified]
    fn step_fused<K: crate::biology::morphogenesis::reaction::ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        if N == 0 || self.geometry.size() == 0 {
            return;
        }

        let dim = (*self.geometry.width, *self.geometry.height);
        let stepper = SpatialDiffusion::<N>::stepper(self);
        let rhs_fn = |_i: usize,
                      curr: [f64; N],
                      left: [f64; N],
                      right: [f64; N],
                      up: [f64; N],
                      down: [f64; N],
                      ops: &pure_math::pure_math::analysis::pde::fused_stepper::StencilOperators| {
            let mut rhs = [0.0; N];
            let rates = kinetics.reaction(curr);
            for s in 0..N {
                let d2u = ops.central_diff_2nd_2d(curr[s], left[s], right[s], up[s], down[s]);
                rhs[s] = coeffs[s] * d2u + rates[s];
            }
            rhs
        };

        match self.boundary_mode {
            BoundaryMode::Neumann => {
                stepper.step_2d_coupled_neumann(dim, state, next_state, dt, 1.0, rhs_fn);
            }
            BoundaryMode::Periodic => {
                stepper.step_2d_coupled_periodic(dim, state, next_state, dt, 1.0, rhs_fn);
            }
        }
    }

    #[verified_engine::verified]
    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], mut op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]),
    {
        if N == 0 {
            return;
        }

        let n = self.geometry.size();
        if n == 0 {
            return;
        }
        assert!(
            state[0].len() >= n,
            "Buffer size mismatch in FiniteDifference2D"
        );

        let inv_dx_sq = 1.0 / (*self.geometry.dx() * *self.geometry.dx());
        let inv_dy_sq = 1.0 / (*self.geometry.dy() * *self.geometry.dy());

        let mut cx = [0.0; N];
        let mut cy = [0.0; N];
        let mut c_center = [0.0; N];
        for (s, coeff) in coeffs.iter().enumerate().take(N) {
            cx[s] = coeff * inv_dx_sq;
            cy[s] = coeff * inv_dy_sq;
            c_center[s] = -2.0 * (cx[s] + cy[s]);
        }

        for (s, buffer) in state.iter().enumerate().take(N) {
            assert!(
                buffer.len() >= n,
                "Buffer too small for diffusion (species {})",
                s
            );
        }

        self.iteration.iterate(
            &self.geometry,
            &self.boundary_mode,
            |idx, idx_l, idx_r, idx_u, idx_d| {
                let mut current_vals = [0.0; N];
                let mut diff_vals = [0.0; N];
                for s in 0..N {
                    let u = state[s];
                    let u_curr = u[*idx];
                    current_vals[s] = u_curr;
                    diff_vals[s] = (u[*idx_r] + u[*idx_l]) * cx[s]
                        + (u[*idx_d] + u[*idx_u]) * cy[s]
                        + u_curr * c_center[s];
                }
                op(*idx, current_vals, diff_vals);
            },
        );
    }
}
// [cite:essay]
