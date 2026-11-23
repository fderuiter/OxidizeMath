//! Zero-allocation 2D advective-diffusive extensions for [`super::TuringSystem`].

use super::{
    DiffusionCoeff, FusedEulerSolver, ReactionKinetics, TuringSolverStrategy, TuringSystem,
};
use crate::biology::diffusion::{AdvectiveFiniteDifference2D, FiniteDifference2D};
use math_commons::math_kernel::types::{Dimension, StepSize};

impl<K: ReactionKinetics<2>> TuringSystem<2, K, AdvectiveFiniteDifference2D, FusedEulerSolver> {
    /// Creates a 2D two-species [`TuringSystem`] backed by [`AdvectiveFiniteDifference2D`]
    /// with preallocated velocity and internal zero-flux solid obstacle buffers.
    #[verified_engine::verified]
    pub fn new_advective_2d(
        dims: (Dimension, Dimension),
        d_u: DiffusionCoeff,
        d_v: DiffusionCoeff,
        spacing: (StepSize, StepSize),
        peclet: f64,
        kinetics: K,
    ) -> Self {
        let base = FiniteDifference2D::new(dims.0, dims.1, spacing.0, spacing.1);
        let advective = AdvectiveFiniteDifference2D::new(base, peclet);
        let total = Dimension(advective.grid_len());
        Self::new_with_kinetics(total, d_u, d_v, kinetics, advective)
    }
}

impl<const N: usize, K: ReactionKinetics<N>, S: TuringSolverStrategy<N>>
    TuringSystem<N, K, AdvectiveFiniteDifference2D, S>
{
    /// Returns the current dimensionless Péclet advection gain.
    #[verified_engine::verified]
    pub fn peclet(&self) -> f64 {
        self.diffusion.peclet
    }

    /// Updates the dimensionless Péclet advection gain.
    #[verified_engine::verified]
    pub fn set_peclet(&mut self, peclet: f64) {
        self.diffusion.peclet = peclet;
    }

    /// Returns the row-major horizontal velocity field.
    #[verified_engine::verified]
    pub fn velocity_x(&self) -> &[f64] {
        self.diffusion.velocity_x()
    }

    /// Returns the row-major vertical velocity field.
    #[verified_engine::verified]
    pub fn velocity_y(&self) -> &[f64] {
        self.diffusion.velocity_y()
    }

    /// Returns the row-major solid obstacle mask (`true` = zero-flux solid).
    #[verified_engine::verified]
    pub fn obstacles(&self) -> &[bool] {
        self.diffusion.obstacles()
    }

    /// Borrows mutable `(ux, uy, obstacles)` row-major buffers for zero-allocation fluid synchronization.
    #[verified_engine::verified]
    pub fn velocity_and_obstacles_mut(&mut self) -> (&mut [f64], &mut [f64], &mut [bool]) {
        self.diffusion.velocity_and_obstacles_mut()
    }

    /// Copies `(ux, uy, obstacles)` from row-major slices without reallocating.
    #[verified_engine::verified]
    pub fn set_velocity_and_obstacles(&mut self, ux: &[f64], uy: &[f64], obstacles: &[bool]) {
        self.diffusion.set_velocity_and_obstacles(ux, uy, obstacles);
    }

    /// Pins species values inside solid obstacle cells (e.g. steady background values) with zero allocations.
    #[verified_engine::verified]
    pub fn pin_solid_species(&mut self, solid_values: [f64; N]) {
        let obs = self.diffusion.obstacles();
        let n = self.state.len().min(obs.len());
        for s in 0..N {
            let slice = &mut self.state.concentrations[s];
            let next_slice = &mut self.next_state.concentrations[s];
            for i in 0..n {
                if obs[i] {
                    slice[i] = solid_values[s];
                    if i < next_slice.len() {
                        next_slice[i] = solid_values[s];
                    }
                }
            }
        }
    }

    /// Computes the combined diffusive + advective explicit CFL time-step bound.
    #[verified_engine::verified]
    pub fn max_cfl_dt(&self, safety: f64) -> f64 {
        self.diffusion.max_stable_dt(&self.diffusion_coeffs, safety)
    }
}
