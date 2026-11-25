//! Coupled 2D velocity-advected time stepping for [`TuringSystem`] over
//! row-major horizontal (`ux`) and vertical (`uy`) velocity slices.

use super::{FusedEulerSolver, ReactionKinetics, TuringState, TuringSystem};
use crate::biology::diffusion::AdvectedSpatialDiffusion;

impl<const N: usize, K: ReactionKinetics<N>, D: AdvectedSpatialDiffusion<N>>
    TuringSystem<N, K, D, FusedEulerSolver>
{
    /// Advances one zero-allocation coupled advection-diffusion-reaction step over
    /// row-major horizontal (`ux`) and vertical (`uy`) velocity slices.
    ///
    /// # Panics
    ///
    /// Panics if `ux.len() < self.state.len()` or `uy.len() < self.state.len()`.
    #[verified_engine::verified]
    pub fn step_advected(&mut self, dt: f64, ux: &[f64], uy: &[f64]) {
        let n = self.state.len();
        if n == 0 {
            return;
        }
        assert!(
            ux.len() >= n && uy.len() >= n,
            "Velocity slice lengths ({}, {}) must be at least grid size {}",
            ux.len(),
            uy.len(),
            n
        );
        if self.next_state.len() != n {
            self.next_state = TuringState::new(n);
        }

        let state_slices: [&[f64]; N] =
            std::array::from_fn(|i| self.state.concentrations[i].as_slice());
        let mut next_slices: [&mut [f64]; N] = std::array::from_fn(|_| &mut [] as &mut [f64]);
        let mut iter = self.next_state.concentrations.iter_mut();
        for slice in &mut next_slices {
            if let Some(vec) = iter.next() {
                *slice = vec.as_mut_slice();
            }
        }

        self.diffusion.step_advected_fused(
            state_slices,
            next_slices,
            ux,
            uy,
            dt,
            self.diffusion_coeffs,
            &self.kinetics,
        );

        std::mem::swap(&mut self.state, &mut self.next_state);
        self.time += dt;
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]
