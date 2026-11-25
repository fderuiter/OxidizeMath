//! Coupled 2D velocity-advected time stepping for [`TuringSystem`] over
//! row-major horizontal (`ux`) and vertical (`uy`) velocity slices.

use super::{FusedEulerSolver, ReactionKinetics, TuringState, TuringSystem};
use crate::biology::diffusion::AdvectedSpatialDiffusion;

impl<const N: usize, K: ReactionKinetics<N>, D: AdvectedSpatialDiffusion<N>>
    TuringSystem<N, K, D, FusedEulerSolver>
{
    /// Fallibly advances one zero-allocation coupled advection-diffusion-reaction step over
    /// row-major horizontal (`ux`) and vertical (`uy`) velocity slices.
    #[verified_engine::verified]
    pub fn try_step_advected(
        &mut self,
        dt: f64,
        ux: &[f64],
        uy: &[f64],
    ) -> Result<(), crate::biology::reaction_diffusion::ReactionDiffusionError> {
        use math_commons::error::MathError;
        use math_commons::math_kernel::types::Dimension;
        let n = self.state.len();
        if n == 0 {
            return Ok(());
        }
        if ux.len() < n {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(n),
                actual: Dimension(ux.len()),
            }
            .into());
        }
        if uy.len() < n {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(n),
                actual: Dimension(uy.len()),
            }
            .into());
        }
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
        Ok(())
    }

    /// Advances one zero-allocation coupled advection-diffusion-reaction step; prefer [`Self::try_step_advected`].
    #[verified_engine::verified]
    pub fn step_advected(&mut self, dt: f64, ux: &[f64], uy: &[f64]) {
        self.try_step_advected(dt, ux, uy)
            .expect("Velocity slice lengths must be at least grid size");
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]
