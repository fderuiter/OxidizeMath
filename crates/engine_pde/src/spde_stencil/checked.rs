//! Checked `#![no_std]` companion steppers for [`ExplicitSpdeStencilStepper`] and [`SplitStepSpdeStencilStepper`].
// [cite:engine_pde]
use crate::spde_stencil::{ExplicitSpdeStencilStepper, SplitStepSpdeStencilStepper};
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, SolverWorkspace};
use rand_core::RngCore;

fn check_non_neg_param(value: f64) -> Result<(), MathError> {
    if !(value >= 0.0 && value.is_finite()) {
        return Err(MathError::OutOfBounds {
            value,
            min: 0.0,
            max: f64::INFINITY,
        });
    }
    Ok(())
}

fn check_grid_and_workspace(
    field_len: usize,
    width: usize,
    height: usize,
    workspace_len: usize,
    mult: usize,
) -> Result<(), MathError> {
    let total = width
        .checked_mul(height)
        .ok_or(MathError::WorkspaceTooSmall {
            required: usize::MAX,
            available: field_len,
        })?;
    if field_len < total {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(total),
            actual: Dimension(field_len),
        });
    }
    let required_ws = total
        .checked_mul(mult)
        .ok_or(MathError::WorkspaceTooSmall {
            required: usize::MAX,
            available: workspace_len,
        })?;
    if workspace_len < required_ws {
        return Err(MathError::WorkspaceTooSmall {
            required: required_ws,
            available: workspace_len,
        });
    }
    Ok(())
}

impl ExplicitSpdeStencilStepper {
    /// Checked companion to [`Self::step_1d`] validating `dt >= 0` and workspace capacity.
    #[verified_engine::verified]
    pub fn try_step_1d<F, R>(
        &self,
        u: &mut [f64],
        dt: f64,
        deterministic_rhs: F,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError>
    where
        F: FnMut(usize, f64, f64, f64) -> f64,
        R: RngCore + ?Sized,
    {
        check_non_neg_param(dt)?;
        check_grid_and_workspace(u.len(), u.len(), 1, workspace.len(), 2)?;
        self.step_1d(u, dt, deterministic_rhs, workspace, rng);
        debug_assert!(workspace.len() >= 2 * u.len());
        Ok(())
    }

    /// Checked companion to [`Self::step_2d`] validating grid buffer, `dt >= 0`, and workspace capacity.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_2d<F, R>(
        &self,
        u: &mut [f64],
        width: usize,
        height: usize,
        dt: f64,
        deterministic_rhs: F,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError>
    where
        F: FnMut(usize, usize, f64, (f64, f64), f64) -> f64,
        R: RngCore + ?Sized,
    {
        check_non_neg_param(dt)?;
        check_grid_and_workspace(u.len(), width, height, workspace.len(), 5)?;
        self.step_2d(u, width, height, dt, deterministic_rhs, workspace, rng);
        debug_assert!(u.len() >= width * height);
        Ok(())
    }

    /// Checked companion to [`Self::step_cahn_hilliard_2d`] validating grid, parameters, and workspace capacity.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_cahn_hilliard_2d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError> {
        check_non_neg_param(mobility)?;
        check_non_neg_param(kappa)?;
        check_non_neg_param(dt)?;
        check_grid_and_workspace(phi.len(), width, height, workspace.len(), 5)?;
        self.step_cahn_hilliard_2d(phi, width, height, mobility, kappa, dt, workspace, rng);
        debug_assert!(phi.len() >= width * height);
        Ok(())
    }
}

impl SplitStepSpdeStencilStepper {
    /// Checked companion to [`Self::step_reaction_diffusion_2d`] validating grid, `diffusivity >= 0`, `dt >= 0`, and workspace capacity.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_reaction_diffusion_2d<F, R>(
        &self,
        u: &mut [f64],
        width: usize,
        height: usize,
        diffusivity: f64,
        dt: f64,
        reaction: F,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError>
    where
        F: FnMut(f64) -> f64,
        R: RngCore + ?Sized,
    {
        check_non_neg_param(diffusivity)?;
        check_non_neg_param(dt)?;
        check_grid_and_workspace(u.len(), width, height, workspace.len(), 3)?;
        self.step_reaction_diffusion_2d(
            u,
            width,
            height,
            diffusivity,
            dt,
            reaction,
            workspace,
            rng,
        );
        debug_assert!(u.len() >= width * height);
        Ok(())
    }
}
