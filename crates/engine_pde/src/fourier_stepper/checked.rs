//! Checked `#![no_std]` companion steppers for [`SemiImplicitFourierStepper`].
// [cite:engine_pde]
use crate::fourier_stepper::SemiImplicitFourierStepper;
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, SolverWorkspace};
use rand_core::RngCore;

fn check_non_negative(value: f64) -> Result<(), MathError> {
    if !(value >= 0.0 && value.is_finite()) {
        return Err(MathError::OutOfBounds {
            value,
            min: 0.0,
            max: f64::INFINITY,
        });
    }
    Ok(())
}

fn check_radix2_1d(n: usize, workspace_len: usize) -> Result<(), MathError> {
    if n < 4 || !n.is_power_of_two() {
        return Err(MathError::InvalidRadix2GridSize { size: n, min: 4 });
    }
    let required = n
        .checked_mul(9)
        .ok_or(MathError::InvalidRadix2GridSize { size: n, min: 4 })?;
    if workspace_len < required {
        return Err(MathError::WorkspaceTooSmall {
            required,
            available: workspace_len,
        });
    }
    Ok(())
}

fn check_radix2_2d(
    field_len: usize,
    width: usize,
    height: usize,
    workspace_len: usize,
) -> Result<(), MathError> {
    if width < 4 || !width.is_power_of_two() {
        return Err(MathError::InvalidRadix2GridSize {
            size: width,
            min: 4,
        });
    }
    if height < 4 || !height.is_power_of_two() {
        return Err(MathError::InvalidRadix2GridSize {
            size: height,
            min: 4,
        });
    }
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
    let required = total
        .checked_mul(10)
        .and_then(|t| height.checked_mul(2).and_then(|h2| t.checked_add(h2)))
        .ok_or(MathError::WorkspaceTooSmall {
            required: usize::MAX,
            available: workspace_len,
        })?;
    if workspace_len < required {
        return Err(MathError::WorkspaceTooSmall {
            required,
            available: workspace_len,
        });
    }
    Ok(())
}

impl SemiImplicitFourierStepper {
    /// Checked companion to [`Self::step_cahn_hilliard_1d`] validating radix-2 size, parameters, and workspace capacity.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_cahn_hilliard_1d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        mobility: f64,
        kappa: f64,
        mobility_temp: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError> {
        check_radix2_1d(phi.len(), workspace.len())?;
        check_non_negative(mobility)?;
        check_non_negative(kappa)?;
        check_non_negative(dt)?;
        self.step_cahn_hilliard_1d(phi, mobility, kappa, mobility_temp, dt, workspace, rng);
        debug_assert!(phi.len() >= 4);
        Ok(())
    }

    /// Checked companion to [`Self::step_cahn_hilliard_2d`] validating radix-2 grid, parameters, and workspace capacity.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_cahn_hilliard_2d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        mobility_temp: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError> {
        check_radix2_2d(phi.len(), width, height, workspace.len())?;
        check_non_negative(mobility)?;
        check_non_negative(kappa)?;
        check_non_negative(dt)?;
        self.step_cahn_hilliard_2d(
            phi,
            width,
            height,
            mobility,
            kappa,
            mobility_temp,
            dt,
            workspace,
            rng,
        );
        debug_assert!(phi.len() >= width * height);
        Ok(())
    }

    /// Checked companion to [`Self::step_kpz_1d`] validating radix-2 size, `nu >= 0`, `dt >= 0`, and workspace capacity.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_kpz_1d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError> {
        check_radix2_1d(h.len(), workspace.len())?;
        check_non_negative(nu)?;
        check_non_negative(dt)?;
        self.step_kpz_1d(h, nu, lambda, noise_intensity, dt, workspace, rng);
        debug_assert!(h.len() >= 4);
        Ok(())
    }

    /// Checked companion to [`Self::step_kpz_2d`] validating radix-2 grid, `nu >= 0`, `dt >= 0`, and workspace capacity.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_kpz_2d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        width: usize,
        height: usize,
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError> {
        check_radix2_2d(h.len(), width, height, workspace.len())?;
        check_non_negative(nu)?;
        check_non_negative(dt)?;
        self.step_kpz_2d(
            h,
            width,
            height,
            nu,
            lambda,
            noise_intensity,
            dt,
            workspace,
            rng,
        );
        debug_assert!(h.len() >= width * height);
        Ok(())
    }

    /// Checked companion to [`Self::step_edwards_wilkinson_1d`].
    #[verified_engine::verified]
    pub fn try_step_edwards_wilkinson_1d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        nu: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError> {
        self.try_step_kpz_1d(h, nu, 0.0, noise_intensity, dt, workspace, rng)?;
        debug_assert!(h.len() >= 4);
        Ok(())
    }

    /// Checked companion to [`Self::step_edwards_wilkinson_2d`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_edwards_wilkinson_2d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        width: usize,
        height: usize,
        nu: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), MathError> {
        self.try_step_kpz_2d(
            h,
            width,
            height,
            nu,
            0.0,
            noise_intensity,
            dt,
            workspace,
            rng,
        )?;
        debug_assert!(h.len() >= width * height);
        Ok(())
    }
}
