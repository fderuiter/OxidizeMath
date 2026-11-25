// [cite:engine_pde]
mod helpers;

use crate::boundary::BoundaryCondition;
use crate::error::PdeError;
use helpers::{
    accumulate_conserved_flux_2d, validate_noise_1d_params, validate_noise_2d_params,
    zero_boundary_normal_fluxes_2d,
};
pub(crate) use helpers::{try_apply_noise_1d, try_apply_noise_2d};
use math_commons::math_kernel::types::{SolverWorkspace, fill_standard_normal, flatten_2d_index};
use math_scalars::RealField;
use rand_core::RngCore;

/// Spatio-temporal stochastic forcing structure for continuum SPDE steppers.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpdeNoiseKind<T = f64> {
    /// Deterministic evolution without stochastic forcing.
    None,
    /// Non-conserved additive white noise $\sqrt{2 D_{\text{noise}} \Delta t / \Delta V}\,\xi_i$
    /// (Model A / Allen-Cahn / Edwards-Wilkinson / KPZ).
    Additive {
        /// Noise variance intensity $D_{\text{noise}} \ge 0$.
        intensity: T,
    },
    /// Conserved divergence-of-flux noise $\nabla \cdot \left(\sqrt{2 M k_B T \Delta t / \Delta V}\,\vec{\xi}\right)$
    /// discretized on staggered cell faces so total mass $\sum_i u_i$ is conserved to machine precision
    /// (Model B / Cahn-Hilliard).
    ConservedFlux {
        /// Fluctuation-dissipation product $M k_B T \ge 0$.
        mobility_temp: T,
    },
}

/// Fallibly injects non-conserved 1D additive Gaussian white noise into `field` using `workspace` and `rng`.
///
/// # Errors
/// Returns [`PdeError`] if `dx <= 0`, `dt < 0`, any parameter is non-finite, or `workspace` lacks `field.len()` scalars.
#[verified_engine::verified]
pub fn try_inject_additive_noise_1d<T: RealField, R: RngCore + ?Sized>(
    field: &mut [T],
    dx: T,
    dt: T,
    intensity: T,
    bc: BoundaryCondition<T>,
    workspace: &mut SolverWorkspace<'_, T>,
    rng: &mut R,
) -> Result<(), PdeError> {
    let n = field.len();
    validate_noise_1d_params(dx, dt)?;
    if !intensity.is_finite() {
        return Err(PdeError::InvalidParameter {
            parameter: "intensity",
        });
    }
    if n == 0 || intensity <= T::ZERO || dt <= T::ZERO {
        return Ok(());
    }
    let scale = (T::TWO * intensity * dt / dx).sqrt();
    let mut ws = workspace.reborrow();
    let z = ws.try_take_slice(n)?;
    fill_standard_normal(z, rng);
    for i in 0..n {
        if let BoundaryCondition::Dirichlet(val) = bc
            && (i == 0 || i + 1 == n)
        {
            field[i] = val;
            continue;
        }
        field[i] += scale * z[i];
    }
    debug_assert_eq!(z.len(), n);
    Ok(())
}

/// Injects non-conserved 1D additive Gaussian white noise into `field` using `workspace` and `rng`.
///
/// # Panics
/// Panics if `dx <= 0`, `dt < 0`, or `workspace` is too small; prefer [`try_inject_additive_noise_1d`].
#[verified_engine::verified]
pub fn inject_additive_noise_1d<T: RealField, R: RngCore + ?Sized>(
    field: &mut [T],
    dx: T,
    dt: T,
    intensity: T,
    bc: BoundaryCondition<T>,
    workspace: &mut SolverWorkspace<'_, T>,
    rng: &mut R,
) {
    try_inject_additive_noise_1d(field, dx, dt, intensity, bc, workspace, rng)
        .expect("inject_additive_noise_1d preconditions violated");
}

/// Fallibly injects non-conserved 2D additive Gaussian white noise into `field` (`width x height`).
///
/// # Errors
/// Returns [`PdeError`] if `field` or `workspace` is undersized or `dx`/`dy`/`dt` is invalid.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn try_inject_additive_noise_2d<T: RealField, R: RngCore + ?Sized>(
    field: &mut [T],
    width: usize,
    height: usize,
    dx: T,
    dy: T,
    dt: T,
    intensity: T,
    bc: BoundaryCondition<T>,
    workspace: &mut SolverWorkspace<'_, T>,
    rng: &mut R,
) -> Result<(), PdeError> {
    let total = validate_noise_2d_params(field.len(), width, height, dx, dy, dt)?;
    if !intensity.is_finite() {
        return Err(PdeError::InvalidParameter {
            parameter: "intensity",
        });
    }
    if total == 0 || intensity <= T::ZERO || dt <= T::ZERO {
        return Ok(());
    }
    let scale = (T::TWO * intensity * dt / (dx * dy)).sqrt();
    let mut ws = workspace.reborrow();
    let z = ws.try_take_slice(total)?;
    fill_standard_normal(z, rng);
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if let BoundaryCondition::Dirichlet(val) = bc
                && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
            {
                field[idx] = val;
                continue;
            }
            field[idx] += scale * z[idx];
        }
    }
    debug_assert_eq!(z.len(), total);
    Ok(())
}

/// Injects non-conserved 2D additive Gaussian white noise into `field` (`width x height`).
///
/// # Panics
/// Panics if `field` or `workspace` is undersized or spacings/step are invalid; prefer [`try_inject_additive_noise_2d`].
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn inject_additive_noise_2d<T: RealField, R: RngCore + ?Sized>(
    field: &mut [T],
    width: usize,
    height: usize,
    dx: T,
    dy: T,
    dt: T,
    intensity: T,
    bc: BoundaryCondition<T>,
    workspace: &mut SolverWorkspace<'_, T>,
    rng: &mut R,
) {
    try_inject_additive_noise_2d(
        field, width, height, dx, dy, dt, intensity, bc, workspace, rng,
    )
    .expect("inject_additive_noise_2d preconditions violated");
}

/// Fallibly injects 1D staggered-face conserved-flux noise $\Delta t \, \partial_x \eta$ into `field`.
///
/// # Errors
/// Returns [`PdeError`] if `dx <= 0`, `dt < 0`, any parameter is non-finite, or `workspace` lacks `field.len()` scalars.
#[verified_engine::verified]
pub fn try_inject_conserved_flux_noise_1d<T: RealField, R: RngCore + ?Sized>(
    field: &mut [T],
    dx: T,
    dt: T,
    mobility_temp: T,
    bc: BoundaryCondition<T>,
    workspace: &mut SolverWorkspace<'_, T>,
    rng: &mut R,
) -> Result<(), PdeError> {
    let n = field.len();
    validate_noise_1d_params(dx, dt)?;
    if !mobility_temp.is_finite() {
        return Err(PdeError::InvalidParameter {
            parameter: "mobility_temp",
        });
    }
    if n < 2 || mobility_temp <= T::ZERO || dt <= T::ZERO {
        return Ok(());
    }
    let flux_scale = (T::TWO * mobility_temp * dt / dx).sqrt() / dx;
    let mut ws = workspace.reborrow();
    let faces = ws.try_take_slice(n)?;
    fill_standard_normal(faces, rng);
    for f in faces.iter_mut() {
        *f *= flux_scale;
    }
    if bc != BoundaryCondition::Periodic {
        faces[n - 1] = T::ZERO;
    }
    for i in 0..n {
        if let BoundaryCondition::Dirichlet(val) = bc
            && (i == 0 || i + 1 == n)
        {
            field[i] = val;
            continue;
        }
        let j_right = faces[i];
        let j_left = if i > 0 {
            faces[i - 1]
        } else if bc == BoundaryCondition::Periodic {
            faces[n - 1]
        } else {
            T::ZERO
        };
        field[i] += j_right - j_left;
    }
    debug_assert_eq!(faces.len(), n);
    Ok(())
}

/// Injects 1D staggered-face conserved-flux noise $\Delta t \, \partial_x \eta$ into `field`,
/// strictly preserving $\sum_i u_i$ under Periodic and Neumann boundaries.
///
/// # Panics
/// Panics if `dx <= 0`, `dt < 0`, or `workspace` is too small; prefer [`try_inject_conserved_flux_noise_1d`].
#[verified_engine::verified]
pub fn inject_conserved_flux_noise_1d<T: RealField, R: RngCore + ?Sized>(
    field: &mut [T],
    dx: T,
    dt: T,
    mobility_temp: T,
    bc: BoundaryCondition<T>,
    workspace: &mut SolverWorkspace<'_, T>,
    rng: &mut R,
) {
    try_inject_conserved_flux_noise_1d(field, dx, dt, mobility_temp, bc, workspace, rng)
        .expect("inject_conserved_flux_noise_1d preconditions violated");
}

/// Fallibly injects 2D staggered-face conserved-flux noise $\Delta t \, \nabla \cdot \vec{\eta}$ into `field`.
///
/// # Errors
/// Returns [`PdeError`] if `field` or `workspace` is undersized or `dx`/`dy`/`dt` is invalid.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn try_inject_conserved_flux_noise_2d<T: RealField, R: RngCore + ?Sized>(
    field: &mut [T],
    width: usize,
    height: usize,
    dx: T,
    dy: T,
    dt: T,
    mobility_temp: T,
    bc: BoundaryCondition<T>,
    workspace: &mut SolverWorkspace<'_, T>,
    rng: &mut R,
) -> Result<(), PdeError> {
    let total = validate_noise_2d_params(field.len(), width, height, dx, dy, dt)?;
    if !mobility_temp.is_finite() {
        return Err(PdeError::InvalidParameter {
            parameter: "mobility_temp",
        });
    }
    if width < 2 || height < 2 || mobility_temp <= T::ZERO || dt <= T::ZERO {
        return Ok(());
    }
    let amp = (T::TWO * mobility_temp * dt / (dx * dy)).sqrt();
    let scale_x = amp / dx;
    let scale_y = amp / dy;
    let mut ws = workspace.reborrow();
    let fx = ws.try_take_slice(total)?;
    let fy = ws.try_take_slice(total)?;
    fill_standard_normal(fx, rng);
    fill_standard_normal(fy, rng);

    if bc != BoundaryCondition::Periodic {
        zero_boundary_normal_fluxes_2d(fx, fy, width, height);
    }

    accumulate_conserved_flux_2d(field, fx, fy, width, height, scale_x, scale_y, bc);
    debug_assert_eq!(fx.len(), total);
    Ok(())
}

/// Injects 2D staggered-face conserved-flux noise $\Delta t \, \nabla \cdot \vec{\eta}$ into `field`,
/// strictly preserving $\sum_{x, y} u_{x, y}$ under Periodic and Neumann boundaries.
///
/// # Panics
/// Panics if `field` or `workspace` is undersized or spacings/step are invalid;
/// prefer [`try_inject_conserved_flux_noise_2d`].
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn inject_conserved_flux_noise_2d<T: RealField, R: RngCore + ?Sized>(
    field: &mut [T],
    width: usize,
    height: usize,
    dx: T,
    dy: T,
    dt: T,
    mobility_temp: T,
    bc: BoundaryCondition<T>,
    workspace: &mut SolverWorkspace<'_, T>,
    rng: &mut R,
) {
    try_inject_conserved_flux_noise_2d(
        field,
        width,
        height,
        dx,
        dy,
        dt,
        mobility_temp,
        bc,
        workspace,
        rng,
    )
    .expect("inject_conserved_flux_noise_2d preconditions violated");
}
