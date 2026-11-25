// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::error::PdeError;
use math_commons::math_kernel::types::{
    Scalar, SolverWorkspace, fill_standard_normal, flatten_2d_index, try_take_workspace_slice,
};
use rand_core::RngCore;

/// Spatio-temporal stochastic forcing structure for continuum SPDE steppers.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpdeNoiseKind {
    /// Deterministic evolution without stochastic forcing.
    None,
    /// Non-conserved additive white noise $\sqrt{2 D_{\text{noise}} \Delta t / \Delta V}\,\xi_i$
    /// (Model A / Allen-Cahn / Edwards-Wilkinson / KPZ).
    Additive {
        /// Noise variance intensity $D_{\text{noise}} \ge 0$.
        intensity: f64,
    },
    /// Conserved divergence-of-flux noise $\nabla \cdot \left(\sqrt{2 M k_B T \Delta t / \Delta V}\,\vec{\xi}\right)$
    /// discretized on staggered cell faces so total mass $\sum_i u_i$ is conserved to machine precision
    /// (Model B / Cahn-Hilliard).
    ConservedFlux {
        /// Fluctuation-dissipation product $M k_B T \ge 0$.
        mobility_temp: f64,
    },
}

#[inline]
#[verified_engine::verified]
fn validate_noise_1d_params(dx: f64, dt: f64) -> Result<(), PdeError> {
    if !(dx > 0.0 && dx.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dx" });
    }
    if !(dt >= 0.0 && dt.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dt" });
    }
    debug_assert!(dx > 0.0 && dt >= 0.0);
    Ok(())
}

#[inline]
#[verified_engine::verified]
fn validate_noise_2d_params(
    field_len: usize,
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    dt: f64,
) -> Result<usize, PdeError> {
    let total = width
        .checked_mul(height)
        .ok_or(PdeError::InvalidGridDimension {
            axis: "grid",
            size: usize::MAX,
        })?;
    if field_len < total {
        return Err(PdeError::BufferTooSmall {
            buffer: "field",
            required: total,
            actual: field_len,
        });
    }
    if !(dx > 0.0 && dx.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dx" });
    }
    if !(dy > 0.0 && dy.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dy" });
    }
    if !(dt >= 0.0 && dt.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dt" });
    }
    debug_assert!(field_len >= total && dx > 0.0 && dy > 0.0 && dt >= 0.0);
    Ok(total)
}

/// Fallibly injects non-conserved 1D additive Gaussian white noise into `field` using `workspace` and `rng`.
///
/// # Errors
/// Returns [`PdeError`] if `dx <= 0`, `dt < 0`, any parameter is non-finite, or `workspace` lacks `field.len()` scalars.
#[verified_engine::verified]
pub fn try_inject_additive_noise_1d<R: RngCore + ?Sized>(
    field: &mut [f64],
    dx: f64,
    dt: f64,
    intensity: f64,
    bc: BoundaryCondition,
    workspace: &mut impl SolverWorkspace,
    rng: &mut R,
) -> Result<(), PdeError> {
    let n = field.len();
    validate_noise_1d_params(dx, dt)?;
    if !intensity.is_finite() {
        return Err(PdeError::InvalidParameter {
            parameter: "intensity",
        });
    }
    if n == 0 || intensity <= 0.0 || dt <= 0.0 {
        return Ok(());
    }
    let scale = <f64 as Scalar>::sqrt(2.0 * intensity * dt / dx);
    let mut ws = workspace.reborrow();
    let z = try_take_workspace_slice::<f64>(&mut ws, n)?;
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
pub fn inject_additive_noise_1d<R: RngCore + ?Sized>(
    field: &mut [f64],
    dx: f64,
    dt: f64,
    intensity: f64,
    bc: BoundaryCondition,
    workspace: &mut impl SolverWorkspace,
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
pub fn try_inject_additive_noise_2d<R: RngCore + ?Sized>(
    field: &mut [f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    intensity: f64,
    bc: BoundaryCondition,
    workspace: &mut impl SolverWorkspace,
    rng: &mut R,
) -> Result<(), PdeError> {
    let total = validate_noise_2d_params(field.len(), width, height, dx, dy, dt)?;
    if !intensity.is_finite() {
        return Err(PdeError::InvalidParameter {
            parameter: "intensity",
        });
    }
    if total == 0 || intensity <= 0.0 || dt <= 0.0 {
        return Ok(());
    }
    let scale = <f64 as Scalar>::sqrt(2.0 * intensity * dt / (dx * dy));
    let mut ws = workspace.reborrow();
    let z = try_take_workspace_slice::<f64>(&mut ws, total)?;
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
pub fn inject_additive_noise_2d<R: RngCore + ?Sized>(
    field: &mut [f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    intensity: f64,
    bc: BoundaryCondition,
    workspace: &mut impl SolverWorkspace,
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
pub fn try_inject_conserved_flux_noise_1d<R: RngCore + ?Sized>(
    field: &mut [f64],
    dx: f64,
    dt: f64,
    mobility_temp: f64,
    bc: BoundaryCondition,
    workspace: &mut impl SolverWorkspace,
    rng: &mut R,
) -> Result<(), PdeError> {
    let n = field.len();
    validate_noise_1d_params(dx, dt)?;
    if !mobility_temp.is_finite() {
        return Err(PdeError::InvalidParameter {
            parameter: "mobility_temp",
        });
    }
    if n < 2 || mobility_temp <= 0.0 || dt <= 0.0 {
        return Ok(());
    }
    let flux_scale = <f64 as Scalar>::sqrt(2.0 * mobility_temp * dt / dx) / dx;
    let mut ws = workspace.reborrow();
    let faces = try_take_workspace_slice(&mut ws, n)?;
    fill_standard_normal(faces, rng);
    for f in faces.iter_mut() {
        *f *= flux_scale;
    }
    if bc != BoundaryCondition::Periodic {
        faces[n - 1] = 0.0;
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
            0.0
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
pub fn inject_conserved_flux_noise_1d<R: RngCore + ?Sized>(
    field: &mut [f64],
    dx: f64,
    dt: f64,
    mobility_temp: f64,
    bc: BoundaryCondition,
    workspace: &mut impl SolverWorkspace,
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
pub fn try_inject_conserved_flux_noise_2d<R: RngCore + ?Sized>(
    field: &mut [f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    mobility_temp: f64,
    bc: BoundaryCondition,
    workspace: &mut impl SolverWorkspace,
    rng: &mut R,
) -> Result<(), PdeError> {
    let total = validate_noise_2d_params(field.len(), width, height, dx, dy, dt)?;
    if !mobility_temp.is_finite() {
        return Err(PdeError::InvalidParameter {
            parameter: "mobility_temp",
        });
    }
    if width < 2 || height < 2 || mobility_temp <= 0.0 || dt <= 0.0 {
        return Ok(());
    }
    let amp = <f64 as Scalar>::sqrt(2.0 * mobility_temp * dt / (dx * dy));
    let scale_x = amp / dx;
    let scale_y = amp / dy;
    let mut ws = workspace.reborrow();
    let fx = try_take_workspace_slice(&mut ws, total)?;
    let fy = try_take_workspace_slice(&mut ws, total)?;
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
pub fn inject_conserved_flux_noise_2d<R: RngCore + ?Sized>(
    field: &mut [f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    mobility_temp: f64,
    bc: BoundaryCondition,
    workspace: &mut impl SolverWorkspace,
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

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn accumulate_conserved_flux_2d(
    field: &mut [f64],
    fx: &[f64],
    fy: &[f64],
    width: usize,
    height: usize,
    scale_x: f64,
    scale_y: f64,
    bc: BoundaryCondition,
) {
    debug_assert!(field.len() >= width * height);
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if let BoundaryCondition::Dirichlet(val) = bc
                && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
            {
                field[idx] = val;
                continue;
            }
            let jx_r = fx[idx] * scale_x;
            let jx_l = if x > 0 {
                fx[flatten_2d_index(x - 1, y, width)] * scale_x
            } else if bc == BoundaryCondition::Periodic {
                fx[flatten_2d_index(width - 1, y, width)] * scale_x
            } else {
                0.0
            };
            let jy_d = fy[idx] * scale_y;
            let jy_u = if y > 0 {
                fy[flatten_2d_index(x, y - 1, width)] * scale_y
            } else if bc == BoundaryCondition::Periodic {
                fy[flatten_2d_index(x, height - 1, width)] * scale_y
            } else {
                0.0
            };
            field[idx] += (jx_r - jx_l) + (jy_d - jy_u);
        }
    }
}

fn zero_boundary_normal_fluxes_2d(fx: &mut [f64], fy: &mut [f64], width: usize, height: usize) {
    for y in 0..height {
        fx[flatten_2d_index(width - 1, y, width)] = 0.0;
    }
    for x in 0..width {
        fy[flatten_2d_index(x, height - 1, width)] = 0.0;
    }
}

pub(crate) fn try_apply_noise_1d<R: RngCore + ?Sized>(
    u: &mut [f64],
    dx: f64,
    dt: f64,
    bc: BoundaryCondition,
    noise: SpdeNoiseKind,
    workspace: &mut impl SolverWorkspace,
    rng: &mut R,
) -> Result<(), PdeError> {
    match noise {
        SpdeNoiseKind::None => Ok(()),
        SpdeNoiseKind::Additive { intensity } => {
            try_inject_additive_noise_1d(u, dx, dt, intensity, bc, workspace, rng)
        }
        SpdeNoiseKind::ConservedFlux { mobility_temp } => {
            try_inject_conserved_flux_noise_1d(u, dx, dt, mobility_temp, bc, workspace, rng)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_apply_noise_2d<R: RngCore + ?Sized>(
    u: &mut [f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    bc: BoundaryCondition,
    noise: SpdeNoiseKind,
    workspace: &mut impl SolverWorkspace,
    rng: &mut R,
) -> Result<(), PdeError> {
    match noise {
        SpdeNoiseKind::None => Ok(()),
        SpdeNoiseKind::Additive { intensity } => try_inject_additive_noise_2d(
            u, width, height, dx, dy, dt, intensity, bc, workspace, rng,
        ),
        SpdeNoiseKind::ConservedFlux { mobility_temp } => try_inject_conserved_flux_noise_2d(
            u,
            width,
            height,
            dx,
            dy,
            dt,
            mobility_temp,
            bc,
            workspace,
            rng,
        ),
    }
}
