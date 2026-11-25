// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::error::PdeError;
use crate::spde_noise::{
    SpdeNoiseKind, try_inject_additive_noise_1d, try_inject_additive_noise_2d,
    try_inject_conserved_flux_noise_1d, try_inject_conserved_flux_noise_2d,
};
use math_commons::math_kernel::types::{SolverWorkspace, flatten_2d_index};
use math_scalars::RealField;
use rand_core::RngCore;

#[inline]
#[verified_engine::verified]
pub(super) fn validate_noise_1d_params<T: RealField>(dx: T, dt: T) -> Result<(), PdeError> {
    if !(dx > T::ZERO && dx.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dx" });
    }
    if !(dt >= T::ZERO && dt.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dt" });
    }
    debug_assert!(dx > T::ZERO && dt >= T::ZERO);
    Ok(())
}

#[inline]
#[verified_engine::verified]
pub(super) fn validate_noise_2d_params<T: RealField>(
    field_len: usize,
    width: usize,
    height: usize,
    dx: T,
    dy: T,
    dt: T,
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
    if !(dx > T::ZERO && dx.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dx" });
    }
    if !(dy > T::ZERO && dy.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dy" });
    }
    if !(dt >= T::ZERO && dt.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: "dt" });
    }
    debug_assert!(field_len >= total && dx > T::ZERO && dy > T::ZERO && dt >= T::ZERO);
    Ok(total)
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub(super) fn accumulate_conserved_flux_2d<T: RealField>(
    field: &mut [T],
    fx: &[T],
    fy: &[T],
    width: usize,
    height: usize,
    scale_x: T,
    scale_y: T,
    bc: BoundaryCondition<T>,
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
                T::ZERO
            };
            let jy_d = fy[idx] * scale_y;
            let jy_u = if y > 0 {
                fy[flatten_2d_index(x, y - 1, width)] * scale_y
            } else if bc == BoundaryCondition::Periodic {
                fy[flatten_2d_index(x, height - 1, width)] * scale_y
            } else {
                T::ZERO
            };
            field[idx] += (jx_r - jx_l) + (jy_d - jy_u);
        }
    }
}

pub(super) fn zero_boundary_normal_fluxes_2d<T: RealField>(
    fx: &mut [T],
    fy: &mut [T],
    width: usize,
    height: usize,
) {
    for y in 0..height {
        fx[flatten_2d_index(width - 1, y, width)] = T::ZERO;
    }
    for x in 0..width {
        fy[flatten_2d_index(x, height - 1, width)] = T::ZERO;
    }
}

pub(crate) fn try_apply_noise_1d<T: RealField, R: RngCore + ?Sized>(
    u: &mut [T],
    dx: T,
    dt: T,
    bc: BoundaryCondition<T>,
    noise: SpdeNoiseKind<T>,
    workspace: &mut SolverWorkspace<'_, T>,
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
pub(crate) fn try_apply_noise_2d<T: RealField, R: RngCore + ?Sized>(
    u: &mut [T],
    width: usize,
    height: usize,
    dx: T,
    dy: T,
    dt: T,
    bc: BoundaryCondition<T>,
    noise: SpdeNoiseKind<T>,
    workspace: &mut SolverWorkspace<'_, T>,
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
