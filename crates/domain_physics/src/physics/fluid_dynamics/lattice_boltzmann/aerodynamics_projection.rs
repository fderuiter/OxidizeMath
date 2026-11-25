//! Row-major velocity and obstacle projection/synchronization helpers for D2Q9 Lattice Boltzmann grids.

use super::super::model::{CollisionModel, D2Q9, Lattice2D, LatticeBoltzmannD2Q9};
use crate::error::FluidError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, flatten_2d_index};

/// Fallibly projects column-major [`LatticeBoltzmannD2Q9`] velocities `(ux, uy)` into
/// pre-allocated row-major slices (`flatten_2d_index(x, y, width)`) with zero heap allocations.
///
/// Solid obstacle cells write `(0.0, 0.0)` and non-finite entries are sanitized to `0.0`.
#[verified_engine::verified]
pub fn try_project_velocity_row_major_into<C: CollisionModel<9, D2Q9>>(
    solver: &LatticeBoltzmannD2Q9<C>,
    out_ux: &mut [f64],
    out_uy: &mut [f64],
) -> Result<(), FluidError> {
    let width = solver.width();
    let height = solver.height();
    let total = width
        .checked_mul(height)
        .ok_or(FluidError::GridDimensionsOverflow { width, height })?;
    for actual in [out_ux.len(), out_uy.len()] {
        if actual != total {
            return Err(FluidError::Math(MathError::DimensionMismatch {
                expected: Dimension(total),
                actual: Dimension(actual),
            }));
        }
    }
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if solver.state.obstacles[(y, x)] {
                out_ux[idx] = 0.0;
                out_uy[idx] = 0.0;
            } else {
                let vx = solver.state.ux[(y, x)];
                let vy = solver.state.uy[(y, x)];
                out_ux[idx] = if vx.is_finite() { vx } else { 0.0 };
                out_uy[idx] = if vy.is_finite() { vy } else { 0.0 };
            }
        }
    }
    debug_assert_eq!(out_ux.len(), total);
    Ok(())
}

/// Projects column-major [`LatticeBoltzmannD2Q9`] velocities `(ux, uy)` into
/// pre-allocated row-major slices (`flatten_2d_index(x, y, width)`); prefer [`try_project_velocity_row_major_into`].
#[verified_engine::verified]
pub fn project_velocity_row_major_into<C: CollisionModel<9, D2Q9>>(
    solver: &LatticeBoltzmannD2Q9<C>,
    out_ux: &mut [f64],
    out_uy: &mut [f64],
) {
    try_project_velocity_row_major_into(solver, out_ux, out_uy)
        .expect("Row-major velocity buffer lengths must equal width * height");
}

/// Fallibly projects the column-major [`LatticeBoltzmannD2Q9`] obstacle mask into a pre-allocated
/// row-major `bool` slice (`flatten_2d_index(x, y, width)`) with zero heap allocations.
#[verified_engine::verified]
pub fn try_project_obstacles_row_major_into<C: CollisionModel<9, D2Q9>>(
    solver: &LatticeBoltzmannD2Q9<C>,
    out_obstacles: &mut [bool],
) -> Result<(), FluidError> {
    let width = solver.width();
    let height = solver.height();
    let total = width
        .checked_mul(height)
        .ok_or(FluidError::GridDimensionsOverflow { width, height })?;
    if out_obstacles.len() != total {
        return Err(FluidError::Math(MathError::DimensionMismatch {
            expected: Dimension(total),
            actual: Dimension(out_obstacles.len()),
        }));
    }
    for y in 0..height {
        for x in 0..width {
            out_obstacles[flatten_2d_index(x, y, width)] = solver.state.obstacles[(y, x)];
        }
    }
    debug_assert_eq!(out_obstacles.len(), total);
    Ok(())
}

/// Projects the column-major [`LatticeBoltzmannD2Q9`] obstacle mask into a pre-allocated
/// row-major `bool` slice (`flatten_2d_index(x, y, width)`); prefer [`try_project_obstacles_row_major_into`].
#[verified_engine::verified]
pub fn project_obstacles_row_major_into<C: CollisionModel<9, D2Q9>>(
    solver: &LatticeBoltzmannD2Q9<C>,
    out_obstacles: &mut [bool],
) {
    try_project_obstacles_row_major_into(solver, out_obstacles)
        .expect("Row-major obstacle buffer length must equal width * height");
}

/// Fallibly synchronizes a row-major obstacle slice (`flatten_2d_index(x, y, width)`) into
/// [`LatticeBoltzmannD2Q9`] with zero heap allocations, restoring rest equilibrium
/// populations when previously solid cells transition back to fluid.
#[verified_engine::verified]
pub fn try_sync_obstacles_row_major<C: CollisionModel<9, D2Q9>>(
    solver: &mut LatticeBoltzmannD2Q9<C>,
    row_major_obstacles: &[bool],
) -> Result<(), FluidError> {
    let width = solver.width();
    let height = solver.height();
    let total = width
        .checked_mul(height)
        .ok_or(FluidError::GridDimensionsOverflow { width, height })?;
    if row_major_obstacles.len() != total {
        return Err(FluidError::Math(MathError::DimensionMismatch {
            expected: Dimension(total),
            actual: Dimension(row_major_obstacles.len()),
        }));
    }
    let rest_eq = D2Q9::equilibrium(1.0, 0.0, 0.0);
    for y in 0..height {
        for x in 0..width {
            let is_obs = row_major_obstacles[flatten_2d_index(x, y, width)];
            let was_obs = solver.state.obstacles[(y, x)];
            solver.state.obstacles[(y, x)] = is_obs;
            if is_obs {
                solver.state.ux[(y, x)] = 0.0;
                solver.state.uy[(y, x)] = 0.0;
            } else if was_obs {
                solver.state.rho[(y, x)] = 1.0;
                solver.state.ux[(y, x)] = 0.0;
                solver.state.uy[(y, x)] = 0.0;
                solver.state.f[(y, x)] = rest_eq;
                solver.state.f_new[(y, x)] = rest_eq;
            }
        }
    }
    debug_assert_eq!(row_major_obstacles.len(), total);
    Ok(())
}

/// Synchronizes a row-major obstacle slice (`flatten_2d_index(x, y, width)`) into
/// [`LatticeBoltzmannD2Q9`]; prefer [`try_sync_obstacles_row_major`].
#[verified_engine::verified]
pub fn sync_obstacles_row_major<C: CollisionModel<9, D2Q9>>(
    solver: &mut LatticeBoltzmannD2Q9<C>,
    row_major_obstacles: &[bool],
) {
    try_sync_obstacles_row_major(solver, row_major_obstacles)
        .expect("Row-major obstacle input length must equal width * height");
}
