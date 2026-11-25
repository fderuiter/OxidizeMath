//! Wall-aware 2D conservative upwind advection fused with 5-point isotropic diffusion
//! and reaction kinetics, enforcing Patankar positivity and zero wall flux.

use super::{FiniteDifference2D, MaskedFiniteDifference2D, ReactionCellRole, SpatialDiffusion};
use crate::biology::diffusion::fd2d::geometry::GeometryStrategy;
use crate::biology::morphogenesis::reaction::ReactionKinetics;
use math_commons::math_kernel::types::flatten_2d_index;
use pure_math::pure_math::analysis::pde::advection_2d::upwind_advection_cell_2d;

/// Extension of [`SpatialDiffusion`] supporting 2D conservative upwind advection
/// driven by row-major horizontal (`ux`) and vertical (`uy`) velocity slices.
pub trait AdvectedSpatialDiffusion<const N: usize>: SpatialDiffusion<N> {
    /// Advances one fused 2D advection-diffusion-reaction step with zero heap allocations.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    fn step_advected_fused<K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    );
}

#[inline(always)]
fn safe_face_vel(vel: &[f64], idx_a: usize, idx_b: usize) -> f64 {
    let va = vel[idx_a];
    let vb = vel[idx_b];
    let a = if va.is_finite() { va } else { 0.0 };
    let b = if vb.is_finite() { vb } else { 0.0 };
    0.5 * (a + b)
}

#[inline(always)]
fn masked_face_vel(roles: &[u8], vel: &[f64], idx_a: usize, idx_b: usize, in_bounds: bool) -> f64 {
    if !in_bounds
        || roles[idx_a] == ReactionCellRole::ZeroFluxWall as u8
        || roles[idx_b] == ReactionCellRole::ZeroFluxWall as u8
    {
        0.0
    } else {
        safe_face_vel(vel, idx_a, idx_b)
    }
}

#[inline(always)]
fn wall_reflected(roles: &[u8], slice: &[f64], center_val: f64, nb_idx: usize) -> f64 {
    if roles[nb_idx] == ReactionCellRole::ZeroFluxWall as u8 {
        center_val
    } else {
        slice[nb_idx]
    }
}

#[inline]
fn validate_advection_slices<const N: usize>(
    total: usize,
    state: &[&[f64]; N],
    next_state: &[&mut [f64]; N],
    ux: &[f64],
    uy: &[f64],
) -> Result<(), crate::biology::reaction_diffusion::ReactionDiffusionError> {
    use math_commons::error::MathError;
    use math_commons::math_kernel::types::Dimension;
    for s in 0..N {
        if state[s].len() < total {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(total),
                actual: Dimension(state[s].len()),
            }
            .into());
        }
        if next_state[s].len() < total {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(total),
                actual: Dimension(next_state[s].len()),
            }
            .into());
        }
    }
    if ux.len() < total {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(total),
            actual: Dimension(ux.len()),
        }
        .into());
    }
    if uy.len() < total {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(total),
            actual: Dimension(uy.len()),
        }
        .into());
    }
    Ok(())
}

impl FiniteDifference2D {
    /// Fallibly evaluates one fused 2D conservative upwind advection + 5-point Neumann diffusion +
    /// reaction kinetics step over row-major `ux` and `uy` slices with zero heap allocations.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_advected_fused<const N: usize, K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) -> Result<(), crate::biology::reaction_diffusion::ReactionDiffusionError> {
        let (w, h, total) = (
            *self.geometry.width,
            *self.geometry.height,
            self.geometry.size(),
        );
        if N == 0 || total == 0 {
            return Ok(());
        }
        validate_advection_slices(total, &state, &next_state, ux, uy)?;
        let stepper = SpatialDiffusion::<N>::stepper(self);
        let ops = &stepper.ops;
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                let (idx_l, idx_r) = (
                    if x > 0 { idx - 1 } else { idx },
                    if x + 1 < w { idx + 1 } else { idx },
                );
                let (idx_u, idx_d) = (
                    if y > 0 { idx - w } else { idx },
                    if y + 1 < h { idx + w } else { idx },
                );
                let face_vel = [
                    if x > 0 {
                        safe_face_vel(ux, idx_l, idx)
                    } else {
                        0.0
                    },
                    if x + 1 < w {
                        safe_face_vel(ux, idx, idx_r)
                    } else {
                        0.0
                    },
                    if y > 0 {
                        safe_face_vel(uy, idx_u, idx)
                    } else {
                        0.0
                    },
                    if y + 1 < h {
                        safe_face_vel(uy, idx, idx_d)
                    } else {
                        0.0
                    },
                ];
                let curr: [f64; N] = core::array::from_fn(|s| state[s][idx]);
                let rates = kinetics.reaction_at(idx, curr);
                for s in 0..N {
                    let cs = state[s];
                    let (cc, cl, cr, cu, cd) =
                        (curr[s], cs[idx_l], cs[idx_r], cs[idx_u], cs[idx_d]);
                    let d2u = ops.central_diff_2nd_2d(cc, cl, cr, cu, cd);
                    let adv = upwind_advection_cell_2d(
                        cc,
                        [cl, cr, cu, cd],
                        face_vel,
                        ops.inv_dx,
                        ops.inv_dy,
                    );
                    next_state[s][idx] = cc + dt * (coeffs[s] * d2u + adv + rates[s]);
                }
            }
        }
        Ok(())
    }

    /// Evaluates one fused 2D conservative upwind advection + 5-point Neumann diffusion +
    /// reaction kinetics step; prefer [`Self::try_step_advected_fused`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_advected_fused<const N: usize, K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        self.try_step_advected_fused(state, next_state, ux, uy, dt, coeffs, kinetics)
            .expect("Velocity slice length mismatch in FiniteDifference2D::step_advected_fused");
    }
}

#[allow(clippy::too_many_arguments)]
fn step_masked_free_cell<const N: usize, K: ReactionKinetics<N>>(
    roles: &[u8],
    state: &[&[f64]; N],
    next_state: &mut [&mut [f64]; N],
    idx: usize,
    nb_indices: [usize; 4],
    face_vel: [f64; 4],
    spacings: [f64; 4],
    dt: f64,
    coeffs: [f64; N],
    kinetics: &K,
) {
    let [idx_l, idx_r, idx_u, idx_d] = nb_indices;
    let [u_west, u_east, v_north, v_south] = face_vel;
    let [inv_dx, inv_dy, inv_dx_sq, inv_dy_sq] = spacings;
    let adv_out_coeff = (-u_west).max(0.0) * inv_dx
        + u_east.max(0.0) * inv_dx
        + (-v_north).max(0.0) * inv_dy
        + v_south.max(0.0) * inv_dy;

    let mut curr = [0.0; N];
    for s in 0..N {
        curr[s] = state[s][idx];
    }
    let rates = kinetics.reaction_at(idx, curr);

    for s in 0..N {
        let u_c = curr[s];
        let u_l = wall_reflected(roles, state[s], u_c, idx_l);
        let u_r = wall_reflected(roles, state[s], u_c, idx_r);
        let u_u = wall_reflected(roles, state[s], u_c, idx_u);
        let u_d = wall_reflected(roles, state[s], u_c, idx_d);

        let adv_in = u_west.max(0.0) * u_l * inv_dx
            + (-u_east).max(0.0) * u_r * inv_dx
            + v_north.max(0.0) * u_u * inv_dy
            + (-v_south).max(0.0) * u_d * inv_dy;
        let d_s = coeffs[s].max(0.0);
        let diff_in = d_s * ((u_l + u_r) * inv_dx_sq + (u_u + u_d) * inv_dy_sq);
        let nb_flux = diff_in + adv_in;
        let alpha = 2.0 * dt * d_s * (inv_dx_sq + inv_dy_sq) + dt * adv_out_coeff;
        let rate = rates[s];
        next_state[s][idx] = if rate < 0.0 && u_c > 1e-12 {
            (u_c + dt * nb_flux) / (1.0 + alpha + dt * (-rate / u_c))
        } else {
            (u_c + dt * (nb_flux + rate)) / (1.0 + alpha)
        };
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_masked_role_cell<const N: usize, K: ReactionKinetics<N>>(
    mask: &super::CellMask2D,
    state: &[&[f64]; N],
    next_state: &mut [&mut [f64]; N],
    ux: &[f64],
    uy: &[f64],
    coords: (usize, usize, usize, usize),
    spacings: [f64; 4],
    dt: f64,
    coeffs: [f64; N],
    kinetics: &K,
) {
    let (x, y, w, h) = coords;
    let roles = mask.roles();
    let idx = flatten_2d_index(x, y, w);
    match ReactionCellRole::from_u8(roles[idx]) {
        ReactionCellRole::ZeroFluxWall | ReactionCellRole::Sink => {
            for next_s in next_state.iter_mut().take(N) {
                next_s[idx] = 0.0;
            }
        }
        ReactionCellRole::EmitterU => {
            next_state[0][idx] = mask.emitter_u_level;
            for s in 1..N {
                next_state[s][idx] = state[s][idx].clamp(0.0, 2.5);
            }
        }
        ReactionCellRole::EmitterV => {
            if N > 1 {
                next_state[0][idx] = state[0][idx].clamp(0.0, 2.5);
                next_state[1][idx] = mask.emitter_v_level;
                for s in 2..N {
                    next_state[s][idx] = state[s][idx].clamp(0.0, 2.5);
                }
            } else {
                next_state[0][idx] = mask.emitter_v_level;
            }
        }
        ReactionCellRole::Free => {
            let nb = [
                flatten_2d_index(x.saturating_sub(1), y, w),
                flatten_2d_index((x + 1).min(w - 1), y, w),
                flatten_2d_index(x, y.saturating_sub(1), w),
                flatten_2d_index(x, (y + 1).min(h - 1), w),
            ];
            let fvel = [
                masked_face_vel(roles, ux, nb[0], idx, x > 0),
                masked_face_vel(roles, ux, idx, nb[1], x + 1 < w),
                masked_face_vel(roles, uy, nb[2], idx, y > 0),
                masked_face_vel(roles, uy, idx, nb[3], y + 1 < h),
            ];
            step_masked_free_cell(
                roles, state, next_state, idx, nb, fvel, spacings, dt, coeffs, kinetics,
            );
        }
    }
}

impl MaskedFiniteDifference2D {
    /// Fallibly evaluates one wall-aware fused 2D donor-cell upwind advection + isotropic diffusion +
    /// reaction kinetics step with Patankar positivity preservation and zero heap allocations.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_advected_fused<const N: usize, K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        mut next_state: [&mut [f64]; N],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) -> Result<(), crate::biology::reaction_diffusion::ReactionDiffusionError> {
        if !self.mask.has_active_cells() {
            return self
                .base
                .try_step_advected_fused(state, next_state, ux, uy, dt, coeffs, kinetics);
        }
        let w = *self.base.geometry.width;
        let h = *self.base.geometry.height;
        let total = w.saturating_mul(h);
        if N == 0 || total == 0 {
            return Ok(());
        }
        validate_advection_slices(total, &state, &next_state, ux, uy)?;
        let inv_dx = 1.0 / *self.base.geometry.dx();
        let inv_dy = 1.0 / *self.base.geometry.dy();
        let spacings = [inv_dx, inv_dy, inv_dx * inv_dx, inv_dy * inv_dy];
        for y in 0..h {
            for x in 0..w {
                apply_masked_role_cell(
                    &self.mask,
                    &state,
                    &mut next_state,
                    ux,
                    uy,
                    (x, y, w, h),
                    spacings,
                    dt,
                    coeffs,
                    kinetics,
                );
            }
        }
        Ok(())
    }

    /// Evaluates one wall-aware fused 2D donor-cell upwind advection + isotropic diffusion +
    /// reaction kinetics step; prefer [`Self::try_step_advected_fused`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_advected_fused<const N: usize, K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        self.try_step_advected_fused(state, next_state, ux, uy, dt, coeffs, kinetics)
            .expect(
                "Velocity slice length mismatch in MaskedFiniteDifference2D::step_advected_fused",
            );
    }
}

impl<const N: usize> AdvectedSpatialDiffusion<N> for FiniteDifference2D {
    #[verified_engine::verified(
        allow_recursion = "delegates trait step_advected_fused to inherent FiniteDifference2D::step_advected_fused"
    )]
    fn step_advected_fused<K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        FiniteDifference2D::step_advected_fused(
            self, state, next_state, ux, uy, dt, coeffs, kinetics,
        );
    }
}

impl<const N: usize> AdvectedSpatialDiffusion<N> for MaskedFiniteDifference2D {
    #[verified_engine::verified(
        allow_recursion = "delegates trait step_advected_fused to inherent MaskedFiniteDifference2D::step_advected_fused"
    )]
    fn step_advected_fused<K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        MaskedFiniteDifference2D::step_advected_fused(
            self, state, next_state, ux, uy, dt, coeffs, kinetics,
        );
    }
}

// [cite:essay]
