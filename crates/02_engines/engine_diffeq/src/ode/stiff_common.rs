//! Shared zero-allocation finite-difference Jacobian and stage matrix helpers for stiff solvers.
// [cite:engine_diffeq]

use super::traits::{OdeState, OdeSystem};
use crate::error::DiffeqError;
use crate::workspace::{lu_factor_in_place, row_major_index};

/// Maximum state dimension supported by default stack buffers inside [`super::Rosenbrock23`] and [`super::Bdf2`].
pub const DEFAULT_STIFF_MAX_DIM: usize = 16;
/// Stack matrix capacity (`DEFAULT_STIFF_MAX_DIM * DEFAULT_STIFF_MAX_DIM`).
pub const DEFAULT_STIFF_MAT_CAP: usize = DEFAULT_STIFF_MAX_DIM * DEFAULT_STIFF_MAX_DIM;

const FD_REL_STEP: f64 = 1.490_116_119_384_765_6e-8; // sqrt(f64::EPSILON)

/// Bundled mutable scratch buffers for dense Jacobian/LU and stage vector arithmetic.
pub(super) struct StiffScratch<'a> {
    pub w_lu: &'a mut [f64],
    pub vec_area: &'a mut [f64],
    pub pivots: &'a mut [usize],
    pub dim: usize,
}

/// Bundled temporary state pair (`work_state`, `work_f`) for finite-difference evaluations.
pub(super) struct StatePair<'a, State> {
    pub state: &'a mut State,
    pub deriv: &'a mut State,
}

/// Assembles and LU-factorizes $W = I - c J$ in-place where $J = \frac{\partial f}{\partial y}(t, y)$
/// is approximated by forward finite differences.
#[verified_engine::verified]
pub(super) fn build_and_factor_stage_matrix<State: OdeState, S: OdeSystem<State> + ?Sized>(
    system: &S,
    t: f64,
    base_state: &State,
    base_f: &State,
    coeff_c: f64,
    work: &mut StatePair<'_, State>,
    scratch: &mut StiffScratch<'_>,
) -> Result<(), DiffeqError> {
    let n = scratch.dim;
    work.state.copy_from(base_state);

    for col in 0..n {
        let y_col = base_state.component(col);
        let sign = if y_col >= 0.0 { 1.0 } else { -1.0 };
        let delta = FD_REL_STEP * y_col.abs().max(1.0) * sign;
        work.state.set_component(col, y_col + delta);
        system.derivative_in_place(t, work.state, work.deriv);
        work.state.set_component(col, y_col);

        let inv_delta = 1.0 / delta;
        for row in 0..n {
            let df_dy = (work.deriv.component(row) - base_f.component(row)) * inv_delta;
            let diag = if row == col { 1.0 } else { 0.0 };
            scratch.w_lu[row_major_index(row, n, col)] = diag - coeff_c * df_dy;
        }
    }
    lu_factor_in_place(scratch.w_lu, scratch.pivots, n)
}

/// Computes the partial time derivative $\frac{\partial f}{\partial t}(t, y)$ via central finite differences.
#[verified_engine::verified]
pub(super) fn compute_time_partial<State: OdeState, S: OdeSystem<State> + ?Sized>(
    system: &S,
    t: f64,
    state: &State,
    h: f64,
    work: &mut StatePair<'_, State>,
    df_dt_out: &mut [f64],
) {
    let n = state.state_dim();
    let delta_t = 1e-6 * t.abs().max(h.abs()).max(1.0);
    system.derivative_in_place(t + delta_t, state, work.state);
    system.derivative_in_place(t - delta_t, state, work.deriv);
    let inv_two_dt = 0.5 / delta_t;
    for (i, slot) in df_dt_out.iter_mut().enumerate().take(n) {
        *slot = (work.state.component(i) - work.deriv.component(i)) * inv_two_dt;
    }
}
