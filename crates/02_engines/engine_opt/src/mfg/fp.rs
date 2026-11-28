// [cite:engine_opt]
//! Conservative donor-cell Fokker-Planck sweep and Mann fixed-point blending for 1D MFG.

use super::{CFL_SAFETY, MAX_SUBSTEPS, MfgConfig1D, MfgHamiltonian, normalize_density_slice};
use crate::error::OptError;
use engine_pde::StencilOperators;
use math_scalars::RealField;

#[verified_engine::verified]
pub(super) fn compute_substeps(dt: f64, rate: f64) -> Result<usize, OptError> {
    if !rate.is_finite() {
        return Err(OptError::NumericalInstability);
    }
    let raw = <f64 as RealField>::ceil(dt * rate / CFL_SAFETY);
    if !raw.is_finite() || raw > (MAX_SUBSTEPS as f64) {
        return Err(OptError::NumericalInstability);
    }
    Ok((raw as usize).max(1))
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub(super) fn forward_fp_sweep<H: MfgHamiltonian>(
    cfg: &MfgConfig1D,
    stencil: &StencilOperators,
    hamiltonian: &H,
    u_grid: &[f64],
    m_grid: &[f64],
    m_cand: &mut [f64],
    work_a: &mut [f64],
    work_b: &mut [f64],
    v_faces: &mut [f64],
) -> Result<usize, OptError> {
    let nx = cfg.nx;
    m_cand[..nx].copy_from_slice(&m_grid[..nx]);
    let mut max_sub = 1usize;

    for n in 0..cfg.nt {
        let u_col = &u_grid[n * nx..(n + 1) * nx];
        let v_max = fill_interface_velocities(u_col, stencil.inv_dx, hamiltonian, v_faces)?;
        let rate = 2.0 * cfg.viscosity * stencil.inv_dx_sq + 2.0 * v_max * stencil.inv_dx;
        let n_sub = compute_substeps(cfg.dt, rate)?;
        max_sub = max_sub.max(n_sub);
        let dt_sub = cfg.dt / (n_sub as f64);

        work_a.copy_from_slice(&m_cand[n * nx..(n + 1) * nx]);
        for _ in 0..n_sub {
            for i in 0..nx {
                let prev = if i == 0 { work_a[0] } else { work_a[i - 1] };
                let curr = work_a[i];
                let next = if i + 1 == nx {
                    work_a[nx - 1]
                } else {
                    work_a[i + 1]
                };
                let adv =
                    stencil.conservative_upwind_flux(v_faces[i], v_faces[i + 1], prev, curr, next);
                let diff = stencil.central_diff_2nd(prev, curr, next);
                let updated = curr + dt_sub * (cfg.viscosity * diff - adv);
                if !updated.is_finite() {
                    return Err(OptError::NumericalInstability);
                }
                work_b[i] = updated.max(0.0);
            }
            work_a.copy_from_slice(work_b);
        }
        normalize_density_slice(work_a, cfg.dx)?;
        m_cand[(n + 1) * nx..(n + 2) * nx].copy_from_slice(work_a);
    }
    Ok(max_sub)
}

#[verified_engine::verified]
fn fill_interface_velocities<H: MfgHamiltonian>(
    u_col: &[f64],
    inv_dx: f64,
    hamiltonian: &H,
    v_faces: &mut [f64],
) -> Result<f64, OptError> {
    let nx = u_col.len();
    v_faces[0] = 0.0;
    v_faces[nx] = 0.0;
    let mut v_max = 0.0_f64;
    for i in 0..(nx - 1) {
        let v = -hamiltonian.deriv((u_col[i + 1] - u_col[i]) * inv_dx);
        if !v.is_finite() {
            return Err(OptError::NumericalInstability);
        }
        v_faces[i + 1] = v;
        v_max = v_max.max(v.abs());
    }
    Ok(v_max)
}

#[verified_engine::verified]
pub(super) fn mann_blend_and_normalize(
    cfg: &MfgConfig1D,
    m_grid: &mut [f64],
    m_cand: &[f64],
) -> Result<f64, OptError> {
    let alpha = cfg.mann_alpha;
    let keep = 1.0 - alpha;
    let mut max_l1 = 0.0_f64;
    for n in 0..=cfg.nt {
        let col = &mut m_grid[n * cfg.nx..(n + 1) * cfg.nx];
        let cand = &m_cand[n * cfg.nx..(n + 1) * cfg.nx];
        let mut l1 = 0.0;
        for i in 0..cfg.nx {
            l1 += (cand[i] - col[i]).abs() * cfg.dx;
            col[i] = keep * col[i] + alpha * cand[i];
        }
        max_l1 = max_l1.max(l1);
        normalize_density_slice(col, cfg.dx)?;
    }
    Ok(max_l1)
}
