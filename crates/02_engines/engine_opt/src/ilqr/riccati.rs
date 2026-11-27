// [cite:engine_opt]
//! Backward Riccati pass and `cholesky_solve` box-projected control gains for `engine_opt::ilqr`.

use super::StageCost;
use crate::error::OptError;
use engine_linalg::{cholesky_solve, cholesky_workspace_len, mat_idx};
use math_scalars::workspace::SolverWorkspace;

/// Workspace slices used inside the backward Riccati recursion.
pub(super) struct RiccatiScratch<'a> {
    pub vx: &'a mut [f64],
    pub vxx: &'a mut [f64],
    pub vx_next: &'a mut [f64],
    pub vxx_next: &'a mut [f64],
    pub lx: &'a mut [f64],
    pub lu: &'a mut [f64],
    pub lxx: &'a mut [f64],
    pub luu: &'a mut [f64],
    pub lux: &'a mut [f64],
    pub qx: &'a mut [f64],
    pub qu: &'a mut [f64],
    pub qxx: &'a mut [f64],
    pub quu: &'a mut [f64],
    pub qux: &'a mut [f64],
    pub tmp_nx_nx: &'a mut [f64],
    pub tmp_nu_nx: &'a mut [f64],
    pub rhs_u: &'a mut [f64],
    pub sol_u: &'a mut [f64],
    pub chol_ws: SolverWorkspace<'a, f64>,
}

#[inline]
pub(super) const fn riccati_scratch_len(nx: usize, nu: usize) -> usize {
    let nx2 = nx.saturating_mul(nx);
    let nu2 = nu.saturating_mul(nu);
    let nunx = nu.saturating_mul(nx);
    let chol = cholesky_workspace_len(nu);
    nx.saturating_mul(4)
        .saturating_add(nu.saturating_mul(4))
        .saturating_add(nx2.saturating_mul(5))
        .saturating_add(nu2.saturating_mul(2))
        .saturating_add(nunx.saturating_mul(3))
        .saturating_add(chol)
}

impl<'a> RiccatiScratch<'a> {
    #[verified_engine::verified]
    pub(super) fn split_from_slice(
        buf: &'a mut [f64],
        nx: usize,
        nu: usize,
    ) -> Result<Self, OptError> {
        let req = riccati_scratch_len(nx, nu);
        if buf.len() < req {
            return Err(OptError::WorkspaceTooSmall {
                required: req,
                available: buf.len(),
            });
        }
        let nx2 = nx * nx;
        let nu2 = nu * nu;
        let nunx = nu * nx;
        let (vx, r) = buf.split_at_mut(nx);
        let (vxx, r) = r.split_at_mut(nx2);
        let (vx_next, r) = r.split_at_mut(nx);
        let (vxx_next, r) = r.split_at_mut(nx2);
        let (lx, r) = r.split_at_mut(nx);
        let (lu, r) = r.split_at_mut(nu);
        let (lxx, r) = r.split_at_mut(nx2);
        let (luu, r) = r.split_at_mut(nu2);
        let (lux, r) = r.split_at_mut(nunx);
        let (qx, r) = r.split_at_mut(nx);
        let (qu, r) = r.split_at_mut(nu);
        let (qxx, r) = r.split_at_mut(nx2);
        let (quu, r) = r.split_at_mut(nu2);
        let (qux, r) = r.split_at_mut(nunx);
        let (tmp_nx_nx, r) = r.split_at_mut(nx2);
        let (tmp_nu_nx, r) = r.split_at_mut(nunx);
        let (rhs_u, r) = r.split_at_mut(nu);
        let (sol_u, chol_buf) = r.split_at_mut(nu);
        Ok(Self {
            vx,
            vxx,
            vx_next,
            vxx_next,
            lx,
            lu,
            lxx,
            luu,
            lux,
            qx,
            qu,
            qxx,
            quu,
            qux,
            tmp_nx_nx,
            tmp_nu_nx,
            rhs_u,
            sol_u,
            chol_ws: SolverWorkspace::new(chol_buf),
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub(super) fn backward_riccati_pass<C: StageCost>(
    cost: &C,
    horizon: usize,
    nx: usize,
    nu: usize,
    reg_start: f64,
    reg_max: f64,
    x_traj: &[f64],
    u_traj: &[f64],
    fx_all: &[f64],
    fu_all: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    k_ff: &mut [f64],
    k_fb: &mut [f64],
    sc: &mut RiccatiScratch<'_>,
) -> Result<(f64, f64, f64), OptError> {
    let mut reg = reg_start;
    for _attempt in 0..16 {
        if let Some((d_lin, max_ff)) = try_backward_sweep(
            cost, horizon, nx, nu, reg, x_traj, u_traj, fx_all, fu_all, u_min, u_max, k_ff, k_fb,
            sc,
        ) {
            return Ok((d_lin, max_ff, reg));
        }
        reg *= 10.0;
        if reg > reg_max {
            break;
        }
    }
    Err(OptError::NumericalInstability)
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn try_backward_sweep<C: StageCost>(
    cost: &C,
    horizon: usize,
    nx: usize,
    nu: usize,
    reg: f64,
    x_traj: &[f64],
    u_traj: &[f64],
    fx_all: &[f64],
    fu_all: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    k_ff: &mut [f64],
    k_fb: &mut [f64],
    sc: &mut RiccatiScratch<'_>,
) -> Option<(f64, f64)> {
    cost.terminal_derivatives(&x_traj[horizon * nx..(horizon + 1) * nx], sc.vx, sc.vxx);
    let mut d_lin = 0.0_f64;
    let mut max_ff = 0.0_f64;

    for k in (0..horizon).rev() {
        let x_k = &x_traj[k * nx..(k + 1) * nx];
        let u_k = &u_traj[k * nu..(k + 1) * nu];
        let fx = &fx_all[k * nx * nx..(k + 1) * nx * nx];
        let fu = &fu_all[k * nx * nu..(k + 1) * nx * nu];
        cost.stage_derivatives(k, x_k, u_k, sc.lx, sc.lu, sc.lxx, sc.luu, sc.lux);
        assemble_q_blocks(nx, nu, reg, fx, fu, sc);

        let ff_k = &mut k_ff[k * nu..(k + 1) * nu];
        let fb_k = &mut k_fb[k * nu * nx..(k + 1) * nu * nx];
        solve_stage_gains(nx, nu, u_k, u_min, u_max, ff_k, fb_k, sc)?;

        for (&ff_j, &qu_j) in ff_k.iter().zip(sc.qu.iter()).take(nu) {
            d_lin += -ff_j * qu_j;
            max_ff = max_ff.max(ff_j.abs());
        }
        update_value_function(nx, nu, ff_k, fb_k, sc);
    }
    Some((d_lin.max(0.0), max_ff))
}

#[verified_engine::verified]
fn assemble_q_blocks(
    nx: usize,
    nu: usize,
    reg: f64,
    fx: &[f64],
    fu: &[f64],
    sc: &mut RiccatiScratch<'_>,
) {
    for i in 0..nx {
        let mut s = sc.lx[i];
        for r in 0..nx {
            s += fx[mat_idx(r, i, nx)] * sc.vx[r];
        }
        sc.qx[i] = s;
    }
    for j in 0..nu {
        let mut s = sc.lu[j];
        for r in 0..nx {
            s += fu[mat_idx(r, j, nu)] * sc.vx[r];
        }
        sc.qu[j] = s;
    }
    // tmp_nx_nx = vxx * fx; tmp_nu_nx = fu^T * vxx
    for r in 0..nx {
        for c in 0..nx {
            let mut s = 0.0;
            for p in 0..nx {
                s += sc.vxx[mat_idx(r, p, nx)] * fx[mat_idx(p, c, nx)];
            }
            sc.tmp_nx_nx[mat_idx(r, c, nx)] = s;
        }
    }
    for i in 0..nx {
        for j in 0..nx {
            let mut s = sc.lxx[mat_idx(i, j, nx)];
            for r in 0..nx {
                s += fx[mat_idx(r, i, nx)] * sc.tmp_nx_nx[mat_idx(r, j, nx)];
            }
            sc.qxx[mat_idx(i, j, nx)] = s;
        }
    }
    for j in 0..nu {
        for c in 0..nx {
            let mut s = sc.lux[mat_idx(j, c, nx)];
            for r in 0..nx {
                s += fu[mat_idx(r, j, nu)] * sc.tmp_nx_nx[mat_idx(r, c, nx)];
            }
            sc.qux[mat_idx(j, c, nx)] = s;
        }
    }
    for j1 in 0..nu {
        for j2 in 0..nu {
            let mut s = sc.luu[mat_idx(j1, j2, nu)];
            for r in 0..nx {
                for p in 0..nx {
                    let v_rp = sc.vxx[mat_idx(r, p, nx)] + if r == p { reg } else { 0.0 };
                    s += fu[mat_idx(r, j1, nu)] * v_rp * fu[mat_idx(p, j2, nu)];
                }
            }
            if j1 == j2 {
                s += reg;
            }
            sc.quu[mat_idx(j1, j2, nu)] = s;
        }
    }
    symmetrize_in_place(sc.quu, nu);
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn solve_stage_gains(
    nx: usize,
    nu: usize,
    u_k: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    ff_k: &mut [f64],
    fb_k: &mut [f64],
    sc: &mut RiccatiScratch<'_>,
) -> Option<()> {
    for j in 0..nu {
        sc.rhs_u[j] = -sc.qu[j];
    }
    let mut sub_ws = sc.chol_ws.reborrow();
    cholesky_solve(sc.quu, sc.rhs_u, ff_k, nu, &mut sub_ws).ok()?;

    for c in 0..nx {
        for j in 0..nu {
            sc.rhs_u[j] = -sc.qux[mat_idx(j, c, nx)];
        }
        let mut col_ws = sc.chol_ws.reborrow();
        cholesky_solve(sc.quu, sc.rhs_u, sc.sol_u, nu, &mut col_ws).ok()?;
        for j in 0..nu {
            fb_k[mat_idx(j, c, nx)] = sc.sol_u[j];
        }
    }

    // Box-projected step clamping and active-constraint feedback zeroing.
    for j in 0..nu {
        let mut stepped = u_k[j] + ff_k[j];
        let mut clamped = false;
        if let Some(lo) = u_min
            && lo[j].is_finite()
            && stepped <= lo[j]
        {
            stepped = lo[j];
            clamped = true;
        }
        if let Some(hi) = u_max
            && hi[j].is_finite()
            && stepped >= hi[j]
        {
            stepped = hi[j];
            clamped = true;
        }
        ff_k[j] = stepped - u_k[j];
        if clamped {
            fb_k[j * nx..(j + 1) * nx].fill(0.0);
        }
    }
    Some(())
}

#[verified_engine::verified]
fn update_value_function(
    nx: usize,
    nu: usize,
    ff_k: &[f64],
    fb_k: &[f64],
    sc: &mut RiccatiScratch<'_>,
) {
    // rhs_u = Quu * ff_k
    for j1 in 0..nu {
        let mut s = 0.0;
        for (j2, &ff_j2) in ff_k.iter().enumerate().take(nu) {
            s += sc.quu[mat_idx(j1, j2, nu)] * ff_j2;
        }
        sc.rhs_u[j1] = s;
    }
    for i in 0..nx {
        let mut g = sc.qx[i];
        for (j, &ff_j) in ff_k.iter().enumerate().take(nu) {
            let ji = mat_idx(j, i, nx);
            g += fb_k[ji] * (sc.rhs_u[j] + sc.qu[j]) + sc.qux[ji] * ff_j;
        }
        sc.vx_next[i] = g;
    }
    // tmp_nu_nx = Quu * K_fb
    for j1 in 0..nu {
        for c in 0..nx {
            let mut s = 0.0;
            for j2 in 0..nu {
                s += sc.quu[mat_idx(j1, j2, nu)] * fb_k[mat_idx(j2, c, nx)];
            }
            sc.tmp_nu_nx[mat_idx(j1, c, nx)] = s;
        }
    }
    for i in 0..nx {
        for c in 0..nx {
            let mut h = sc.qxx[mat_idx(i, c, nx)];
            for j in 0..nu {
                let ji = mat_idx(j, i, nx);
                let jc = mat_idx(j, c, nx);
                h += fb_k[ji] * (sc.tmp_nu_nx[jc] + sc.qux[jc]) + sc.qux[ji] * fb_k[jc];
            }
            sc.vxx_next[mat_idx(i, c, nx)] = h;
        }
    }
    symmetrize_in_place(sc.vxx_next, nx);
    sc.vx.copy_from_slice(sc.vx_next);
    sc.vxx.copy_from_slice(sc.vxx_next);
}

#[verified_engine::verified]
fn symmetrize_in_place(mat: &mut [f64], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let ij = mat_idx(i, j, n);
            let ji = mat_idx(j, i, n);
            let avg = 0.5 * (mat[ij] + mat[ji]);
            mat[ij] = avg;
            mat[ji] = avg;
        }
    }
}
