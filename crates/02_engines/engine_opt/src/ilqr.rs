// [cite:engine_opt]
//! Single-shooting box-projected iterative LQR (`iLQR`) over `SolverWorkspace`.

mod cost;
mod riccati;

pub use cost::{ControlModel, QuadraticBoxCost, StageCost};

use crate::error::OptError;
use math_scalars::workspace::SolverWorkspace;
use riccati::{RiccatiScratch, backward_riccati_pass};

const BACKTRACK_STEPS: usize = 12;

/// Solver parameters for single-shooting box-projected `iLQR`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IlqrConfig {
    /// Prediction horizon $N \ge 1$.
    pub horizon: usize,
    /// Maximum backward-forward iLQR iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on cost reduction and feedforward norm.
    pub tolerance: f64,
    /// Initial Levenberg-Marquardt Hessian regularization $\lambda_0 > 0$.
    pub reg_init: f64,
    /// Maximum Hessian regularization $\lambda_{\max}$.
    pub reg_max: f64,
    /// Armijo sufficient-decrease parameter $c_1 \in (0, 0.5)$.
    pub armijo_c1: f64,
}

impl Default for IlqrConfig {
    fn default() -> Self {
        Self {
            horizon: 12,
            max_iterations: 40,
            tolerance: 1.0e-7,
            reg_init: 1.0e-6,
            reg_max: 1.0e8,
            armijo_c1: 1.0e-4,
        }
    }
}

/// Convergence telemetry returned by [`solve_ilqr`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IlqrStats {
    /// Number of iLQR iterations executed.
    pub iterations: usize,
    /// Optimal total trajectory cost $J(X^\star, U^\star)$.
    pub total_cost: f64,
    /// Infinity norm of the final feedforward step $\|k_{\text{ff}}\|_\infty$.
    pub max_feedforward: f64,
    /// Whether convergence criteria were satisfied.
    pub converged: bool,
}

/// Returns the exact `SolverWorkspace<f64>` capacity required by [`solve_ilqr`].
#[inline]
#[must_use]
pub const fn ilqr_workspace_len(horizon: usize, nx: usize, nu: usize) -> usize {
    let fx_all = horizon.saturating_mul(nx).saturating_mul(nx);
    let fu_all = horizon.saturating_mul(nx).saturating_mul(nu);
    let k_ff = horizon.saturating_mul(nu);
    let k_fb = horizon.saturating_mul(nu).saturating_mul(nx);
    let x_cand = horizon.saturating_add(1).saturating_mul(nx);
    let u_cand = horizon.saturating_mul(nu);
    let riccati = riccati::riccati_scratch_len(nx, nu);
    fx_all
        .saturating_add(fu_all)
        .saturating_add(k_ff)
        .saturating_add(k_fb)
        .saturating_add(x_cand)
        .saturating_add(u_cand)
        .saturating_add(riccati)
}

/// Solves a finite-horizon optimal-control problem via single-shooting `iLQR` with `cholesky_solve` and Armijo backtracking.
///
/// $$
/// \min_{u_{0:N-1} \in [u_{\min}, u_{\max}]} \sum_{k=0}^{N-1} \ell_k(x_k, u_k) + \ell_N(x_N)
/// \quad \text{s.t.} \quad x_{k+1} = f_k(x_k, u_k)
/// $$
///
/// Reference: Tassa, Mansard & Todorov (2014), DOI: 10.1109/ICRA.2014.6907001;
/// Lucia et al. (`do-mpc`, 2017), DOI: 10.1016/j.conengprac.2017.02.002.
///
/// # Stability
/// Symmetrized Joseph-form Riccati value updates, adaptive Levenberg regularization on $Q_{uu}$ factored by
/// `cholesky_solve`, and Armijo backtracking rollouts ensure monotonic cost descent inside $[u_{\min}, u_{\max}]$.
///
/// # Domain of Validity
/// Requires `nx >= 1`, `nu >= 1`, `cfg.horizon >= 1`, `x0.len() == nx`, `x_traj.len() == (horizon + 1) * nx`,
/// `u_traj.len() == horizon * nu`, valid box bounds `u_min <= u_max`, and `ws.len() >= ilqr_workspace_len(...)`.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn solve_ilqr<M: ControlModel, C: StageCost>(
    model: &M,
    cost: &C,
    cfg: &IlqrConfig,
    x0: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    x_traj: &mut [f64],
    u_traj: &mut [f64],
    ws: &mut SolverWorkspace<'_, f64>,
) -> Result<IlqrStats, OptError> {
    let nx = model.state_dim();
    let nu = model.control_dim();
    validate_ilqr_inputs(cfg, nx, nu, x0, u_min, u_max, x_traj.len(), u_traj.len())?;
    let req = ilqr_workspace_len(cfg.horizon, nx, nu);
    if ws.len() < req {
        return Err(OptError::WorkspaceTooSmall {
            required: req,
            available: ws.len(),
        });
    }
    let h = cfg.horizon;
    let (fx_all, mut r1) = ws.try_split(h * nx * nx)?;
    let (fu_all, mut r2) = r1.try_split(h * nx * nu)?;
    let (k_ff, mut r3) = r2.try_split(h * nu)?;
    let (k_fb, mut r4) = r3.try_split(h * nu * nx)?;
    let (x_cand, mut r5) = r4.try_split((h + 1) * nx)?;
    let (u_cand, mut ric_ws) = r5.try_split(h * nu)?;

    clamp_control_traj(u_traj, h, nu, u_min, u_max);
    let mut curr_cost = rollout_and_linearize(model, cost, h, x0, u_traj, x_traj, fx_all, fu_all)?;
    let mut reg = cfg.reg_init.max(1.0e-12);
    let mut stats = IlqrStats {
        iterations: 0,
        total_cost: curr_cost,
        max_feedforward: 0.0,
        converged: cfg.max_iterations == 0,
    };

    for iter in 0..cfg.max_iterations {
        let mut scratch = RiccatiScratch::split_from_slice(ric_ws.as_mut_slice(), nx, nu)?;
        let (d_lin, max_ff, used_reg) = backward_riccati_pass(
            cost,
            h,
            nx,
            nu,
            reg,
            cfg.reg_max,
            x_traj,
            u_traj,
            fx_all,
            fu_all,
            u_min,
            u_max,
            k_ff,
            k_fb,
            &mut scratch,
        )?;
        reg = used_reg;
        stats.max_feedforward = max_ff;
        stats.iterations = iter + 1;
        if max_ff <= cfg.tolerance {
            stats.converged = true;
            break;
        }
        if let Some(next_cost) = armijo_line_search(
            model, cost, cfg, nx, nu, curr_cost, d_lin, x0, x_traj, u_traj, k_ff, k_fb, u_min,
            u_max, x_cand, u_cand,
        )? {
            let improvement = curr_cost - next_cost;
            x_traj.copy_from_slice(x_cand);
            u_traj.copy_from_slice(u_cand);
            curr_cost = rollout_and_linearize(model, cost, h, x0, u_traj, x_traj, fx_all, fu_all)?;
            stats.total_cost = curr_cost;
            reg = (reg * 0.5).max(1.0e-10);
            if improvement <= cfg.tolerance * (1.0 + curr_cost.abs()) {
                stats.converged = true;
                break;
            }
        } else {
            reg *= 8.0;
            if reg > cfg.reg_max {
                stats.converged = max_ff <= 10.0 * cfg.tolerance;
                break;
            }
        }
    }
    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn validate_ilqr_inputs(
    cfg: &IlqrConfig,
    nx: usize,
    nu: usize,
    x0: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    x_len: usize,
    u_len: usize,
) -> Result<(), OptError> {
    if nx == 0
        || nu == 0
        || cfg.horizon == 0
        || x0.len() != nx
        || x_len != (cfg.horizon + 1) * nx
        || u_len != cfg.horizon * nu
    {
        return Err(OptError::InvalidDimension);
    }
    if x0.iter().any(|v| !v.is_finite())
        || !cfg.tolerance.is_finite()
        || !cfg.reg_init.is_finite()
        || !cfg.reg_max.is_finite()
        || !cfg.armijo_c1.is_finite()
    {
        return Err(OptError::NonFiniteInput);
    }
    if cfg.tolerance < 0.0
        || cfg.reg_init <= 0.0
        || cfg.reg_max < cfg.reg_init
        || cfg.armijo_c1 < 0.0
    {
        return Err(OptError::InvalidParameter);
    }
    if let Some(lo) = u_min
        && lo.len() != nu
    {
        return Err(OptError::InvalidDimension);
    }
    if let Some(hi) = u_max
        && hi.len() != nu
    {
        return Err(OptError::InvalidDimension);
    }
    if let (Some(lo), Some(hi)) = (u_min, u_max) {
        for j in 0..nu {
            if lo[j] > hi[j] {
                return Err(OptError::InvalidParameter);
            }
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn clamp_control_traj(
    u_traj: &mut [f64],
    horizon: usize,
    nu: usize,
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
) {
    for k in 0..horizon {
        let u_k = &mut u_traj[k * nu..(k + 1) * nu];
        for j in 0..nu {
            if let Some(lo) = u_min
                && lo[j].is_finite()
            {
                u_k[j] = u_k[j].max(lo[j]);
            }
            if let Some(hi) = u_max
                && hi[j].is_finite()
            {
                u_k[j] = u_k[j].min(hi[j]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn rollout_and_linearize<M: ControlModel, C: StageCost>(
    model: &M,
    cost: &C,
    horizon: usize,
    x0: &[f64],
    u_traj: &[f64],
    x_traj: &mut [f64],
    fx_all: &mut [f64],
    fu_all: &mut [f64],
) -> Result<f64, OptError> {
    let nx = model.state_dim();
    let nu = model.control_dim();
    x_traj[..nx].copy_from_slice(x0);
    let mut total = 0.0;
    for k in 0..horizon {
        let (head, tail) = x_traj.split_at_mut((k + 1) * nx);
        let x_k = &head[k * nx..(k + 1) * nx];
        let x_next = &mut tail[..nx];
        let u_k = &u_traj[k * nu..(k + 1) * nu];
        if u_k.iter().any(|v| !v.is_finite()) {
            return Err(OptError::NonFiniteInput);
        }
        total += cost.stage_cost(k, x_k, u_k);
        model.step_and_linearize(
            k,
            x_k,
            u_k,
            x_next,
            &mut fx_all[k * nx * nx..(k + 1) * nx * nx],
            &mut fu_all[k * nx * nu..(k + 1) * nx * nu],
        )?;
        if x_next.iter().any(|v| !v.is_finite()) {
            return Err(OptError::NumericalInstability);
        }
    }
    total += cost.terminal_cost(&x_traj[horizon * nx..(horizon + 1) * nx]);
    if !total.is_finite() {
        return Err(OptError::NumericalInstability);
    }
    Ok(total)
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn armijo_line_search<M: ControlModel, C: StageCost>(
    model: &M,
    cost: &C,
    cfg: &IlqrConfig,
    nx: usize,
    nu: usize,
    curr_cost: f64,
    d_lin: f64,
    x0: &[f64],
    x_nom: &[f64],
    u_nom: &[f64],
    k_ff: &[f64],
    k_fb: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    x_cand: &mut [f64],
    u_cand: &mut [f64],
) -> Result<Option<f64>, OptError> {
    let mut fx_dummy_ok = true;
    let mut alpha = 1.0_f64;
    for _ in 0..BACKTRACK_STEPS {
        if let Some(cand_cost) = try_single_shoot(
            model,
            cost,
            cfg.horizon,
            nx,
            nu,
            alpha,
            x0,
            x_nom,
            u_nom,
            k_ff,
            k_fb,
            u_min,
            u_max,
            x_cand,
            u_cand,
        ) {
            let thresh = curr_cost - cfg.armijo_c1 * alpha * d_lin.max(0.0);
            if cand_cost <= thresh && cand_cost < curr_cost {
                return Ok(Some(cand_cost));
            }
        } else {
            fx_dummy_ok = false;
        }
        alpha *= 0.5;
    }
    let _ = fx_dummy_ok;
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn try_single_shoot<M: ControlModel, C: StageCost>(
    model: &M,
    cost: &C,
    horizon: usize,
    nx: usize,
    nu: usize,
    alpha: f64,
    x0: &[f64],
    x_nom: &[f64],
    u_nom: &[f64],
    k_ff: &[f64],
    k_fb: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    x_cand: &mut [f64],
    u_cand: &mut [f64],
) -> Option<f64> {
    x_cand[..nx].copy_from_slice(x0);
    let mut total = 0.0;
    let mut fx_tmp = [0.0_f64; 64];
    let mut fu_tmp = [0.0_f64; 64];
    if nx * nx > fx_tmp.len() || nx * nu > fu_tmp.len() {
        return None;
    }
    for k in 0..horizon {
        let (head, tail) = x_cand.split_at_mut((k + 1) * nx);
        let x_k = &head[k * nx..(k + 1) * nx];
        let x_next = &mut tail[..nx];
        let x_bar = &x_nom[k * nx..(k + 1) * nx];
        let u_bar = &u_nom[k * nu..(k + 1) * nu];
        let ff_k = &k_ff[k * nu..(k + 1) * nu];
        let fb_k = &k_fb[k * nu * nx..(k + 1) * nu * nx];
        let u_out = &mut u_cand[k * nu..(k + 1) * nu];
        for j in 0..nu {
            let mut val = u_bar[j] + alpha * ff_k[j];
            for i in 0..nx {
                val += fb_k[j * nx + i] * (x_k[i] - x_bar[i]);
            }
            if let Some(lo) = u_min
                && lo[j].is_finite()
            {
                val = val.max(lo[j]);
            }
            if let Some(hi) = u_max
                && hi[j].is_finite()
            {
                val = val.min(hi[j]);
            }
            u_out[j] = val;
        }
        total += cost.stage_cost(k, x_k, u_out);
        model
            .step_and_linearize(
                k,
                x_k,
                u_out,
                x_next,
                &mut fx_tmp[..nx * nx],
                &mut fu_tmp[..nx * nu],
            )
            .ok()?;
        if x_next.iter().any(|v| !v.is_finite()) || !total.is_finite() {
            return None;
        }
    }
    total += cost.terminal_cost(&x_cand[horizon * nx..(horizon + 1) * nx]);
    total.is_finite().then_some(total)
}
