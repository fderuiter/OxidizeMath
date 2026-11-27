// [cite:engine_opt]
//! Constrained receding-horizon projected nonlinear Model Predictive Control (`MPC`) over `SolverWorkspace`.

use crate::error::OptError;
use crate::ilqr::{ControlModel, IlqrConfig, QuadraticBoxCost, ilqr_workspace_len, solve_ilqr};
use math_scalars::workspace::SolverWorkspace;

/// Hard control box bounds $[u_{\min}, u_{\max}]$ and tightened state window $[x_{\min}, x_{\max}]$ for [`solve_mpc`].
#[derive(Debug, Clone, Copy)]
pub struct MpcProblem<'a> {
    /// Diagonal running state tracking weights $Q \in \mathbb{R}^{n_x}_{\ge 0}$.
    pub q_diag: &'a [f64],
    /// Diagonal running control effort weights $R \in \mathbb{R}^{n_u}_{> 0}$.
    pub r_diag: &'a [f64],
    /// Diagonal terminal state tracking weights $Q_f \in \mathbb{R}^{n_x}_{\ge 0}$.
    pub qf_diag: &'a [f64],
    /// Reference state setpoint $x_{\text{ref}} \in \mathbb{R}^{n_x}$.
    pub x_ref: &'a [f64],
    /// Reference control input $u_{\text{ref}} \in \mathbb{R}^{n_u}$.
    pub u_ref: &'a [f64],
    /// Lower control bounds $u_{\min} \in \mathbb{R}^{n_u}$.
    pub u_min: &'a [f64],
    /// Upper control bounds $u_{\max} \in \mathbb{R}^{n_u}$.
    pub u_max: &'a [f64],
    /// Lower state window bounds $x_{\min} \in \mathbb{R}^{n_x}$ (`f64::NEG_INFINITY` disables per state).
    pub x_min: &'a [f64],
    /// Upper state window bounds $x_{\max} \in \mathbb{R}^{n_x}$ (`f64::INFINITY` disables per state).
    pub x_max: &'a [f64],
    /// Exact-like quadratic state barrier weight $w_{\text{state}} \ge 0$.
    pub state_penalty_weight: f64,
    /// Optional inward safety margin $\delta \ge 0$ tightening $[x_{\min} + \delta, x_{\max} - \delta]$.
    pub state_margin: f64,
}

/// Solver parameters for receding-horizon projected `MPC`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MpcConfig {
    /// Receding prediction horizon $N \ge 1$.
    pub horizon: usize,
    /// Maximum projected iLQR iterations per MPC solve.
    pub max_iterations: usize,
    /// Optimality tolerance on feedforward step and cost change.
    pub tolerance: f64,
    /// Initial Cholesky Hessian regularization $\lambda_0 > 0$.
    pub reg_init: f64,
}

impl Default for MpcConfig {
    fn default() -> Self {
        Self {
            horizon: 12,
            max_iterations: 35,
            tolerance: 1.0e-6,
            reg_init: 1.0e-6,
        }
    }
}

/// Telemetry returned by [`solve_mpc`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MpcStats {
    /// Number of projected iLQR sweeps executed.
    pub iterations: usize,
    /// Optimal finite-horizon objective value.
    pub optimal_cost: f64,
    /// Maximum violation of un-tightened `[x_min, x_max]` across predicted steps `1..=N`.
    pub max_state_violation: f64,
    /// Whether the inner projected solver converged within tolerance.
    pub converged: bool,
}

/// Returns the exact `SolverWorkspace<f64>` capacity required by [`solve_mpc`].
#[inline]
#[must_use]
pub const fn mpc_workspace_len(horizon: usize, nx: usize, nu: usize) -> usize {
    nx.saturating_mul(2)
        .saturating_add(ilqr_workspace_len(horizon, nx, nu))
}

/// Shifts a warm-start control sequence forward by one receding-horizon step ($u_k \leftarrow u_{k+1}$).
#[verified_engine::verified]
pub fn warm_start_shift_controls(u_traj: &mut [f64], horizon: usize, nu: usize) {
    if horizon <= 1 || nu == 0 || u_traj.len() < horizon * nu {
        return;
    }
    u_traj.copy_within(nu..horizon * nu, 0);
    let (head, tail) = u_traj.split_at_mut((horizon - 1) * nu);
    tail[..nu].copy_from_slice(&head[(horizon - 2) * nu..(horizon - 1) * nu]);
}

/// Solves one constrained receding-horizon projected `MPC` step into `(x_traj, u_traj, u0_out)`.
///
/// $$
/// \min_{u_{0:N-1} \in [u_{\min}, u_{\max}]} \sum_{k=0}^{N-1} \ell_k(x_k, u_k) + \ell_N(x_N)
/// \quad \text{s.t.} \quad x_{k+1} = f_k(x_k, u_k), \; x_k \in [x_{\min}, x_{\max}]
/// $$
///
/// Reference: Lucia et al. (`do-mpc`, 2017), DOI: 10.1016/j.conengprac.2017.02.002;
/// Tassa, Mansard & Todorov (2014), DOI: 10.1109/ICRA.2014.6907001.
///
/// # Stability
/// Enforces hard box projection $u_k \in [u_{\min}, u_{\max}]$ on every forward rollout while penalizing
/// tightened state excursions via `cholesky_solve` Riccati sweeps and Armijo backtracking.
///
/// # Domain of Validity
/// Requires `nx >= 1`, `nu >= 1`, `cfg.horizon >= 1`, matching slice dimensions on `prob`, `x0`, `x_traj`,
/// `u_traj`, and `u0_out`, and `ws.len() >= mpc_workspace_len(cfg.horizon, nx, nu)`.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn solve_mpc<M: ControlModel>(
    model: &M,
    prob: &MpcProblem<'_>,
    cfg: &MpcConfig,
    x0: &[f64],
    x_traj: &mut [f64],
    u_traj: &mut [f64],
    u0_out: &mut [f64],
    ws: &mut SolverWorkspace<'_, f64>,
) -> Result<MpcStats, OptError> {
    let nx = model.state_dim();
    let nu = model.control_dim();
    validate_mpc_problem(prob, cfg, nx, nu, u0_out.len())?;
    let req = mpc_workspace_len(cfg.horizon, nx, nu);
    if ws.len() < req {
        return Err(OptError::WorkspaceTooSmall {
            required: req,
            available: ws.len(),
        });
    }
    let (x_lo_tight, mut r1) = ws.try_split(nx)?;
    let (x_hi_tight, mut ilqr_ws) = r1.try_split(nx)?;
    build_tightened_bounds(
        prob.x_min,
        prob.x_max,
        prob.state_margin,
        x_lo_tight,
        x_hi_tight,
    )?;

    let quad_cost = QuadraticBoxCost {
        q_diag: prob.q_diag,
        r_diag: prob.r_diag,
        qf_diag: prob.qf_diag,
        x_ref: prob.x_ref,
        u_ref: prob.u_ref,
        x_min: Some(x_lo_tight),
        x_max: Some(x_hi_tight),
        bound_weight: prob.state_penalty_weight,
    };
    let ilqr_cfg = IlqrConfig {
        horizon: cfg.horizon,
        max_iterations: cfg.max_iterations,
        tolerance: cfg.tolerance,
        reg_init: cfg.reg_init,
        reg_max: 1.0e9,
        armijo_c1: 1.0e-4,
    };
    let res = solve_ilqr(
        model,
        &quad_cost,
        &ilqr_cfg,
        x0,
        Some(prob.u_min),
        Some(prob.u_max),
        x_traj,
        u_traj,
        &mut ilqr_ws,
    )?;
    u0_out.copy_from_slice(&u_traj[..nu]);
    let viol = compute_max_state_violation(x_traj, cfg.horizon, nx, prob.x_min, prob.x_max);
    debug_assert!(res.total_cost.is_finite());
    debug_assert!(viol.is_finite() && viol >= 0.0);
    debug_assert_eq!(u0_out.len(), nu);
    Ok(MpcStats {
        iterations: res.iterations,
        optimal_cost: res.total_cost,
        max_state_violation: viol,
        converged: res.converged,
    })
}

#[verified_engine::verified]
fn validate_mpc_problem(
    prob: &MpcProblem<'_>,
    cfg: &MpcConfig,
    nx: usize,
    nu: usize,
    u0_len: usize,
) -> Result<(), OptError> {
    if prob.q_diag.len() != nx
        || prob.qf_diag.len() != nx
        || prob.x_ref.len() != nx
        || prob.x_min.len() != nx
        || prob.x_max.len() != nx
        || prob.r_diag.len() != nu
        || prob.u_ref.len() != nu
        || prob.u_min.len() != nu
        || prob.u_max.len() != nu
        || u0_len != nu
        || cfg.horizon == 0
    {
        return Err(OptError::InvalidDimension);
    }
    if !prob.state_penalty_weight.is_finite() || !prob.state_margin.is_finite() {
        return Err(OptError::NonFiniteInput);
    }
    if prob.state_penalty_weight < 0.0 || prob.state_margin < 0.0 {
        return Err(OptError::InvalidParameter);
    }
    for i in 0..nx {
        if prob.q_diag[i] < 0.0 || prob.qf_diag[i] < 0.0 || prob.x_min[i] > prob.x_max[i] {
            return Err(OptError::InvalidParameter);
        }
    }
    for j in 0..nu {
        if prob.r_diag[j] <= 0.0 || prob.u_min[j] > prob.u_max[j] {
            return Err(OptError::InvalidParameter);
        }
    }
    debug_assert!(nx >= 1 && nu >= 1);
    debug_assert!(cfg.horizon >= 1);
    Ok(())
}

#[verified_engine::verified]
fn build_tightened_bounds(
    x_min: &[f64],
    x_max: &[f64],
    margin: f64,
    lo_out: &mut [f64],
    hi_out: &mut [f64],
) -> Result<(), OptError> {
    for i in 0..x_min.len() {
        let lo = x_min[i];
        let hi = x_max[i];
        if lo.is_finite() && hi.is_finite() {
            let half = 0.5 * (hi - lo);
            let m = margin.min(0.45 * half);
            lo_out[i] = lo + m;
            hi_out[i] = hi - m;
        } else {
            lo_out[i] = if lo.is_finite() { lo + margin } else { lo };
            hi_out[i] = if hi.is_finite() { hi - margin } else { hi };
        }
        debug_assert!(lo_out[i] <= hi_out[i]);
    }
    debug_assert_eq!(lo_out.len(), x_min.len());
    debug_assert_eq!(hi_out.len(), x_max.len());
    Ok(())
}

#[verified_engine::verified]
fn compute_max_state_violation(
    x_traj: &[f64],
    horizon: usize,
    nx: usize,
    x_min: &[f64],
    x_max: &[f64],
) -> f64 {
    let mut max_v = 0.0_f64;
    for k in 1..=horizon {
        let x_k = &x_traj[k * nx..(k + 1) * nx];
        for i in 0..nx {
            if x_min[i].is_finite() && x_k[i] < x_min[i] {
                max_v = max_v.max(x_min[i] - x_k[i]);
            }
            if x_max[i].is_finite() && x_k[i] > x_max[i] {
                max_v = max_v.max(x_k[i] - x_max[i]);
            }
        }
    }
    max_v
}
