//! Multi-period Almgren-Chriss optimal market execution over `engine_opt::mpc`.

use engine_opt::ilqr::ControlModel;
use engine_opt::mpc::{MpcConfig, MpcProblem, MpcStats, mpc_workspace_len, solve_mpc};
use engine_opt::{OptError, ScalarWorkspace};

/// Parameters for multi-period mean-variance Almgren-Chriss optimal trade execution.
///
/// Liquidates initial inventory $X_0 > 0$ over $N$ intervals of duration $\tau = T / N$ with
/// inventory state $x_{k+1} = x_k - u_k$ and participation bounds $0 \le u_k \le u_{\max}$:
///
/// $$
/// \min_{0 \le u_k \le u_{\max}} \sum_{k=0}^{N-1} \left(\frac{1}{2}\lambda \sigma^2 \tau \, x_k^2 + \frac{\eta}{2 \tau} u_k^2\right) + \frac{w_T}{2} x_N^2
/// $$
///
/// Reference: Almgren & Chriss (2001), *Journal of Risk* 3(2), 5–39, DOI: 10.21314/JOR.2001.041;
/// Lucia et al. (`do-mpc`, 2017), DOI: 10.1016/j.conengprac.2017.02.002;
/// [`docs/math_specs/pure_math.md`](../../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Strictly positive temporary-impact curvature $\eta / \tau > 0$ guarantees SPD Riccati $Q_{uu}$
/// blocks, and in the risk-neutral limit $\kappa T \to 0$ the closed-form benchmark smoothly
/// reduces to TWAP $x_k = X_0(1 - k/N)$ without $0/0$ cancellation.
///
/// # Domain of Validity
/// Requires $X_0 > 0$, $T > 0$, $\sigma \ge 0$, $\lambda \ge 0$, $\eta > 0$, and $u_{\max} \ge X_0 / N$.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AlmgrenChrissConfig {
    /// Initial share or notional inventory $X_0 > 0$ to liquidate.
    pub initial_inventory: f64,
    /// Total execution horizon $T > 0$ (e.g., trading days or hours).
    pub horizon_time: f64,
    /// Asset return volatility $\sigma \ge 0$ per $\sqrt{\text{time}}$.
    pub volatility: f64,
    /// Mean-variance risk-aversion parameter $\lambda \ge 0$.
    pub risk_aversion: f64,
    /// Temporary linear price-impact coefficient $\eta > 0$.
    pub temporary_impact: f64,
    /// Maximum slice volume $u_{\max} > 0$ permitted in any single period (`f64::INFINITY` for unconstrained).
    pub max_trade_per_period: f64,
}

/// Exact `f64` workspace length required by [`solve_almgren_chriss_mpc_into`] for `periods`.
#[inline]
#[must_use]
#[verified_engine::verified]
pub const fn almgren_chriss_workspace_len(periods: usize) -> usize {
    mpc_workspace_len(periods, 1, 1)
}

struct InventoryDepletionModel;

impl ControlModel for InventoryDepletionModel {
    #[inline]
    fn state_dim(&self) -> usize {
        1
    }

    #[inline]
    fn control_dim(&self) -> usize {
        1
    }

    #[verified_engine::verified]
    fn step_and_linearize(
        &self,
        _step: usize,
        x: &[f64],
        u: &[f64],
        x_next: &mut [f64],
        fx: &mut [f64],
        fu: &mut [f64],
    ) -> Result<(), OptError> {
        x_next[0] = x[0] - u[0];
        fx[0] = 1.0;
        fu[0] = -1.0;
        Ok(())
    }
}

/// Evaluates the exact unconstrained discrete-time Almgren-Chriss hyperbolic inventory curve into `inventory_out`.
///
/// $$
/// x_k^\star = X_0 \frac{\sinh\bigl(\kappa (T - k\tau)\bigr)}{\sinh(\kappa T)}, \qquad
/// \cosh(\kappa \tau) = 1 + \frac{\lambda \sigma^2 \tau^2}{2 \eta}
/// $$
///
/// Reference: Almgren & Chriss (2001), DOI: 10.21314/JOR.2001.041;
/// [`docs/math_specs/pure_math.md`](../../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Switches to exact linear TWAP $X_0(1 - k/N)$ when $\kappa T < 10^{-6}$ to avoid $\sinh(\kappa T)$ underflow.
///
/// # Domain of Validity
/// Requires `periods >= 1`, `inventory_out.len() == periods + 1`, and validated [`AlmgrenChrissConfig`].
#[verified_engine::verified]
pub fn almgren_chriss_analytical_trajectory_into(
    cfg: &AlmgrenChrissConfig,
    periods: usize,
    inventory_out: &mut [f64],
) -> Result<f64, OptError> {
    validate_ac_cfg(cfg, periods)?;
    if inventory_out.len() != periods + 1 {
        return Err(OptError::InvalidDimension);
    }
    let tau = cfg.horizon_time / (periods as f64);
    let half_ratio = 0.5 * cfg.risk_aversion * cfg.volatility * cfg.volatility * tau * tau
        / cfg.temporary_impact;
    let kappa = (1.0 + half_ratio).acosh() / tau;
    let kt_total = kappa * cfg.horizon_time;

    if kt_total < 1.0e-6 {
        for (k, slot) in inventory_out.iter_mut().enumerate() {
            *slot = cfg.initial_inventory * (1.0 - (k as f64) / (periods as f64));
        }
    } else {
        let denom = kt_total.sinh();
        for (k, slot) in inventory_out.iter_mut().enumerate() {
            let rem_t = cfg.horizon_time - (k as f64) * tau;
            *slot = cfg.initial_inventory * (kappa * rem_t).sinh() / denom;
        }
        inventory_out[periods] = 0.0;
    }
    Ok(kappa)
}

/// Solves constrained multi-period Almgren-Chriss trade execution via `engine_opt::mpc`.
///
/// Computes optimal child-order sizes `trades_out` (`len == periods`) and remaining inventory
/// `inventory_out` (`len == periods + 1`) over caller `SolverWorkspace` with zero heap allocations:
///
/// $$
/// x_{k+1} = x_k - u_k, \qquad 0 \le u_k \le u_{\max}, \qquad x_0 = X_0
/// $$
///
/// Reference: Almgren & Chriss (2001), DOI: 10.21314/JOR.2001.041; Lucia et al. (`do-mpc`, 2017),
/// DOI: 10.1016/j.conengprac.2017.02.002;
/// [`docs/math_specs/pure_math.md`](../../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Recovers the exact discrete hyperbolic schedule $\sinh(\kappa(T - t_k))/\sinh(\kappa T)$ when
/// $u_{\max}$ is non-binding and smoothly caps child slices when participation limits bind.
///
/// # Domain of Validity
/// Requires `periods >= 1`, `trades_out.len() == periods`, `inventory_out.len() == periods + 1`,
/// and `ws.len() >= almgren_chriss_workspace_len(periods)`.
#[verified_engine::verified]
pub fn solve_almgren_chriss_mpc_into(
    cfg: &AlmgrenChrissConfig,
    periods: usize,
    inventory_out: &mut [f64],
    trades_out: &mut [f64],
    ws: &mut ScalarWorkspace<'_, f64>,
) -> Result<MpcStats, OptError> {
    validate_ac_cfg(cfg, periods)?;
    if inventory_out.len() != periods + 1 || trades_out.len() != periods {
        return Err(OptError::InvalidDimension);
    }

    let tau = cfg.horizon_time / (periods as f64);
    let q_val = cfg.risk_aversion * cfg.volatility * cfg.volatility * tau;
    let r_val = cfg.temporary_impact / tau;
    let u_cap = if cfg.max_trade_per_period.is_finite() {
        cfg.max_trade_per_period
    } else {
        cfg.initial_inventory * 2.0
    };
    let twap = (cfg.initial_inventory / (periods as f64)).min(u_cap);
    trades_out.fill(twap);

    let q_diag = [q_val];
    let r_diag = [r_val];
    let qf_diag = [1.0e9 * r_val];
    let x_ref = [0.0];
    let u_ref = [0.0];
    let u_min = [0.0];
    let u_max = [u_cap];
    let x_min = [0.0];
    let x_max = [cfg.initial_inventory];
    let prob = MpcProblem {
        q_diag: &q_diag,
        r_diag: &r_diag,
        qf_diag: &qf_diag,
        x_ref: &x_ref,
        u_ref: &u_ref,
        u_min: &u_min,
        u_max: &u_max,
        x_min: &x_min,
        x_max: &x_max,
        state_penalty_weight: 1.0e4 * r_val,
        state_margin: 0.0,
    };
    let mpc_cfg = MpcConfig {
        horizon: periods,
        max_iterations: 40,
        tolerance: 1.0e-10,
        reg_init: 1.0e-10,
    };
    let x0 = [cfg.initial_inventory];
    let mut u0 = [0.0_f64; 1];
    solve_mpc(
        &InventoryDepletionModel,
        &prob,
        &mpc_cfg,
        &x0,
        inventory_out,
        trades_out,
        &mut u0,
        ws,
    )
}

#[verified_engine::verified]
fn validate_ac_cfg(cfg: &AlmgrenChrissConfig, periods: usize) -> Result<(), OptError> {
    if periods == 0 {
        return Err(OptError::InvalidDimension);
    }
    if !cfg.initial_inventory.is_finite()
        || !cfg.horizon_time.is_finite()
        || !cfg.volatility.is_finite()
        || !cfg.risk_aversion.is_finite()
        || !cfg.temporary_impact.is_finite()
        || cfg.max_trade_per_period.is_nan()
    {
        return Err(OptError::NonFiniteInput);
    }
    let min_slice = cfg.initial_inventory / (periods as f64);
    if cfg.initial_inventory <= 0.0
        || cfg.horizon_time <= 0.0
        || cfg.volatility < 0.0
        || cfg.risk_aversion < 0.0
        || cfg.temporary_impact <= 0.0
        || cfg.max_trade_per_period < min_slice - 1.0e-12
    {
        return Err(OptError::InvalidParameter);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    static AC_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_almgren_chriss_mpc_matches_sinh_closed_form_zero_alloc() {
        const N: usize = 10;
        let cfg = AlmgrenChrissConfig {
            initial_inventory: 10_000.0,
            horizon_time: 1.0,
            volatility: 0.30,
            risk_aversion: 2.5e-3,
            temporary_impact: 2.0e-5,
            max_trade_per_period: f64::INFINITY,
        };
        let mut inv_mpc = [0.0_f64; N + 1];
        let mut trades_mpc = [0.0_f64; N];
        let mut inv_exact = [0.0_f64; N + 1];
        let mut ws_buf = [0.0_f64; almgren_chriss_workspace_len(N)];

        let stats = {
            let _g = AC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let (s, _) = VerifiedEngine::run_verified(|| {
                let _ = almgren_chriss_analytical_trajectory_into(&cfg, N, &mut inv_exact).unwrap();
                let mut ws = ScalarWorkspace::new(&mut ws_buf);
                solve_almgren_chriss_mpc_into(&cfg, N, &mut inv_mpc, &mut trades_mpc, &mut ws)
                    .unwrap()
            });
            s
        };

        assert!(stats.converged);
        assert!(trades_mpc[0] > trades_mpc[N - 1] * 1.5);
        for k in 0..=N {
            let rel_err = (inv_mpc[k] - inv_exact[k]).abs() / cfg.initial_inventory;
            assert!(rel_err < 1.0e-5, "Period {k}: rel_err={rel_err}");
        }
    }

    #[test]
    fn test_almgren_chriss_mpc_enforces_participation_cap_zero_alloc() {
        const N: usize = 10;
        let cfg = AlmgrenChrissConfig {
            initial_inventory: 10_000.0,
            horizon_time: 1.0,
            volatility: 0.30,
            risk_aversion: 2.5e-3,
            temporary_impact: 2.0e-5,
            max_trade_per_period: 1_500.0,
        };
        let mut inv_cap = [0.0_f64; N + 1];
        let mut trades_cap = [0.0_f64; N];
        let mut ws_buf = [0.0_f64; almgren_chriss_workspace_len(N)];

        let stats = {
            let _g = AC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let (s, _) = VerifiedEngine::run_verified(|| {
                let mut ws = ScalarWorkspace::new(&mut ws_buf);
                solve_almgren_chriss_mpc_into(&cfg, N, &mut inv_cap, &mut trades_cap, &mut ws)
                    .unwrap()
            });
            s
        };
        assert!(stats.converged);
        assert!(
            trades_cap
                .iter()
                .all(|&u| (0.0..=1_500.0 + 1.0e-9).contains(&u))
        );
        assert!(inv_cap[N].abs() < 1.0e-2);
    }
}
