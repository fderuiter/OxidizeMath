// [cite:engine_opt]
//! Deterministic regret matching (`CFR+` / Hart-Mas-Colell) for $N$-player polymatrix games over `SolverWorkspace`.

use crate::error::OptError;
use math_scalars::workspace::SolverWorkspace;

/// Configuration for polymatrix regret-matching equilibrium computation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PolymatrixConfig {
    /// Maximum regret-matching iterations $T_{\max} \ge 1$.
    pub max_iterations: usize,
    /// Target Nash exploitability $\varepsilon \ge 0$ on the time-averaged strategy profile.
    pub tolerance: f64,
    /// Whether to floor negative cumulative regrets (`CFR+` acceleration) and use linear iteration weights.
    pub cfr_plus: bool,
}

impl Default for PolymatrixConfig {
    fn default() -> Self {
        Self {
            max_iterations: 4_000,
            tolerance: 1.0e-4,
            cfr_plus: true,
        }
    }
}

/// Convergence telemetry returned by [`solve_polymatrix_regret_matching`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PolymatrixNashStats {
    /// Iterations executed.
    pub iterations: usize,
    /// Worst-case unilateral best-response exploitability $\max_p (\max_a u_p(a, \bar{\sigma}_{-p}) - u_p(\bar{\sigma}))$.
    pub exploitability: f64,
    /// Whether `exploitability <= tolerance` was reached.
    pub converged: bool,
}

/// Returns the exact `SolverWorkspace<f64>` capacity required by [`solve_polymatrix_regret_matching`].
#[inline]
#[must_use]
pub const fn nash_polymatrix_workspace_len(total_actions: usize) -> usize {
    total_actions.saturating_mul(4)
}

/// Computes an empirical strategy profile of an $N$-player polymatrix game via deterministic regret matching.
///
/// $$
/// R_p^{t}(a) = \bigl[R_p^{t-1}(a) + u_p(a, \sigma_{-p}^t) - u_p(\sigma^t)\bigr]_+, \qquad
/// \sigma_p^{t+1}(a) \propto \max\bigl(R_p^t(a), 0\bigr)
/// $$
///
/// Reference: Hart & Mas-Colell (2000), DOI: 10.1111/1468-0262.00153;
/// Tammelin et al. (`CFR+`, 2015), arXiv:1407.5042.
///
/// # Stability
/// Positive regret matching satisfies Blackwell approachability with $O(T^{-1/2})$ (or faster under linear-weighted `CFR+`)
/// decay of external regret while keeping every player's strategy on the probability simplex.
///
/// # Domain of Validity
/// Requires `action_counts.len() >= 2`, every `action_counts[p] >= 1`, `payoff_matrix.len() == M * M`
/// where $M = \sum_p m_p$, `avg_strategies_out.len() == M`, and `ws.len() >= nash_polymatrix_workspace_len(M)`.
#[verified_engine::verified]
pub fn solve_polymatrix_regret_matching(
    action_counts: &[usize],
    payoff_matrix: &[f64],
    cfg: &PolymatrixConfig,
    avg_strategies_out: &mut [f64],
    ws: &mut SolverWorkspace<'_, f64>,
) -> Result<PolymatrixNashStats, OptError> {
    let m_total =
        validate_polymatrix_inputs(action_counts, payoff_matrix, cfg, avg_strategies_out.len())?;
    let req = nash_polymatrix_workspace_len(m_total);
    if ws.len() < req {
        return Err(OptError::WorkspaceTooSmall {
            required: req,
            available: ws.len(),
        });
    }
    let (curr_strat, mut r1) = ws.try_split(m_total)?;
    let (cum_regret, mut r2) = r1.try_split(m_total)?;
    let (cum_strat, mut r3) = r2.try_split(m_total)?;
    let (action_utils, _) = r3.try_split(m_total)?;

    init_uniform_profile(action_counts, curr_strat);
    cum_regret.fill(0.0);
    cum_strat.fill(0.0);

    let mut stats = PolymatrixNashStats {
        iterations: 0,
        exploitability: f64::INFINITY,
        converged: false,
    };

    for iter in 0..cfg.max_iterations {
        let weight = if cfg.cfr_plus { (iter + 1) as f64 } else { 1.0 };
        evaluate_action_utilities(action_counts, payoff_matrix, curr_strat, action_utils);
        update_regrets_and_strategies(
            action_counts,
            cfg.cfr_plus,
            weight,
            action_utils,
            curr_strat,
            cum_regret,
            cum_strat,
        );

        stats.iterations = iter + 1;
        if (iter + 1) % 16 == 0 || iter + 1 == cfg.max_iterations {
            extract_average_profile(action_counts, cum_strat, avg_strategies_out);
            stats.exploitability = compute_polymatrix_exploitability(
                action_counts,
                payoff_matrix,
                avg_strategies_out,
                action_utils,
            );
            if stats.exploitability <= cfg.tolerance {
                stats.converged = true;
                break;
            }
        }
    }
    Ok(stats)
}

#[verified_engine::verified]
fn validate_polymatrix_inputs(
    action_counts: &[usize],
    payoff_matrix: &[f64],
    cfg: &PolymatrixConfig,
    out_len: usize,
) -> Result<usize, OptError> {
    if action_counts.len() < 2 || cfg.max_iterations == 0 {
        return Err(OptError::InvalidDimension);
    }
    let mut total = 0usize;
    for &c in action_counts {
        if c == 0 {
            return Err(OptError::InvalidDimension);
        }
        total = total.checked_add(c).ok_or(OptError::InvalidDimension)?;
    }
    let mat_len = total.checked_mul(total).ok_or(OptError::InvalidDimension)?;
    if payoff_matrix.len() != mat_len || out_len != total {
        return Err(OptError::InvalidDimension);
    }
    if !cfg.tolerance.is_finite() || payoff_matrix.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput);
    }
    if cfg.tolerance < 0.0 {
        return Err(OptError::InvalidParameter);
    }
    Ok(total)
}

#[verified_engine::verified]
fn init_uniform_profile(action_counts: &[usize], strat: &mut [f64]) {
    let mut off = 0usize;
    for &cnt in action_counts {
        let p = 1.0 / (cnt as f64);
        strat[off..off + cnt].fill(p);
        off += cnt;
    }
}

#[verified_engine::verified]
fn evaluate_action_utilities(
    action_counts: &[usize],
    payoff_matrix: &[f64],
    strat: &[f64],
    action_utils: &mut [f64],
) {
    let m_total = strat.len();
    let mut off_i = 0usize;
    for (p_i, &cnt_i) in action_counts.iter().enumerate() {
        for a in 0..cnt_i {
            let row = (off_i + a) * m_total;
            let mut u = 0.0;
            let mut off_j = 0usize;
            for (p_j, &cnt_j) in action_counts.iter().enumerate() {
                if p_i != p_j {
                    for b in 0..cnt_j {
                        u += payoff_matrix[row + off_j + b] * strat[off_j + b];
                    }
                }
                off_j += cnt_j;
            }
            action_utils[off_i + a] = u;
        }
        off_i += cnt_i;
    }
}

#[verified_engine::verified]
fn update_regrets_and_strategies(
    action_counts: &[usize],
    cfr_plus: bool,
    weight: f64,
    action_utils: &[f64],
    curr_strat: &mut [f64],
    cum_regret: &mut [f64],
    cum_strat: &mut [f64],
) {
    let mut off = 0usize;
    for &cnt in action_counts {
        let mut exp_u = 0.0;
        for a in 0..cnt {
            exp_u += curr_strat[off + a] * action_utils[off + a];
            cum_strat[off + a] += weight * curr_strat[off + a];
        }
        let mut pos_sum = 0.0;
        for a in 0..cnt {
            let mut r = cum_regret[off + a] + (action_utils[off + a] - exp_u);
            if cfr_plus && r < 0.0 {
                r = 0.0;
            }
            cum_regret[off + a] = r;
            pos_sum += r.max(0.0);
        }
        if pos_sum > 1.0e-15 {
            let inv = 1.0 / pos_sum;
            for a in 0..cnt {
                curr_strat[off + a] = cum_regret[off + a].max(0.0) * inv;
            }
        } else {
            let unif = 1.0 / (cnt as f64);
            curr_strat[off..off + cnt].fill(unif);
        }
        off += cnt;
    }
}

#[verified_engine::verified]
fn extract_average_profile(action_counts: &[usize], cum_strat: &[f64], avg_out: &mut [f64]) {
    let mut off = 0usize;
    for &cnt in action_counts {
        let sum: f64 = cum_strat[off..off + cnt].iter().copied().sum();
        if sum > 1.0e-15 {
            let inv = 1.0 / sum;
            for a in 0..cnt {
                avg_out[off + a] = cum_strat[off + a] * inv;
            }
        } else {
            avg_out[off..off + cnt].fill(1.0 / (cnt as f64));
        }
        off += cnt;
    }
}

#[verified_engine::verified]
fn compute_polymatrix_exploitability(
    action_counts: &[usize],
    payoff_matrix: &[f64],
    avg_strat: &[f64],
    action_utils: &mut [f64],
) -> f64 {
    evaluate_action_utilities(action_counts, payoff_matrix, avg_strat, action_utils);
    let mut max_exp = 0.0_f64;
    let mut off = 0usize;
    for &cnt in action_counts {
        let mut best_u = f64::NEG_INFINITY;
        let mut exp_u = 0.0;
        for a in 0..cnt {
            let u = action_utils[off + a];
            best_u = best_u.max(u);
            exp_u += avg_strat[off + a] * u;
        }
        max_exp = max_exp.max((best_u - exp_u).max(0.0));
        off += cnt;
    }
    max_exp
}
