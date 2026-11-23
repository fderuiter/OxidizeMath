//! Busseti, Ryu, and Boyd (2016) martingale drawdown risk-constrained Kelly solvers.
//!
//! Enforces `E[R^(-lambda)] <= 1` with `lambda = ln(beta) / ln(alpha)` (equivalently
//! `E[R^(1 - gamma)] <= 1` for `gamma = 1 + lambda`), bounding the infinite-horizon probability
//! `P(inf_{t >= 1} W_t / W_0 <= alpha) <= beta` across single wagers and simultaneous slates.

use super::core::{BankrollFraction, DrawdownConstraint, MAX_SLATE_WAGERS, Odds, UnitInterval};
use super::criterion::kelly_fraction;
use super::push::{PushWager, push_kelly_fraction};
use super::quarter_line::{QuarterLineWager, quarter_line_kelly_fraction};
use super::slate::{SlateSolution, maximize_penalized_slate};
use super::slate_eval::{
    SlateEvalMode, SlateProblem, SlateScratch, SlateWager, eval_slate_objective, resolve_eval_mode,
};
use crate::error::KellyError;
use crate::pure_math::analysis::roots::{AnalysisError, Bisection, RootFinder};

const DRAWDOWN_ROOT_ITERS: usize = 128;
const DRAWDOWN_ROOT_TOL: f64 = 1e-13;
const DUAL_OUTER_MAX_ITERS: usize = 32;

fn solve_single_drawdown<H>(f_unconstrained: f64, h_fn: H) -> Result<BankrollFraction, KellyError>
where
    H: Fn(f64) -> f64,
{
    let h_star = h_fn(f_unconstrained);
    if h_star <= 0.0 {
        return BankrollFraction::new(f_unconstrained);
    }

    let mut low = 0.5 * f_unconstrained;
    while low > 1e-15 && h_fn(low) >= 0.0 {
        low *= 0.5;
    }
    if h_fn(low) >= 0.0 {
        return BankrollFraction::new(0.0);
    }

    let solver = Bisection::new(DRAWDOWN_ROOT_ITERS, DRAWDOWN_ROOT_TOL);
    let root = match solver.find_root(&h_fn, low, f_unconstrained) {
        Ok(r) => r,
        Err(AnalysisError::ConvergenceError(best)) => best,
        Err(AnalysisError::InvalidParameters(_)) => {
            return Err(KellyError::ConvergenceFailed {
                iterations: DRAWDOWN_ROOT_ITERS,
                residual: h_fn(low).abs(),
            });
        }
    };
    BankrollFraction::new(root.clamp(0.0, f_unconstrained))
}

/// Computes the Busseti-Ryu-Boyd drawdown-constrained Kelly fraction for a binary wager.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{
///     DrawdownConstraint, Odds, UnitInterval, drawdown_constrained_kelly,
/// };
///
/// let p = UnitInterval::new(0.55).unwrap();
/// let odds = Odds::from_net_odds(1.0).unwrap();
/// let dc = DrawdownConstraint::new(0.70, 0.10).unwrap();
/// let f_rc = drawdown_constrained_kelly(&p, &odds, &dc).unwrap();
/// assert!((f_rc.value() - 0.02689974).abs() < 1e-7);
/// ```
#[verified_engine::verified]
pub fn drawdown_constrained_kelly(
    probability: &UnitInterval,
    odds: &Odds,
    constraint: &DrawdownConstraint,
) -> Result<BankrollFraction, KellyError> {
    let f_star = kelly_fraction(probability, odds)?.value();
    let p = probability.value();
    let q = probability.complement();
    let b = odds.net_profit_multiplier();
    let lam = constraint.lambda();

    solve_single_drawdown(f_star, |f| {
        p * (1.0 + b * f).powf(-lam) + q * (1.0 - f).powf(-lam) - 1.0
    })
}

/// Computes the Busseti-Ryu-Boyd drawdown-constrained Kelly fraction for a three-state push wager.
#[verified_engine::verified]
pub fn drawdown_constrained_push_kelly(
    wager: &PushWager,
    constraint: &DrawdownConstraint,
) -> Result<BankrollFraction, KellyError> {
    let f_star = push_kelly_fraction(wager)?.value();
    let b = wager.odds().net_profit_multiplier();
    let lam = constraint.lambda();

    solve_single_drawdown(f_star, |f| {
        wager.p_win() * (1.0 + b * f).powf(-lam)
            + wager.p_push()
            + wager.p_loss() * (1.0 - f).powf(-lam)
            - 1.0
    })
}

/// Computes the Busseti-Ryu-Boyd drawdown-constrained Kelly fraction for a five-state quarter-line.
#[verified_engine::verified]
pub fn drawdown_constrained_quarter_line_kelly(
    wager: &QuarterLineWager,
    constraint: &DrawdownConstraint,
) -> Result<BankrollFraction, KellyError> {
    let f_star = quarter_line_kelly_fraction(wager)?.value();
    let b = wager.odds().net_profit_multiplier();
    let lam = constraint.lambda();

    solve_single_drawdown(f_star, |f| {
        wager.p_full_win() * (1.0 + b * f).powf(-lam)
            + wager.p_half_win() * (1.0 + 0.5 * b * f).powf(-lam)
            + wager.p_push()
            + wager.p_half_loss() * (1.0 - 0.5 * f).powf(-lam)
            + wager.p_full_loss() * (1.0 - f).powf(-lam)
            - 1.0
    })
}

fn build_slate_solution(
    wagers: &[SlateWager],
    stakes: &[f64],
    log_growth: f64,
    drawdown_exp: f64,
    kkt: f64,
    mode: SlateEvalMode,
) -> SlateSolution {
    let k = wagers.len();
    let mut arr = [0.0; MAX_SLATE_WAGERS];
    arr[..k].copy_from_slice(&stakes[..k]);
    let ev: f64 = arr[..k]
        .iter()
        .zip(wagers.iter())
        .map(|(&f_i, w_i)| f_i * w_i.expected_value())
        .sum();
    SlateSolution::from_parts(k, arr, ev, log_growth, drawdown_exp, kkt, mode)
}

fn run_illinois_dual_search(
    wagers: &[SlateWager],
    lam: f64,
    mode: SlateEvalMode,
    mut psi_lo: f64,
    mut psi_hi: f64,
    scratch: &mut SlateScratch,
) -> f64 {
    let k = wagers.len();
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    let mut side = 0i8;
    let mut best_kappa = 0.5;

    for _ in 0..DUAL_OUTER_MAX_ITERS {
        let sec = lo - psi_lo * (hi - lo) / (psi_hi - psi_lo);
        let margin = 0.05 * (hi - lo);
        let mid = sec.clamp(lo + margin, hi - margin);
        best_kappa = mid;
        let tol_in = (0.01 * (hi - lo)).clamp(1e-12, 1e-5);
        let prob = SlateProblem {
            wagers,
            lambda: lam,
            kappa: mid,
            mode,
        };
        let (eval_m, _) = maximize_penalized_slate(prob, tol_in, true, scratch);
        scratch.warm_stakes[..k].copy_from_slice(&scratch.x_curr[..k]);
        if (eval_m.drawdown_exp - 1.0).abs() < 1e-11 || (hi - lo) < 1e-13 {
            break;
        }
        let psi_m = eval_m.drawdown_exp.ln();
        if psi_m > 0.0 {
            lo = mid;
            psi_lo = psi_m;
            if side == -1 {
                psi_hi *= 0.5;
            }
            side = -1;
        } else {
            hi = mid;
            psi_hi = psi_m;
            if side == 1 {
                psi_lo *= 0.5;
            }
            side = 1;
        }
    }
    best_kappa
}

/// Solves a simultaneous `K`-wager slate subject to `E[R^(-lambda)] <= 1` (`SlateEvalMode::Auto`).
#[verified_engine::verified]
pub fn solve_drawdown_constrained_slate(
    wagers: &[SlateWager],
    constraint: &DrawdownConstraint,
    scratch: &mut SlateScratch,
) -> Result<SlateSolution, KellyError> {
    solve_drawdown_constrained_slate_with_mode(wagers, constraint, SlateEvalMode::Auto, scratch)
}

/// Solves a simultaneous `K`-wager slate subject to `E[R^(-lambda)] <= 1` with explicit `mode`.
#[verified_engine::verified]
pub fn solve_drawdown_constrained_slate_with_mode(
    wagers: &[SlateWager],
    constraint: &DrawdownConstraint,
    mode: SlateEvalMode,
    scratch: &mut SlateScratch,
) -> Result<SlateSolution, KellyError> {
    let (resolved_mode, _) = resolve_eval_mode(wagers, mode)?;
    let k = wagers.len();
    let lam = constraint.lambda();

    let prob_0 = SlateProblem {
        wagers,
        lambda: lam,
        kappa: 0.0,
        mode: resolved_mode,
    };
    let (_, kkt_unc) = maximize_penalized_slate(prob_0, 1e-11, false, scratch);
    let mut unc_stakes = [0.0; MAX_SLATE_WAGERS];
    let mut g_tmp = [0.0; MAX_SLATE_WAGERS];
    unc_stakes[..k].copy_from_slice(&scratch.x_curr[..k]);
    let eval_0 = eval_slate_objective(prob_0, &unc_stakes[..k], scratch, &mut g_tmp);
    if eval_0.drawdown_exp <= 1.0 + 1e-12 {
        return Ok(build_slate_solution(
            wagers,
            &unc_stakes[..k],
            eval_0.log_growth,
            eval_0.drawdown_exp,
            kkt_unc,
            resolved_mode,
        ));
    }

    let prob_1 = SlateProblem {
        wagers,
        lambda: lam,
        kappa: 1.0,
        mode: resolved_mode,
    };
    let (eval_1, kkt_1) = maximize_penalized_slate(prob_1, 1e-10, false, scratch);
    scratch.warm_stakes[..k].copy_from_slice(&scratch.x_curr[..k]);
    if eval_1.drawdown_exp >= 1.0 - 1e-12 {
        let stakes_1 = scratch.warm_stakes;
        return Ok(build_slate_solution(
            wagers,
            &stakes_1[..k],
            eval_1.log_growth,
            eval_1.drawdown_exp,
            kkt_1,
            resolved_mode,
        ));
    }

    let best_kappa = run_illinois_dual_search(
        wagers,
        lam,
        resolved_mode,
        eval_0.drawdown_exp.ln(),
        eval_1.drawdown_exp.ln(),
        scratch,
    );
    let prob_final = SlateProblem {
        wagers,
        lambda: lam,
        kappa: best_kappa,
        mode: resolved_mode,
    };
    let (eval_final, kkt_final) = maximize_penalized_slate(prob_final, 1e-12, true, scratch);
    let mut final_stakes = [0.0; MAX_SLATE_WAGERS];
    final_stakes[..k].copy_from_slice(&scratch.x_curr[..k]);
    Ok(build_slate_solution(
        wagers,
        &final_stakes[..k],
        eval_final.log_growth,
        eval_final.drawdown_exp,
        kkt_final,
        resolved_mode,
    ))
}
