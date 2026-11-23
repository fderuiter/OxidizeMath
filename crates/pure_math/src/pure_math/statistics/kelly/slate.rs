//! Simplex convex optimization for simultaneous independent Kelly slates (`sum f_i <= 1`).
//!
//! Solves `max_{f >= 0, sum f_i <= 1} E[ln(1 + sum_i f_i X_i)]` using spectral Barzilai-Borwein
//! and gradient-restarted FISTA projections onto the unit probability simplex inside caller-owned
//! [`SlateScratch`] buffers with zero dynamic heap allocations.

use super::core::MAX_SLATE_WAGERS;
use super::slate_eval::{
    SlateEvalMode, SlateEvalOutput, SlateProblem, SlateScratch, SlateWager, eval_slate_objective,
    resolve_eval_mode,
};
use crate::error::KellyError;

const SIMPLEX_RUIN_CAP: f64 = 1.0 - 1e-12;
const SLATE_MAX_ITERS: usize = 200;
const SLATE_KKT_TOL: f64 = 1e-11;

/// Joint optimal stake vector and diagnostic metrics for a simultaneous `K`-wager slate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlateSolution {
    len: usize,
    stakes: [f64; MAX_SLATE_WAGERS],
    total_fraction: f64,
    expected_value: f64,
    expected_growth_rate: f64,
    drawdown_expectation: f64,
    kkt_residual: f64,
    eval_mode_used: SlateEvalMode,
}

impl SlateSolution {
    pub(super) fn from_parts(
        len: usize,
        stakes: [f64; MAX_SLATE_WAGERS],
        expected_value: f64,
        expected_growth_rate: f64,
        drawdown_expectation: f64,
        kkt_residual: f64,
        eval_mode_used: SlateEvalMode,
    ) -> Self {
        let total_fraction = stakes[..len].iter().sum();
        Self {
            len,
            stakes,
            total_fraction,
            expected_value,
            expected_growth_rate,
            drawdown_expectation,
            kkt_residual,
            eval_mode_used,
        }
    }

    /// Returns the joint optimal stake slice `&[f64]` for all `K` wagers.
    #[verified_engine::verified]
    pub fn stakes(&self) -> &[f64] {
        &self.stakes[..self.len]
    }

    /// Returns the total fraction of bankroll allocated across the slate (`sum_i f_i <= 1`).
    #[verified_engine::verified]
    pub fn total_fraction(&self) -> f64 {
        self.total_fraction
    }

    /// Returns the expected net return `E[R - 1]` of the joint slate portfolio.
    #[verified_engine::verified]
    pub fn expected_value(&self) -> f64 {
        self.expected_value
    }

    /// Returns the joint expected logarithmic growth rate `E[ln R]`.
    #[verified_engine::verified]
    pub fn expected_growth_rate(&self) -> f64 {
        self.expected_growth_rate
    }

    /// Returns `E[R^(-lambda)]` when evaluated under a drawdown constraint (or `1.0` unconstrained).
    #[verified_engine::verified]
    pub fn drawdown_expectation(&self) -> f64 {
        self.drawdown_expectation
    }

    /// Returns the infinity-norm KKT projected-gradient optimality residual.
    #[verified_engine::verified]
    pub fn kkt_residual(&self) -> f64 {
        self.kkt_residual
    }

    /// Returns the concrete evaluation backend (`ExactConvolution` or `LaplaceQuadrature`) used.
    #[verified_engine::verified]
    pub fn eval_mode_used(&self) -> SlateEvalMode {
        self.eval_mode_used
    }
}

/// Projects `src[..k]` in Euclidean norm onto `{f >= 0 : sum_i f_i <= cap}` using `sort_buf`.
pub(super) fn project_simplex_capped(
    src: &[f64],
    k: usize,
    cap: f64,
    sort_buf: &mut [f64; MAX_SLATE_WAGERS],
    dst: &mut [f64],
) {
    let mut pos_sum = 0.0;
    for i in 0..k {
        let clamped = src[i].max(0.0);
        dst[i] = clamped;
        pos_sum += clamped;
    }
    if pos_sum <= cap {
        return;
    }

    sort_buf[..k].copy_from_slice(&src[..k]);
    for i in 1..k {
        let key = sort_buf[i];
        let mut j = i;
        while j > 0 && sort_buf[j - 1] < key {
            sort_buf[j] = sort_buf[j - 1];
            j -= 1;
        }
        sort_buf[j] = key;
    }

    let mut cssv = 0.0;
    let mut rho = 0usize;
    for (i, &val) in sort_buf.iter().enumerate().take(k) {
        cssv += val;
        if val - (cssv - cap) / ((i + 1) as f64) > 0.0 {
            rho = i;
        }
    }
    let prefix_sum: f64 = sort_buf[..=rho].iter().sum();
    let theta = (prefix_sum - cap) / ((rho + 1) as f64);
    for i in 0..k {
        dst[i] = (src[i] - theta).max(0.0);
    }
}

pub(super) fn simplex_kkt_residual(
    x: &[f64],
    grad: &[f64],
    k: usize,
    cap: f64,
    sort_buf: &mut [f64; MAX_SLATE_WAGERS],
) -> f64 {
    let mut trial = [0.0; MAX_SLATE_WAGERS];
    let mut projected = [0.0; MAX_SLATE_WAGERS];
    for i in 0..k {
        trial[i] = x[i] + grad[i];
    }
    project_simplex_capped(&trial, k, cap, sort_buf, &mut projected);
    let mut max_diff = 0.0_f64;
    for i in 0..k {
        max_diff = max_diff.max((projected[i] - x[i]).abs());
    }
    max_diff
}

fn backtrack_armijo_step(
    prob: SlateProblem<'_>,
    obj_y: f64,
    alpha_init: f64,
    scratch: &mut SlateScratch,
) -> (f64, SlateEvalOutput) {
    let k = prob.wagers.len();
    let mut step = alpha_init;
    let mut trial = [0.0; MAX_SLATE_WAGERS];
    loop {
        for (i, slot) in trial.iter_mut().enumerate().take(k) {
            *slot = scratch.y_curr[i] + step * scratch.grad_curr[i];
        }
        let mut cand = [0.0; MAX_SLATE_WAGERS];
        project_simplex_capped(
            &trial,
            k,
            SIMPLEX_RUIN_CAP,
            &mut scratch.sort_buf,
            &mut cand,
        );
        scratch.x_cand[..k].copy_from_slice(&cand[..k]);

        let mut dir_deriv = 0.0;
        let mut norm_sq = 0.0;
        for i in 0..k {
            let d = scratch.x_cand[i] - scratch.y_curr[i];
            dir_deriv += scratch.grad_curr[i] * d;
            norm_sq += d * d;
        }
        let mut g_cand = [0.0; MAX_SLATE_WAGERS];
        let eval_c = eval_slate_objective(prob, &cand[..k], scratch, &mut g_cand);
        scratch.grad_cand[..k].copy_from_slice(&g_cand[..k]);

        let sufficient = obj_y + 1e-4 * dir_deriv.max(0.5 * norm_sq / step) - 1e-15;
        if eval_c.obj.is_finite() && (eval_c.obj >= sufficient || step < 1e-16) {
            return (step, eval_c);
        }
        step *= 0.5;
    }
}

fn advance_fista_momentum(
    prob: SlateProblem<'_>,
    t_fista: f64,
    restart_dot: f64,
    eval_c: SlateEvalOutput,
    scratch: &mut SlateScratch,
) -> (f64, SlateEvalOutput) {
    let k = prob.wagers.len();
    if restart_dot < 0.0 {
        scratch.y_curr[..k].copy_from_slice(&scratch.x_cand[..k]);
        scratch.x_curr[..k].copy_from_slice(&scratch.x_cand[..k]);
        scratch.grad_curr[..k].copy_from_slice(&scratch.grad_cand[..k]);
        return (1.0, eval_c);
    }

    let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_fista * t_fista).sqrt());
    let beta_m = (t_fista - 1.0) / t_next;
    let mut extrap = [0.0; MAX_SLATE_WAGERS];
    for (i, slot) in extrap.iter_mut().enumerate().take(k) {
        *slot = scratch.x_cand[i] + beta_m * (scratch.x_cand[i] - scratch.x_curr[i]);
    }
    let mut y_next = [0.0; MAX_SLATE_WAGERS];
    project_simplex_capped(
        &extrap,
        k,
        SIMPLEX_RUIN_CAP,
        &mut scratch.sort_buf,
        &mut y_next,
    );
    let mut g_buf = [0.0; MAX_SLATE_WAGERS];
    let eval_yn = eval_slate_objective(prob, &y_next[..k], scratch, &mut g_buf);
    scratch.x_curr[..k].copy_from_slice(&scratch.x_cand[..k]);

    if eval_yn.obj.is_finite() && eval_yn.obj >= eval_c.obj {
        scratch.y_curr[..k].copy_from_slice(&y_next[..k]);
        scratch.grad_curr[..k].copy_from_slice(&g_buf[..k]);
        (t_next, eval_yn)
    } else {
        scratch.y_curr[..k].copy_from_slice(&scratch.x_cand[..k]);
        scratch.grad_curr[..k].copy_from_slice(&scratch.grad_cand[..k]);
        (1.0, eval_c)
    }
}

/// Inner convex simplex maximizer using Barzilai-Borwein + O'Donoghue-Candès restarted FISTA.
pub(super) fn maximize_penalized_slate(
    prob: SlateProblem<'_>,
    tol: f64,
    use_warm_start: bool,
    scratch: &mut SlateScratch,
) -> (SlateEvalOutput, f64) {
    let k = prob.wagers.len();
    if use_warm_start {
        scratch.x_curr[..k].copy_from_slice(&scratch.warm_stakes[..k]);
    } else {
        scratch.x_curr[..k].fill(0.0);
    }
    scratch.y_curr[..k].copy_from_slice(&scratch.x_curr[..k]);

    let mut y_copy = [0.0; MAX_SLATE_WAGERS];
    let mut g_buf = [0.0; MAX_SLATE_WAGERS];
    y_copy[..k].copy_from_slice(&scratch.y_curr[..k]);
    let mut eval_y = eval_slate_objective(prob, &y_copy[..k], scratch, &mut g_buf);
    scratch.grad_curr[..k].copy_from_slice(&g_buf[..k]);

    let mut t_fista = 1.0_f64;
    let mut alpha = 0.5_f64;

    for _ in 0..SLATE_MAX_ITERS {
        let kkt = simplex_kkt_residual(
            &scratch.y_curr,
            &scratch.grad_curr,
            k,
            SIMPLEX_RUIN_CAP,
            &mut scratch.sort_buf,
        );
        if kkt < tol {
            scratch.x_curr[..k].copy_from_slice(&scratch.y_curr[..k]);
            return (eval_y, kkt);
        }

        let (used_step, eval_c) = backtrack_armijo_step(prob, eval_y.obj, alpha, scratch);
        let (mut s_dot_s, mut s_dot_y, mut restart_dot) = (0.0, 0.0, 0.0);
        for i in 0..k {
            let dy = scratch.x_cand[i] - scratch.y_curr[i];
            let dx = scratch.x_cand[i] - scratch.x_curr[i];
            let dg = scratch.grad_curr[i] - scratch.grad_cand[i];
            s_dot_s += dy * dy;
            s_dot_y += dy * dg;
            restart_dot += scratch.grad_curr[i] * dx;
        }
        alpha = if s_dot_y > 1e-16 {
            (s_dot_s / s_dot_y).clamp(1e-6, 10.0)
        } else {
            (used_step * 2.0).min(10.0)
        };
        (t_fista, eval_y) = advance_fista_momentum(prob, t_fista, restart_dot, eval_c, scratch);
    }

    y_copy[..k].copy_from_slice(&scratch.x_curr[..k]);
    let eval_f = eval_slate_objective(prob, &y_copy[..k], scratch, &mut g_buf);
    scratch.grad_curr[..k].copy_from_slice(&g_buf[..k]);
    let kkt = simplex_kkt_residual(&y_copy, &g_buf, k, SIMPLEX_RUIN_CAP, &mut scratch.sort_buf);
    (eval_f, kkt)
}

/// Maximizes joint expected log-wealth across `K` simultaneous independent wagers (`SlateEvalMode::Auto`).
///
/// Uses exact state convolution for `K <= 10` binary wagers and `O(Q * K)` 1D Laplace quadrature
/// for `K > 10` wagers, with zero heap allocations inside `scratch`.
#[verified_engine::verified]
pub fn solve_simultaneous_slate(
    wagers: &[SlateWager],
    scratch: &mut SlateScratch,
) -> Result<SlateSolution, KellyError> {
    solve_simultaneous_slate_with_mode(wagers, SlateEvalMode::Auto, scratch)
}

/// Maximizes joint expected log-wealth across `K` simultaneous independent wagers with explicit `mode`.
#[verified_engine::verified]
pub fn solve_simultaneous_slate_with_mode(
    wagers: &[SlateWager],
    mode: SlateEvalMode,
    scratch: &mut SlateScratch,
) -> Result<SlateSolution, KellyError> {
    let (resolved_mode, _) = resolve_eval_mode(wagers, mode)?;
    let k = wagers.len();
    let prob = SlateProblem {
        wagers,
        lambda: 0.0,
        kappa: 0.0,
        mode: resolved_mode,
    };
    let (eval, kkt) = maximize_penalized_slate(prob, SLATE_KKT_TOL, false, scratch);
    if kkt > 1e-6 {
        return Err(KellyError::ConvergenceFailed {
            iterations: SLATE_MAX_ITERS,
            residual: kkt,
        });
    }
    let mut stakes = [0.0; MAX_SLATE_WAGERS];
    stakes[..k].copy_from_slice(&scratch.x_curr[..k]);
    let ev: f64 = stakes[..k]
        .iter()
        .zip(wagers.iter())
        .map(|(&f_i, w_i)| f_i * w_i.expected_value())
        .sum();

    Ok(SlateSolution::from_parts(
        k,
        stakes,
        ev,
        eval.log_growth,
        1.0,
        kkt,
        resolved_mode,
    ))
}
