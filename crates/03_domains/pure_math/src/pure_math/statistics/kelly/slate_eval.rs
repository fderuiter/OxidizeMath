//! Exact state convolution (`K <= 10`) and `O(Q * K)` 1D Laplace quadrature (`K > 10`)
//! evaluators for simultaneous independent Kelly slates.

use super::core::{
    EXACT_CONVOLUTION_MAX_WAGERS, LAPLACE_QUADRATURE_NODES, MAX_SLATE_WAGERS, Odds, UnitInterval,
};
use super::push::PushWager;
use super::quarter_line::QuarterLineWager;
use crate::error::KellyError;

/// Maximum product state capacity for exact discrete convolution (`2^10 = 1024`).
pub const MAX_CONVOLUTION_STATES: usize = 1 << EXACT_CONVOLUTION_MAX_WAGERS;

const LAPLACE_S_MIN: f64 = -32.0;
const LAPLACE_S_MAX: f64 = 18.0;

/// Evaluation backend used for simultaneous independent Kelly slates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlateEvalMode {
    /// Uses exact state convolution when `K <= 10` and total states `<= 1024`, else Laplace quadrature.
    #[default]
    Auto,
    /// Forces exact discrete state-space convolution (`<= 1024` product states).
    ExactConvolution,
    /// Forces 1D Frullani / Gamma Laplace quadrature factorization in `O(Q * K)` operations.
    LaplaceQuadrature,
}

/// Independent discrete wager inside a simultaneous `K`-game slate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlateWager {
    state_count: usize,
    probs: [f64; 5],
    net_returns: [f64; 5],
}

impl SlateWager {
    /// Constructs a two-state binary slate wager `(win: +b, loss: -1)`.
    #[verified_engine::verified]
    pub fn binary(probability: UnitInterval, odds: Odds) -> Self {
        let p = probability.value();
        let q = probability.complement();
        let b = odds.net_profit_multiplier();
        Self {
            state_count: 2,
            probs: [p, q, 0.0, 0.0, 0.0],
            net_returns: [b, -1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Constructs a three-state pushable slate wager `(win: +b, push: 0, loss: -1)`.
    #[verified_engine::verified]
    pub fn from_push(wager: PushWager) -> Self {
        let b = wager.odds().net_profit_multiplier();
        Self {
            state_count: 3,
            probs: [wager.p_win(), wager.p_push(), wager.p_loss(), 0.0, 0.0],
            net_returns: [b, 0.0, -1.0, 0.0, 0.0],
        }
    }

    /// Constructs a five-state Asian quarter-line slate wager.
    #[verified_engine::verified]
    pub fn from_quarter_line(wager: QuarterLineWager) -> Self {
        let b = wager.odds().net_profit_multiplier();
        Self {
            state_count: 5,
            probs: [
                wager.p_full_win(),
                wager.p_half_win(),
                wager.p_push(),
                wager.p_half_loss(),
                wager.p_full_loss(),
            ],
            net_returns: [b, 0.5 * b, 0.0, -0.5, -1.0],
        }
    }

    /// Returns the number of discrete settlement states (`2`, `3`, or `5`).
    #[verified_engine::verified]
    pub fn state_count(&self) -> usize {
        self.state_count
    }

    /// Returns unit expected net return `E[X_i]`.
    #[verified_engine::verified]
    pub fn expected_value(&self) -> f64 {
        let mut ev = 0.0;
        for m in 0..self.state_count {
            ev += self.probs[m] * self.net_returns[m];
        }
        ev
    }

    /// Samples the realized net return multiplier `X_i` from a uniform `[0, 1)` variate `u`.
    #[verified_engine::verified]
    pub fn sample_net_return(&self, u: f64) -> f64 {
        let mut cum = 0.0;
        for m in 0..self.state_count.saturating_sub(1) {
            cum += self.probs[m];
            if u < cum {
                return self.net_returns[m];
            }
        }
        self.net_returns[self.state_count - 1]
    }
}

/// Caller-owned preallocated scratch buffers for zero-allocation slate optimization.
#[derive(Clone)]
pub struct SlateScratch {
    pub(super) probs: [f64; MAX_CONVOLUTION_STATES],
    pub(super) wealths: [f64; MAX_CONVOLUTION_STATES],
    pub(super) weights: [f64; MAX_CONVOLUTION_STATES],
    pub(super) x_curr: [f64; MAX_SLATE_WAGERS],
    pub(super) y_curr: [f64; MAX_SLATE_WAGERS],
    pub(super) x_cand: [f64; MAX_SLATE_WAGERS],
    pub(super) grad_curr: [f64; MAX_SLATE_WAGERS],
    pub(super) grad_cand: [f64; MAX_SLATE_WAGERS],
    pub(super) sort_buf: [f64; MAX_SLATE_WAGERS],
    pub(super) prefix: [f64; MAX_SLATE_WAGERS + 1],
    pub(super) phis: [f64; MAX_SLATE_WAGERS],
    pub(super) mus: [f64; MAX_SLATE_WAGERS],
    pub(super) warm_stakes: [f64; MAX_SLATE_WAGERS],
}

impl Default for SlateScratch {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl SlateScratch {
    /// Creates a zero-initialized caller-owned workspace for simultaneous slate optimization.
    #[verified_engine::verified]
    pub const fn new() -> Self {
        Self {
            probs: [0.0; MAX_CONVOLUTION_STATES],
            wealths: [0.0; MAX_CONVOLUTION_STATES],
            weights: [0.0; MAX_CONVOLUTION_STATES],
            x_curr: [0.0; MAX_SLATE_WAGERS],
            y_curr: [0.0; MAX_SLATE_WAGERS],
            x_cand: [0.0; MAX_SLATE_WAGERS],
            grad_curr: [0.0; MAX_SLATE_WAGERS],
            grad_cand: [0.0; MAX_SLATE_WAGERS],
            sort_buf: [0.0; MAX_SLATE_WAGERS],
            prefix: [0.0; MAX_SLATE_WAGERS + 1],
            phis: [0.0; MAX_SLATE_WAGERS],
            mus: [0.0; MAX_SLATE_WAGERS],
            warm_stakes: [0.0; MAX_SLATE_WAGERS],
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SlateProblem<'a> {
    pub wagers: &'a [SlateWager],
    pub lambda: f64,
    pub kappa: f64,
    pub mode: SlateEvalMode,
}

pub(super) struct SlateEvalOutput {
    pub obj: f64,
    pub log_growth: f64,
    pub drawdown_exp: f64,
}

fn lanczos_ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        let sin_pi_z = (std::f64::consts::PI * z).sin();
        return std::f64::consts::PI.ln() - sin_pi_z.ln() - lanczos_ln_gamma(1.0 - z);
    }
    let x = z - 1.0;
    let mut sum = COEFFS[0];
    for (idx, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (x + idx as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

pub(super) fn resolve_eval_mode(
    wagers: &[SlateWager],
    requested: SlateEvalMode,
) -> Result<(SlateEvalMode, usize), KellyError> {
    let k = wagers.len();
    if k == 0 {
        return Err(KellyError::EmptyPortfolio);
    }
    if k > MAX_SLATE_WAGERS {
        return Err(KellyError::PortfolioTooLarge {
            count: k,
            max: MAX_SLATE_WAGERS,
        });
    }
    let product_states = wagers
        .iter()
        .try_fold(1usize, |acc, w| acc.checked_mul(w.state_count));

    match requested {
        SlateEvalMode::Auto => {
            if k <= EXACT_CONVOLUTION_MAX_WAGERS
                && let Some(states) = product_states
                && states <= MAX_CONVOLUTION_STATES
            {
                Ok((SlateEvalMode::ExactConvolution, states))
            } else {
                Ok((SlateEvalMode::LaplaceQuadrature, 0))
            }
        }
        SlateEvalMode::ExactConvolution => match product_states {
            Some(states) if states <= MAX_CONVOLUTION_STATES => {
                Ok((SlateEvalMode::ExactConvolution, states))
            }
            _ => Err(KellyError::PortfolioTooLarge {
                count: product_states.unwrap_or(usize::MAX),
                max: MAX_CONVOLUTION_STATES,
            }),
        },
        SlateEvalMode::LaplaceQuadrature => Ok((SlateEvalMode::LaplaceQuadrature, 0)),
    }
}

pub(super) fn eval_slate_objective(
    prob: SlateProblem<'_>,
    stakes: &[f64],
    scratch: &mut SlateScratch,
    grad_out: &mut [f64],
) -> SlateEvalOutput {
    match prob.mode {
        SlateEvalMode::ExactConvolution | SlateEvalMode::Auto => {
            eval_exact_convolution(prob, stakes, scratch, grad_out)
        }
        SlateEvalMode::LaplaceQuadrature => {
            eval_laplace_quadrature(prob, stakes, scratch, grad_out)
        }
    }
}

fn expand_convolution_states(
    wagers: &[SlateWager],
    stakes: &[f64],
    scratch: &mut SlateScratch,
) -> usize {
    let mut len = 1usize;
    scratch.probs[0] = 1.0;
    scratch.wealths[0] = 1.0;
    for (wager, &f_i) in wagers.iter().zip(stakes.iter()) {
        let m_cnt = wager.state_count;
        for j in (0..len).rev() {
            let pr = scratch.probs[j];
            let w = scratch.wealths[j];
            let base = j * m_cnt;
            for m in 0..m_cnt {
                scratch.probs[base + m] = pr * wager.probs[m];
                scratch.wealths[base + m] = w + f_i * wager.net_returns[m];
            }
        }
        len *= m_cnt;
    }
    len
}

fn reduce_convolution_gradient(
    wagers: &[SlateWager],
    len: usize,
    weights: &[f64; MAX_CONVOLUTION_STATES],
    grad_out: &mut [f64],
) {
    let mut stride = len;
    for (i, wager) in wagers.iter().enumerate() {
        let m_cnt = wager.state_count;
        stride /= m_cnt;
        let block = stride * m_cnt;
        let mut buckets = [0.0; 5];
        let mut base = 0;
        while base < len {
            for (m, bucket) in buckets.iter_mut().enumerate().take(m_cnt) {
                let start = base + m * stride;
                for &w_val in &weights[start..(start + stride)] {
                    *bucket += w_val;
                }
            }
            base += block;
        }
        let mut g_i = 0.0;
        for (m, &bucket) in buckets.iter().enumerate().take(m_cnt) {
            g_i += wager.net_returns[m] * bucket;
        }
        grad_out[i] = g_i;
    }
}

fn eval_exact_convolution(
    prob: SlateProblem<'_>,
    stakes: &[f64],
    scratch: &mut SlateScratch,
    grad_out: &mut [f64],
) -> SlateEvalOutput {
    let k = prob.wagers.len();
    let len = expand_convolution_states(prob.wagers, stakes, scratch);
    let mut log_growth = 0.0;
    let mut drawdown_exp = 0.0;

    for idx in 0..len {
        let pr = scratch.probs[idx];
        let w = scratch.wealths[idx];
        if pr > 0.0 && w <= 1e-14 {
            grad_out[..k].fill(0.0);
            return SlateEvalOutput {
                obj: f64::NEG_INFINITY,
                log_growth: f64::NEG_INFINITY,
                drawdown_exp: f64::INFINITY,
            };
        }
        if pr == 0.0 {
            scratch.weights[idx] = 0.0;
            continue;
        }
        let ln_w = w.ln();
        log_growth += pr * ln_w;
        if prob.lambda > 0.0 {
            let w_neg_lam = (-prob.lambda * ln_w).exp();
            drawdown_exp += pr * w_neg_lam;
            let num = if prob.kappa > 0.0 {
                (1.0 - prob.kappa) + prob.kappa * w_neg_lam
            } else {
                1.0
            };
            scratch.weights[idx] = pr * num / w;
        } else {
            scratch.weights[idx] = pr / w;
        }
    }

    reduce_convolution_gradient(prob.wagers, len, &scratch.weights, grad_out);
    finalize_slate_output(log_growth, drawdown_exp, prob.lambda, prob.kappa)
}

fn evaluate_laplace_node(
    wagers: &[SlateWager],
    stakes: &[f64],
    t: f64,
    f0: f64,
    scratch: &mut SlateScratch,
) -> (f64, f64) {
    scratch.prefix[0] = (-t * f0).exp();
    let mut delta = (-t * f0).exp_m1();
    for (i, (wager, &f_i)) in wagers.iter().zip(stakes.iter()).enumerate() {
        let mut phi = 0.0;
        let mut phi_m1 = 0.0;
        let mut mu = 0.0;
        for m in 0..wager.state_count {
            let p_m = wager.probs[m];
            let x_m = wager.net_returns[m];
            let arg = -t * f_i * (1.0 + x_m);
            let ew = arg.exp();
            phi += p_m * ew;
            phi_m1 += p_m * arg.exp_m1();
            mu += p_m * x_m * ew;
        }
        scratch.phis[i] = phi;
        scratch.mus[i] = mu;
        scratch.prefix[i + 1] = scratch.prefix[i] * phi;
        delta = delta + phi_m1 + delta * phi_m1;
    }
    (scratch.prefix[wagers.len()], delta)
}

fn eval_laplace_quadrature(
    prob: SlateProblem<'_>,
    stakes: &[f64],
    scratch: &mut SlateScratch,
    grad_out: &mut [f64],
) -> SlateEvalOutput {
    let k = prob.wagers.len();
    let f0 = 1.0 - stakes[..k].iter().sum::<f64>();
    if f0 <= 1e-14 {
        grad_out[..k].fill(0.0);
        return SlateEvalOutput {
            obj: f64::NEG_INFINITY,
            log_growth: f64::NEG_INFINITY,
            drawdown_exp: f64::INFINITY,
        };
    }

    let ds = (LAPLACE_S_MAX - LAPLACE_S_MIN) / (LAPLACE_QUADRATURE_NODES as f64);
    let ln_gamma_lam = if prob.lambda > 0.0 {
        lanczos_ln_gamma(prob.lambda)
    } else {
        0.0
    };
    let mut log_growth = 0.0;
    let mut drawdown_exp = 0.0;
    grad_out[..k].fill(0.0);

    for q in 0..LAPLACE_QUADRATURE_NODES {
        let s = LAPLACE_S_MIN + (q as f64 + 0.5) * ds;
        let t = s.exp();
        let (prod_all, delta) = evaluate_laplace_node(prob.wagers, stakes, t, f0, scratch);
        log_growth += ((-t).exp_m1() - delta) * ds;

        let grad_weight = if prob.lambda > 0.0 {
            let gamma_kernel = (prob.lambda * s - ln_gamma_lam).exp();
            drawdown_exp += gamma_kernel * prod_all * ds;
            if prob.kappa > 0.0 {
                t * ds * ((1.0 - prob.kappa) + prob.kappa * gamma_kernel)
            } else {
                t * ds
            }
        } else {
            t * ds
        };

        let mut suffix = 1.0;
        for i in (0..k).rev() {
            grad_out[i] += grad_weight * (scratch.prefix[i] * suffix) * scratch.mus[i];
            suffix *= scratch.phis[i];
        }
    }

    finalize_slate_output(log_growth, drawdown_exp, prob.lambda, prob.kappa)
}

fn finalize_slate_output(
    log_growth: f64,
    drawdown_exp: f64,
    lambda: f64,
    kappa: f64,
) -> SlateEvalOutput {
    let obj = if lambda > 0.0 && kappa > 0.0 {
        (1.0 - kappa) * log_growth + kappa * (1.0 - drawdown_exp) / lambda
    } else {
        log_growth
    };
    SlateEvalOutput {
        obj,
        log_growth,
        drawdown_exp: if lambda > 0.0 { drawdown_exp } else { 1.0 },
    }
}
