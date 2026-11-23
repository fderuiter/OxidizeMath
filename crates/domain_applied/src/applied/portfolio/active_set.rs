//! Zero-allocation primal-dual active-set Markowitz solver for long-only and weight-capped bounds.

use super::linalg::{cholesky_16, cholesky_solve_16};
use super::{MAX_PORTFOLIO_ASSETS, PortfolioError, PortfolioSpec};

const MAX_ACTIVE_SET_ITERS: usize = 128;
const STEP_TOL: f64 = 1e-12;
pub(super) const KKT_TOL: f64 = 1e-11;
pub(super) const BOUND_TOL: f64 = 1e-12;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BoundState {
    Free,
    Lower,
    Upper,
}

pub(super) fn validate_upper_bounds(
    upper_bounds: &[f64],
    n: usize,
) -> Result<[f64; MAX_PORTFOLIO_ASSETS], PortfolioError> {
    if upper_bounds.len() != n {
        return Err(PortfolioError::DimensionMismatch {
            expected: n,
            actual: upper_bounds.len(),
        });
    }
    let mut ub = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    let mut cap_sum = 0.0_f64;
    for (dst, &u) in ub.iter_mut().zip(upper_bounds.iter()) {
        if !u.is_finite() {
            return Err(PortfolioError::NonFiniteInput);
        }
        if !(0.0..=1.0).contains(&u) {
            return Err(PortfolioError::InvalidWeightBounds);
        }
        *dst = u;
        cap_sum += u;
    }
    if cap_sum < 1.0 - BOUND_TOL {
        return Err(PortfolioError::InvalidWeightBounds);
    }
    Ok(ub)
}

fn solve_free_step(
    cov: &[[f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS],
    grad: &[f64; MAX_PORTFOLIO_ASSETS],
    free_indices: &[usize; MAX_PORTFOLIO_ASSETS],
    m: usize,
) -> Result<([f64; MAX_PORTFOLIO_ASSETS], f64), PortfolioError> {
    let mut p = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    if m == 1 {
        return Ok((p, grad[free_indices[0]]));
    }
    let mut sub_cov = [[0.0_f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS];
    let mut ones = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    let mut g_sub = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for a in 0..m {
        ones[a] = 1.0;
        g_sub[a] = grad[free_indices[a]];
        for b in 0..m {
            sub_cov[a][b] = cov[free_indices[a]][free_indices[b]];
        }
    }
    let l = cholesky_16(&sub_cov, m)?;
    let y = cholesky_solve_16(&l, &ones, m);
    let z = cholesky_solve_16(&l, &g_sub, m);
    let mut sum_y = 0.0_f64;
    let mut sum_z = 0.0_f64;
    for a in 0..m {
        sum_y += y[a];
        sum_z += z[a];
    }
    let nu = sum_z / sum_y;
    for a in 0..m {
        p[free_indices[a]] = nu * y[a] - z[a];
    }
    Ok((p, nu))
}

fn try_release_vertex_pair(
    grad: &[f64; MAX_PORTFOLIO_ASSETS],
    ub: &[f64; MAX_PORTFOLIO_ASSETS],
    states: &mut [BoundState; MAX_PORTFOLIO_ASSETS],
    n: usize,
) -> bool {
    let mut max_upper: Option<(usize, f64)> = None;
    let mut min_lower: Option<(usize, f64)> = None;
    for i in 0..n {
        if ub[i] <= BOUND_TOL {
            continue;
        }
        match states[i] {
            BoundState::Upper => {
                if max_upper.is_none_or(|(_, best)| grad[i] > best) {
                    max_upper = Some((i, grad[i]));
                }
            }
            BoundState::Lower => {
                if min_lower.is_none_or(|(_, best)| grad[i] < best) {
                    min_lower = Some((i, grad[i]));
                }
            }
            BoundState::Free => {}
        }
    }
    if let (Some((u_idx, g_u)), Some((l_idx, g_l))) = (max_upper, min_lower)
        && g_u > g_l + KKT_TOL
    {
        states[u_idx] = BoundState::Free;
        states[l_idx] = BoundState::Free;
        return true;
    }
    false
}

fn advance_blocking_step(
    w: &mut [f64; MAX_PORTFOLIO_ASSETS],
    p: &[f64; MAX_PORTFOLIO_ASSETS],
    ub: &[f64; MAX_PORTFOLIO_ASSETS],
    states: &mut [BoundState; MAX_PORTFOLIO_ASSETS],
    n: usize,
) {
    let mut alpha = 1.0_f64;
    let mut blocker: Option<(usize, BoundState)> = None;
    for i in 0..n {
        if states[i] != BoundState::Free {
            continue;
        }
        if p[i] < -STEP_TOL {
            let cand = (-w[i] / p[i]).max(0.0);
            if cand < alpha {
                alpha = cand;
                blocker = Some((i, BoundState::Lower));
            }
        } else if p[i] > STEP_TOL {
            let cand = ((ub[i] - w[i]) / p[i]).max(0.0);
            if cand < alpha {
                alpha = cand;
                blocker = Some((i, BoundState::Upper));
            }
        }
    }
    for i in 0..n {
        if states[i] == BoundState::Free {
            w[i] = (w[i] + alpha * p[i]).clamp(0.0, ub[i]);
        }
    }
    if let Some((idx, bound)) = blocker {
        w[idx] = if bound == BoundState::Lower {
            0.0
        } else {
            ub[idx]
        };
        states[idx] = bound;
    }
}

fn find_most_violated_bound(
    grad: &[f64; MAX_PORTFOLIO_ASSETS],
    ub: &[f64; MAX_PORTFOLIO_ASSETS],
    states: &[BoundState; MAX_PORTFOLIO_ASSETS],
    nu: f64,
    n: usize,
) -> Option<usize> {
    let mut worst: Option<(usize, f64)> = None;
    for i in 0..n {
        if ub[i] <= BOUND_TOL || states[i] == BoundState::Free {
            continue;
        }
        let lam = if states[i] == BoundState::Lower {
            grad[i] - nu
        } else {
            nu - grad[i]
        };
        if lam < -KKT_TOL && worst.is_none_or(|(_, best)| lam < best) {
            worst = Some((i, lam));
        }
    }
    worst.map(|(idx, _)| idx)
}

pub(super) fn solve_box_simplex_qp(
    spec: &PortfolioSpec,
    linear_c: &[f64; MAX_PORTFOLIO_ASSETS],
    ub: &[f64; MAX_PORTFOLIO_ASSETS],
) -> Result<[f64; MAX_PORTFOLIO_ASSETS], PortfolioError> {
    let n = spec.asset_count;
    let cap_sum: f64 = ub[..n].iter().sum();
    let mut w = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    let mut states = [BoundState::Lower; MAX_PORTFOLIO_ASSETS];
    if cap_sum <= 1.0 + BOUND_TOL {
        w[..n].copy_from_slice(&ub[..n]);
        return Ok(w);
    }
    for i in 0..n {
        if ub[i] > BOUND_TOL {
            w[i] = ub[i] / cap_sum;
            states[i] = BoundState::Free;
        }
    }
    for _iter in 0..MAX_ACTIVE_SET_ITERS {
        let mut grad = [0.0_f64; MAX_PORTFOLIO_ASSETS];
        let mut free_indices = [0_usize; MAX_PORTFOLIO_ASSETS];
        let mut m = 0_usize;
        for i in 0..n {
            grad[i] = linear_c[i] + (0..n).map(|j| spec.covariance[i][j] * w[j]).sum::<f64>();
            if states[i] == BoundState::Free {
                free_indices[m] = i;
                m += 1;
            }
        }
        if m == 0 {
            if !try_release_vertex_pair(&grad, ub, &mut states, n) {
                return Ok(w);
            }
            continue;
        }
        let (p, nu) = solve_free_step(&spec.covariance, &grad, &free_indices, m)?;
        if p[..n].iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())) > STEP_TOL {
            advance_blocking_step(&mut w, &p, ub, &mut states, n);
        } else if let Some(drop_idx) = find_most_violated_bound(&grad, ub, &states, nu, n) {
            states[drop_idx] = BoundState::Free;
        } else {
            return Ok(w);
        }
    }
    Err(PortfolioError::ConvergenceFailed {
        iterations: MAX_ACTIVE_SET_ITERS,
    })
}

fn solve_risk_tolerance(
    spec: &PortfolioSpec,
    gamma: f64,
    ub: &[f64; MAX_PORTFOLIO_ASSETS],
) -> Result<([f64; MAX_PORTFOLIO_ASSETS], f64), PortfolioError> {
    let n = spec.asset_count;
    let mut c = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for (ci, &ret_i) in c[..n].iter_mut().zip(&spec.expected_returns[..n]) {
        *ci = -gamma * ret_i;
    }
    let w = solve_box_simplex_qp(spec, &c, ub)?;
    let mut ret = 0.0_f64;
    for (&wi, &ret_i) in w[..n].iter().zip(&spec.expected_returns[..n]) {
        ret += wi * ret_i;
    }
    Ok((w, ret))
}

pub(super) fn extreme_return_bounds(
    spec: &PortfolioSpec,
    ub: &[f64; MAX_PORTFOLIO_ASSETS],
) -> (f64, f64) {
    let n = spec.asset_count;
    let mut order = [0_usize; MAX_PORTFOLIO_ASSETS];
    for (idx, slot) in order.iter_mut().enumerate().take(n) {
        *slot = idx;
    }
    order[..n]
        .sort_unstable_by(|&a, &b| spec.expected_returns[a].total_cmp(&spec.expected_returns[b]));

    let mut rem_low = 1.0_f64;
    let mut r_min = 0.0_f64;
    for &idx in &order[..n] {
        let take = rem_low.min(ub[idx]);
        r_min += take * spec.expected_returns[idx];
        rem_low = (rem_low - take).max(0.0);
    }
    let mut rem_high = 1.0_f64;
    let mut r_max = 0.0_f64;
    for k in 0..n {
        let idx = order[n - 1 - k];
        let take = rem_high.min(ub[idx]);
        r_max += take * spec.expected_returns[idx];
        rem_high = (rem_high - take).max(0.0);
    }
    (r_min, r_max)
}

enum ToleranceBracket {
    Solved([f64; MAX_PORTFOLIO_ASSETS]),
    Bracket {
        g_lo: f64,
        f_lo: f64,
        g_hi: f64,
        f_hi: f64,
    },
}

fn bracket_risk_tolerance(
    spec: &PortfolioSpec,
    target_return: f64,
    r0: f64,
    ub: &[f64; MAX_PORTFOLIO_ASSETS],
) -> Result<ToleranceBracket, PortfolioError> {
    let dir = if target_return > r0 { 1.0 } else { -1.0 };
    let mut step = 0.25_f64 * dir;
    let mut g_lo = 0.0_f64;
    let mut f_lo = r0 - target_return;
    let mut g_hi = step;
    for _ in 0..48 {
        let (w_try, r_try) = solve_risk_tolerance(spec, g_hi, ub)?;
        let f_hi = r_try - target_return;
        if f_hi.abs() <= KKT_TOL {
            return Ok(ToleranceBracket::Solved(w_try));
        }
        if (f_lo < 0.0 && f_hi > 0.0) || (f_lo > 0.0 && f_hi < 0.0) {
            return if g_lo <= g_hi {
                Ok(ToleranceBracket::Bracket {
                    g_lo,
                    f_lo,
                    g_hi,
                    f_hi,
                })
            } else {
                Ok(ToleranceBracket::Bracket {
                    g_lo: g_hi,
                    f_lo: f_hi,
                    g_hi: g_lo,
                    f_hi: f_lo,
                })
            };
        }
        g_lo = g_hi;
        f_lo = f_hi;
        step *= 2.0;
        g_hi += step;
    }
    let (w_edge, _) = solve_risk_tolerance(spec, g_hi, ub)?;
    Ok(ToleranceBracket::Solved(w_edge))
}

pub(super) fn root_find_target_return(
    spec: &PortfolioSpec,
    target_return: f64,
    ub: &[f64; MAX_PORTFOLIO_ASSETS],
) -> Result<[f64; MAX_PORTFOLIO_ASSETS], PortfolioError> {
    let (w0, r0) = solve_risk_tolerance(spec, 0.0, ub)?;
    if (r0 - target_return).abs() <= KKT_TOL {
        return Ok(w0);
    }
    let (mut g_lo, mut f_lo, mut g_hi, mut f_hi) =
        match bracket_risk_tolerance(spec, target_return, r0, ub)? {
            ToleranceBracket::Solved(exact) => return Ok(exact),
            ToleranceBracket::Bracket {
                g_lo,
                f_lo,
                g_hi,
                f_hi,
            } => (g_lo, f_lo, g_hi, f_hi),
        };
    for iter in 0..64 {
        let use_secant = iter % 3 != 2 && (f_hi - f_lo).abs() > 1e-16;
        let cand = if use_secant {
            let sec = g_hi - f_hi * (g_hi - g_lo) / (f_hi - f_lo);
            let pad = 0.05 * (g_hi - g_lo);
            if sec > g_lo + pad && sec < g_hi - pad {
                sec
            } else {
                0.5 * (g_lo + g_hi)
            }
        } else {
            0.5 * (g_lo + g_hi)
        };
        let (w_mid, r_mid) = solve_risk_tolerance(spec, cand, ub)?;
        let f_mid = r_mid - target_return;
        if f_mid.abs() <= KKT_TOL || (g_hi - g_lo).abs() <= 1e-14 * (1.0 + cand.abs()) {
            return Ok(w_mid);
        }
        if (f_lo < 0.0 && f_mid > 0.0) || (f_lo > 0.0 && f_mid < 0.0) {
            g_hi = cand;
            f_hi = f_mid;
        } else {
            g_lo = cand;
            f_lo = f_mid;
        }
    }
    let (w_final, _) = solve_risk_tolerance(spec, 0.5 * (g_lo + g_hi), ub)?;
    Ok(w_final)
}

// [cite:clinical_trials_statistics]
