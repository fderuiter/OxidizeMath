//! `#![no_std]` optimal transport: exact 1D Wasserstein ($W_1$ / EMD) and log-domain stabilized
//! Sinkhorn-Knopp entropic optimal transport over caller-supplied scratch slices.

use crate::calibration::sort_f64_in_place;
use crate::error::ProbError;

#[inline]
const fn idx2d(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

/// Computes the exact 1D 1-Wasserstein distance $W_1(u, v) = \int_0^1 |F_u^{-1}(q) - F_v^{-1}(q)|\,dq$
/// between two ascending empirical sample slices (supporting both equal and unequal lengths)
/// with zero heap allocations.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn wasserstein_1d_sorted(u_sorted: &[f64], v_sorted: &[f64]) -> Result<f64, ProbError> {
    if u_sorted.is_empty() || v_sorted.is_empty() {
        return Err(ProbError::EmptyInput {
            context: "wasserstein_1d_sorted",
        });
    }
    if u_sorted.len() == v_sorted.len() {
        let mut sum = 0.0;
        for (&a, &b) in u_sorted.iter().zip(v_sorted.iter()) {
            if !a.is_finite() || !b.is_finite() {
                return Err(ProbError::NumericalFailure {
                    reason: "non-finite sample in wasserstein_1d_sorted",
                });
            }
            sum += (a - b).abs();
        }
        return Ok(sum / (u_sorted.len() as f64));
    }
    let m = u_sorted.len();
    let n = v_sorted.len();
    let inv_m = 1.0 / (m as f64);
    let inv_n = 1.0 / (n as f64);
    let mut i = 0usize;
    let mut j = 0usize;
    let mut prev_x = u_sorted[0].min(v_sorted[0]);
    let mut area = 0.0;
    while i < m || j < n {
        let next_u = if i < m { u_sorted[i] } else { f64::INFINITY };
        let next_v = if j < n { v_sorted[j] } else { f64::INFINITY };
        let next_x = next_u.min(next_v);
        if !next_x.is_finite() {
            return Err(ProbError::NumericalFailure {
                reason: "non-finite sample in wasserstein_1d_sorted",
            });
        }
        let cdf_diff = ((i as f64) * inv_m - (j as f64) * inv_n).abs();
        area += cdf_diff * (next_x - prev_x);
        while i < m && u_sorted[i] <= next_x {
            i += 1;
        }
        while j < n && v_sorted[j] <= next_x {
            j += 1;
        }
        prev_x = next_x;
    }
    debug_assert!(area >= 0.0);
    Ok(area)
}

/// Sorts `u` and `v` in-place via heapsort and returns exact $W_1(u, v)$ with zero heap allocations.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn wasserstein_1d_into(u: &mut [f64], v: &mut [f64]) -> Result<f64, ProbError> {
    if u.is_empty() || v.is_empty() {
        return Err(ProbError::EmptyInput {
            context: "wasserstein_1d_into",
        });
    }
    sort_f64_in_place(u)?;
    sort_f64_in_place(v)?;
    wasserstein_1d_sorted(u, v)
}

/// Convenience `alloc`-enabled wrapper computing $W_1(u, v)$ without mutating input slices.
#[cfg(feature = "alloc")]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn wasserstein_1d(u: &[f64], v: &[f64]) -> Result<f64, ProbError> {
    let mut u_buf = alloc::vec::Vec::from(u);
    let mut v_buf = alloc::vec::Vec::from(v);
    wasserstein_1d_into(&mut u_buf, &mut v_buf)
}

/// Computes exact 1D Earth Mover's Distance between two discrete histograms `a` and `b` over
/// ascending `support` locations ($W_1 = \sum_{k=0}^{n-2} |F_a(k) - F_b(k)| (x_{k+1} - x_k)$).
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn wasserstein_1d_weights(
    weights_a: &[f64],
    weights_b: &[f64],
    support: &[f64],
) -> Result<f64, ProbError> {
    let n = support.len();
    if n == 0 {
        return Err(ProbError::EmptyInput {
            context: "wasserstein_1d_weights",
        });
    }
    if weights_a.len() != n || weights_b.len() != n {
        return Err(ProbError::DimensionMismatch {
            expected: n,
            actual: weights_a.len().min(weights_b.len()),
        });
    }
    let sum_a: f64 = weights_a.iter().copied().sum();
    let sum_b: f64 = weights_b.iter().copied().sum();
    if !sum_a.is_finite() || sum_a <= 0.0 || !sum_b.is_finite() || sum_b <= 0.0 {
        return Err(ProbError::InvalidParameter {
            name: "histogram_mass",
            value: sum_a.min(sum_b),
        });
    }
    let mut cdf_a = 0.0;
    let mut cdf_b = 0.0;
    let mut cost = 0.0;
    for k in 0..n.saturating_sub(1) {
        let wa = weights_a[k];
        let wb = weights_b[k];
        let dx = support[k + 1] - support[k];
        if wa < 0.0 || wb < 0.0 || !dx.is_finite() || dx < 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "support_or_weight",
                value: dx.min(wa).min(wb),
            });
        }
        cdf_a += wa / sum_a;
        cdf_b += wb / sum_b;
        cost += (cdf_a - cdf_b).abs() * dx;
    }
    Ok(cost)
}

/// Configuration for log-domain stabilized Sinkhorn-Knopp optimal transport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinkhornConfig {
    /// Entropic regularization $\varepsilon > 0$.
    pub epsilon: f64,
    /// Maximum dual coordinate-ascent sweeps.
    pub max_iters: usize,
    /// Convergence tolerance on $\ell_1$ row-marginal error.
    pub tolerance: f64,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            max_iters: 250,
            tolerance: 1.0e-9,
        }
    }
}

/// Convergence and cost summary produced by [`sinkhorn_log_domain_into`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinkhornReport {
    /// Unregularized transport cost $\langle C, P \rangle = \sum_{i,j} C_{ij} P_{ij}$.
    pub transport_cost: f64,
    /// Entropically regularized primal objective $\langle C, P \rangle + \varepsilon \operatorname{KL}(P \| a \otimes b)$.
    pub regularized_objective: f64,
    /// Number of dual sweeps executed.
    pub iterations: usize,
    /// Final $\ell_1$ marginal residual $\sum_i |\sum_j P_{ij} - a_i|$.
    pub marginal_error: f64,
    /// Whether `marginal_error <= config.tolerance`.
    pub converged: bool,
}

/// Minimum `f64` scratch slice length (`2 * (m + n)`) required by [`sinkhorn_log_domain_into`].
#[must_use]
#[inline]
pub const fn sinkhorn_scratch_len(m: usize, n: usize) -> usize {
    2 * (m + n)
}

fn validate_simplex_weights(w: &[f64], log_out: &mut [f64]) -> Result<(), ProbError> {
    let mut sum = 0.0;
    for &v in w {
        if !v.is_finite() || v < 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "marginal_weight",
                value: v,
            });
        }
        sum += v;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err(ProbError::InvalidParameter {
            name: "marginal_sum",
            value: sum,
        });
    }
    for (slot, &v) in log_out.iter_mut().zip(w.iter()) {
        let p = v / sum;
        *slot = if p > 0.0 {
            libm::log(p)
        } else {
            f64::NEG_INFINITY
        };
    }
    Ok(())
}

fn update_row_potentials(
    f: &mut [f64],
    g: &[f64],
    log_b: &[f64],
    cost: &[f64],
    inv_eps: f64,
    eps: f64,
) {
    let n = g.len();
    for (i, f_slot) in f.iter_mut().enumerate() {
        let mut max_term = f64::NEG_INFINITY;
        for j in 0..n {
            if log_b[j].is_finite() {
                let t = log_b[j] + (g[j] - cost[idx2d(i, j, n)]) * inv_eps;
                if t > max_term {
                    max_term = t;
                }
            }
        }
        let mut sum_exp = 0.0;
        for j in 0..n {
            if log_b[j].is_finite() {
                let t = log_b[j] + (g[j] - cost[idx2d(i, j, n)]) * inv_eps;
                sum_exp += libm::exp(t - max_term);
            }
        }
        *f_slot = -eps * (max_term + libm::log(sum_exp));
    }
}

fn update_col_potentials(
    g: &mut [f64],
    f: &[f64],
    log_a: &[f64],
    cost: &[f64],
    inv_eps: f64,
    eps: f64,
) {
    let m = f.len();
    let n = g.len();
    for (j, g_slot) in g.iter_mut().enumerate() {
        let mut max_term = f64::NEG_INFINITY;
        for i in 0..m {
            if log_a[i].is_finite() {
                let t = log_a[i] + (f[i] - cost[idx2d(i, j, n)]) * inv_eps;
                if t > max_term {
                    max_term = t;
                }
            }
        }
        let mut sum_exp = 0.0;
        for i in 0..m {
            if log_a[i].is_finite() {
                let t = log_a[i] + (f[i] - cost[idx2d(i, j, n)]) * inv_eps;
                sum_exp += libm::exp(t - max_term);
            }
        }
        *g_slot = -eps * (max_term + libm::log(sum_exp));
    }
}

fn row_marginal_l1_error(
    f: &[f64],
    g: &[f64],
    log_a: &[f64],
    log_b: &[f64],
    cost: &[f64],
    inv_eps: f64,
) -> f64 {
    let n = g.len();
    let mut err = 0.0;
    for i in 0..f.len() {
        if !log_a[i].is_finite() {
            continue;
        }
        let ai = libm::exp(log_a[i]);
        let mut row_sum = 0.0;
        for j in 0..n {
            if log_b[j].is_finite() {
                let log_p = log_a[i] + log_b[j] + (f[i] + g[j] - cost[idx2d(i, j, n)]) * inv_eps;
                row_sum += libm::exp(log_p);
            }
        }
        err += (row_sum - ai).abs();
    }
    err
}

struct SinkhornDualView<'a> {
    f: &'a [f64],
    g: &'a [f64],
    log_a: &'a [f64],
    log_b: &'a [f64],
    cost_matrix: &'a [f64],
    epsilon: f64,
}

fn accumulate_sinkhorn_report(
    dual: &SinkhornDualView<'_>,
    iterations: usize,
    marginal_error: f64,
    converged: bool,
    mut plan_out: Option<&mut [f64]>,
) -> SinkhornReport {
    let m = dual.f.len();
    let n = dual.g.len();
    let inv_eps = 1.0 / dual.epsilon;
    let mut transport_cost = 0.0;
    let mut kl_div = 0.0;
    for i in 0..m {
        for j in 0..n {
            let flat = idx2d(i, j, n);
            let exponent = (dual.f[i] + dual.g[j] - dual.cost_matrix[flat]) * inv_eps;
            let pij = if dual.log_a[i].is_finite() && dual.log_b[j].is_finite() {
                libm::exp(dual.log_a[i] + dual.log_b[j] + exponent)
            } else {
                0.0
            };
            transport_cost += pij * dual.cost_matrix[flat];
            if pij > 0.0 {
                kl_div += pij * exponent;
            }
            if let Some(ref mut plan) = plan_out {
                plan[flat] = pij;
            }
        }
    }
    debug_assert!(transport_cost.is_finite());
    SinkhornReport {
        transport_cost,
        regularized_objective: transport_cost + dual.epsilon * kl_div,
        iterations,
        marginal_error,
        converged,
    }
}

fn check_sinkhorn_inputs(
    m: usize,
    n: usize,
    cost_matrix: &[f64],
    epsilon: f64,
    scratch_len: usize,
    plan_len: Option<usize>,
) -> Result<usize, ProbError> {
    if m == 0 || n == 0 {
        return Err(ProbError::EmptyInput {
            context: "sinkhorn_log_domain_into",
        });
    }
    if cost_matrix.len() != m * n {
        return Err(ProbError::DimensionMismatch {
            expected: m * n,
            actual: cost_matrix.len(),
        });
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(ProbError::InvalidParameter {
            name: "epsilon",
            value: epsilon,
        });
    }
    let req = sinkhorn_scratch_len(m, n);
    if scratch_len < req {
        return Err(ProbError::BufferTooSmall {
            required: req,
            actual: scratch_len,
        });
    }
    if let Some(plen) = plan_len
        && plen < m * n
    {
        return Err(ProbError::BufferTooSmall {
            required: m * n,
            actual: plen,
        });
    }
    for &c in cost_matrix {
        if !c.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "cost",
                value: c,
            });
        }
    }
    Ok(req)
}

/// Solves entropically regularized optimal transport in the log domain using caller-provided
/// `scratch` (`len >= sinkhorn_scratch_len(a.len(), b.len())`) with zero heap allocations.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn sinkhorn_log_domain_into(
    a: &[f64],
    b: &[f64],
    cost_matrix: &[f64],
    config: &SinkhornConfig,
    scratch: &mut [f64],
    plan_out: Option<&mut [f64]>,
) -> Result<SinkhornReport, ProbError> {
    let m = a.len();
    let n = b.len();
    let req = check_sinkhorn_inputs(
        m,
        n,
        cost_matrix,
        config.epsilon,
        scratch.len(),
        plan_out.as_ref().map(|p| p.len()),
    )?;
    let (f_slice, rest) = scratch[..req].split_at_mut(m);
    let (g_slice, rest2) = rest.split_at_mut(n);
    let (log_a, log_b) = rest2.split_at_mut(m);
    validate_simplex_weights(a, log_a)?;
    validate_simplex_weights(b, log_b)?;
    f_slice.fill(0.0);
    g_slice.fill(0.0);

    let inv_eps = 1.0 / config.epsilon;
    let mut iterations = 0usize;
    let mut marginal_error = f64::INFINITY;
    let mut converged = false;
    let max_iters = config.max_iters.max(1);
    for iter in 1..=max_iters {
        update_row_potentials(
            f_slice,
            g_slice,
            log_b,
            cost_matrix,
            inv_eps,
            config.epsilon,
        );
        update_col_potentials(
            g_slice,
            f_slice,
            log_a,
            cost_matrix,
            inv_eps,
            config.epsilon,
        );
        iterations = iter;
        if iter % 5 == 0 || iter == max_iters {
            marginal_error =
                row_marginal_l1_error(f_slice, g_slice, log_a, log_b, cost_matrix, inv_eps);
            if marginal_error <= config.tolerance {
                converged = true;
                break;
            }
        }
    }
    let dual = SinkhornDualView {
        f: f_slice,
        g: g_slice,
        log_a,
        log_b,
        cost_matrix,
        epsilon: config.epsilon,
    };
    Ok(accumulate_sinkhorn_report(
        &dual,
        iterations,
        marginal_error,
        converged,
        plan_out,
    ))
}

/// Convenience `alloc`-enabled wrapper for [`sinkhorn_log_domain_into`] returning `(SinkhornReport, Vec<f64>)`.
#[cfg(feature = "alloc")]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn sinkhorn_log_domain(
    a: &[f64],
    b: &[f64],
    cost_matrix: &[f64],
    config: &SinkhornConfig,
) -> Result<(SinkhornReport, alloc::vec::Vec<f64>), ProbError> {
    let mut scratch = alloc::vec![0.0_f64; sinkhorn_scratch_len(a.len(), b.len())];
    let mut plan = alloc::vec![0.0_f64; a.len() * b.len()];
    let report =
        sinkhorn_log_domain_into(a, b, cost_matrix, config, &mut scratch, Some(&mut plan))?;
    Ok((report, plan))
}
