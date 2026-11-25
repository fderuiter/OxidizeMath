//! Multi-asset covariance Kelly (`Sigma^-1 (mu - r * 1)`), diagonal independent Kelly,
//! and zero-allocation projected-gradient long-only Kelly solvers.

use crate::error::KellyError;
use engine_linalg::{cholesky_factor, cholesky_solve_factored, mat_idx};
use math_memory::{ArenaLayout, OwnedByteArena};
use nalgebra::DMatrix;

const CHOL_PIVOT_TOL: f64 = 1e-12;
const PG_CONVERGENCE_TOL: f64 = 1e-10;
const DEFAULT_PG_MAX_ITERS: usize = 400;

/// Summary metrics for a multi-asset Kelly portfolio allocation vector `w`.
#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioKellyMetrics {
    /// Gross leverage $\sum_{i=1}^N |w_i|$.
    pub gross_leverage: f64,
    /// Net leverage $\sum_{i=1}^N w_i$.
    pub net_leverage: f64,
    /// Portfolio expected excess return $w^T (\mu - r \mathbf{1})$.
    pub expected_excess_return: f64,
    /// Portfolio return variance $\sigma_p^2 = w^T \Sigma w$.
    pub portfolio_variance: f64,
    /// Portfolio annualized volatility $\sigma_p = \sqrt{w^T \Sigma w}$.
    pub portfolio_volatility: f64,
    /// Asymptotic continuous-time excess logarithmic growth $g_{\text{excess}}(w) = w^T(\mu - r\mathbf{1}) - \frac{1}{2}w^T\Sigma w$.
    pub excess_log_growth: f64,
    /// Total expected logarithmic growth $r + g_{\text{excess}}(w)$.
    pub total_log_growth: f64,
}

/// Complete multi-asset Kelly allocation paired with its portfolio growth and risk metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioKellySolution {
    /// Optimal or evaluated portfolio weights per asset.
    pub weights: Vec<f64>,
    /// Portfolio leverage, variance, and logarithmic growth summary.
    pub metrics: PortfolioKellyMetrics,
}

pub(crate) struct KellySlices<'a> {
    pub(crate) sym_flat: &'a mut [f64],
    pub(crate) chol_flat: &'a mut [f64],
    pub(crate) excess: &'a mut [f64],
    pub(crate) grad: &'a mut [f64],
    pub(crate) candidate: &'a mut [f64],
    pub(crate) sort_scratch: &'a mut [f64],
}

/// Preallocated scratch buffers enabling zero-allocation covariance and projected-gradient Kelly solves.
#[derive(Debug, Clone)]
pub struct PortfolioKellyWorkspace {
    arena: OwnedByteArena,
    max_assets: usize,
}

impl PortfolioKellyWorkspace {
    /// Allocates reusable scratch storage for portfolios up to `max_assets` legs.
    #[verified_engine::verified]
    pub fn with_capacity(max_assets: usize) -> Self {
        let n = max_assets.max(1);
        let nn = n.checked_mul(n).expect("kelly matrix capacity overflow");
        let mut layout = ArenaLayout::new();
        let _ = layout.add_slice::<f64>(nn).expect("kelly sym_flat");
        let _ = layout.add_slice::<f64>(nn).expect("kelly chol_flat");
        let _ = layout.add_slice::<f64>(n).expect("kelly excess");
        let _ = layout.add_slice::<f64>(n).expect("kelly grad");
        let _ = layout.add_slice::<f64>(n).expect("kelly candidate");
        let _ = layout.add_slice::<f64>(n).expect("kelly sort_scratch");
        Self {
            arena: OwnedByteArena::with_capacity(layout.total_bytes()),
            max_assets: n,
        }
    }

    pub(crate) fn check_capacity(&self, n: usize) -> Result<(), KellyError> {
        if self.max_assets < n {
            return Err(KellyError::DimensionMismatch {
                expected: n,
                actual: self.max_assets,
            });
        }
        Ok(())
    }

    pub(crate) fn slices_mut(&mut self, n: usize) -> KellySlices<'_> {
        let nn = n * n;
        let mut scope = self.arena.reborrow();
        KellySlices {
            sym_flat: scope.retain_slice::<f64>(nn).expect("sym_flat"),
            chol_flat: scope.retain_slice::<f64>(nn).expect("chol_flat"),
            excess: scope.retain_slice::<f64>(n).expect("excess"),
            grad: scope.retain_slice::<f64>(n).expect("grad"),
            candidate: scope.retain_slice::<f64>(n).expect("candidate"),
            sort_scratch: scope.retain_slice::<f64>(n).expect("sort_scratch"),
        }
    }
}

pub(crate) fn validate_covariance_inputs(
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
) -> Result<usize, KellyError> {
    let n = expected_returns.len();
    if n == 0 {
        return Err(KellyError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    if covariance.nrows() != n || covariance.ncols() != n {
        return Err(KellyError::DimensionMismatch {
            expected: n,
            actual: covariance.ncols(),
        });
    }
    if !risk_free_rate.is_finite() || expected_returns.iter().any(|v| !v.is_finite()) {
        return Err(KellyError::InvalidParameter {
            reason: "expected returns and risk-free rate must be finite",
        });
    }
    for i in 0..n {
        if !covariance[(i, i)].is_finite() || covariance[(i, i)] <= 0.0 {
            return Err(KellyError::NotPositiveDefinite);
        }
        for j in (i + 1)..n {
            let a = covariance[(i, j)];
            let b = covariance[(j, i)];
            if !a.is_finite()
                || !b.is_finite()
                || (a - b).abs() > math_commons::registry::TOLERANCE_STANDARD
            {
                return Err(KellyError::InvalidParameter {
                    reason: "covariance matrix must be finite and symmetric",
                });
            }
        }
    }
    Ok(n)
}

/// Computes in-place lower-triangular Cholesky factor `L` into row-major `chol_flat[..n*n]`
/// via [`engine_linalg::cholesky_factor`].
pub(crate) fn cholesky_into_flat(
    covariance: &DMatrix<f64>,
    n: usize,
    sym_flat: &mut [f64],
    chol_flat: &mut [f64],
) -> Result<(), KellyError> {
    for i in 0..n {
        sym_flat[mat_idx(i, i, n)] = covariance[(i, i)];
        for j in (i + 1)..n {
            let avg = 0.5 * (covariance[(i, j)] + covariance[(j, i)]);
            sym_flat[mat_idx(i, j, n)] = avg;
            sym_flat[mat_idx(j, i, n)] = avg;
        }
    }
    cholesky_factor(&sym_flat[..n * n], &mut chol_flat[..n * n], n)
        .map_err(|_| KellyError::NotPositiveDefinite)?;
    for i in 0..n {
        if chol_flat[mat_idx(i, i, n)] <= CHOL_PIVOT_TOL {
            return Err(KellyError::NotPositiveDefinite);
        }
    }
    Ok(())
}

/// Evaluates leverage, variance, and continuous-time log-growth metrics for arbitrary weights `w`.
#[verified_engine::verified]
pub fn evaluate_portfolio_kelly_metrics(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
) -> Result<PortfolioKellyMetrics, KellyError> {
    let n = validate_covariance_inputs(expected_returns, covariance, risk_free_rate)?;
    if weights.len() != n {
        return Err(KellyError::DimensionMismatch {
            expected: n,
            actual: weights.len(),
        });
    }
    if weights.iter().any(|w| !w.is_finite()) {
        return Err(KellyError::InvalidParameter {
            reason: "portfolio weights must be finite",
        });
    }
    Ok(compute_metrics_unchecked(
        weights,
        expected_returns,
        covariance,
        risk_free_rate,
    ))
}

fn compute_metrics_unchecked(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
) -> PortfolioKellyMetrics {
    let n = weights.len();
    let mut gross_leverage = 0.0;
    let mut net_leverage = 0.0;
    let mut expected_excess_return = 0.0;
    let mut portfolio_variance = 0.0;

    for i in 0..n {
        let wi = weights[i];
        gross_leverage += wi.abs();
        net_leverage += wi;
        expected_excess_return += wi * (expected_returns[i] - risk_free_rate);
        let mut row_dot = 0.0;
        for (j, &wj) in weights.iter().enumerate() {
            row_dot += covariance[(i, j)] * wj;
        }
        portfolio_variance += wi * row_dot;
    }
    let clean_var = portfolio_variance.max(0.0);
    let excess_log_growth = expected_excess_return - 0.5 * clean_var;
    PortfolioKellyMetrics {
        gross_leverage,
        net_leverage,
        expected_excess_return,
        portfolio_variance: clean_var,
        portfolio_volatility: clean_var.sqrt(),
        excess_log_growth,
        total_log_growth: risk_free_rate + excess_log_growth,
    }
}

#[doc = include_str!("../../../../../../docs/math_specs/snippets/kelly_unconstrained_covariance.md")]
#[verified_engine::verified]
pub fn solve_unconstrained_kelly_into(
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
    out_weights: &mut [f64],
    workspace: &mut PortfolioKellyWorkspace,
) -> Result<PortfolioKellyMetrics, KellyError> {
    let n = validate_covariance_inputs(expected_returns, covariance, risk_free_rate)?;
    if out_weights.len() != n {
        return Err(KellyError::DimensionMismatch {
            expected: n,
            actual: out_weights.len(),
        });
    }
    workspace.check_capacity(n)?;
    let s = workspace.slices_mut(n);
    cholesky_into_flat(covariance, n, s.sym_flat, s.chol_flat)?;
    for (dst, &mu_i) in s.excess.iter_mut().zip(expected_returns) {
        *dst = mu_i - risk_free_rate;
    }
    cholesky_solve_factored(s.chol_flat, s.excess, out_weights, n)
        .map_err(|_| KellyError::NotPositiveDefinite)?;

    Ok(compute_metrics_unchecked(
        out_weights,
        expected_returns,
        covariance,
        risk_free_rate,
    ))
}

/// Computes the unconstrained multi-asset covariance Kelly allocation `Sigma^-1 (mu - r * 1)`.
#[verified_engine::verified]
pub fn unconstrained_covariance_kelly(
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
) -> Result<PortfolioKellySolution, KellyError> {
    let n = expected_returns.len();
    let mut weights = vec![0.0; n];
    let mut ws = PortfolioKellyWorkspace::with_capacity(n.max(1));
    let metrics = solve_unconstrained_kelly_into(
        expected_returns,
        covariance,
        risk_free_rate,
        &mut weights,
        &mut ws,
    )?;
    Ok(PortfolioKellySolution { weights, metrics })
}

/// Zero-allocation naive diagonal (independent) Kelly weights `w_i = (mu_i - r) / Sigma_ii`
/// evaluated against the true correlated covariance matrix `Sigma`.
#[verified_engine::verified]
pub fn independent_kelly_into(
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
    out_weights: &mut [f64],
    workspace: &mut PortfolioKellyWorkspace,
) -> Result<PortfolioKellyMetrics, KellyError> {
    let n = validate_covariance_inputs(expected_returns, covariance, risk_free_rate)?;
    if out_weights.len() != n {
        return Err(KellyError::DimensionMismatch {
            expected: n,
            actual: out_weights.len(),
        });
    }
    workspace.check_capacity(n)?;
    let s = workspace.slices_mut(n);
    cholesky_into_flat(covariance, n, s.sym_flat, s.chol_flat)?;
    for i in 0..n {
        out_weights[i] = (expected_returns[i] - risk_free_rate) / covariance[(i, i)];
    }
    Ok(compute_metrics_unchecked(
        out_weights,
        expected_returns,
        covariance,
        risk_free_rate,
    ))
}

/// Computes naive independent Kelly weights `w_i = (mu_i - r) / Sigma_ii` and evaluates true correlated growth.
#[verified_engine::verified]
pub fn independent_kelly_weights(
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
) -> Result<PortfolioKellySolution, KellyError> {
    let n = expected_returns.len();
    let mut weights = vec![0.0; n];
    let mut ws = PortfolioKellyWorkspace::with_capacity(n.max(1));
    let metrics = independent_kelly_into(
        expected_returns,
        covariance,
        risk_free_rate,
        &mut weights,
        &mut ws,
    )?;
    Ok(PortfolioKellySolution { weights, metrics })
}

fn project_nonneg_capped_simplex(
    vec: &mut [f64],
    max_leverage: Option<f64>,
    sort_scratch: &mut [f64],
) {
    let mut pos_sum = 0.0;
    for v in vec.iter_mut() {
        *v = v.max(0.0);
        pos_sum += *v;
    }
    let Some(cap) = max_leverage else {
        return;
    };
    if cap <= 0.0 {
        vec.fill(0.0);
        return;
    }
    if pos_sum <= cap {
        return;
    }

    let n = vec.len();
    sort_scratch[..n].copy_from_slice(vec);
    sort_scratch[..n].sort_unstable_by(|a, b| b.total_cmp(a));

    let mut cssv = 0.0;
    let mut rho_idx = 0usize;
    for (idx, &u_val) in sort_scratch[..n].iter().enumerate() {
        cssv += u_val;
        let theta_cand = (cssv - cap) / ((idx + 1) as f64);
        if u_val > theta_cand {
            rho_idx = idx;
        }
    }
    let prefix_sum: f64 = sort_scratch[..=rho_idx].iter().sum();
    let theta = (prefix_sum - cap) / ((rho_idx + 1) as f64);
    for v in vec.iter_mut() {
        *v = (*v - theta).max(0.0);
    }
}

fn run_projected_gradient_loop(
    covariance: &DMatrix<f64>,
    n: usize,
    max_gross_leverage: Option<f64>,
    out_weights: &mut [f64],
    excess: &[f64],
    candidate: &mut [f64],
    sort_scratch: &mut [f64],
) {
    let mut l_bound = 0.0_f64;
    for i in 0..n {
        let mut row_abs = 0.0;
        for j in 0..n {
            row_abs += covariance[(i, j)].abs();
        }
        l_bound = l_bound.max(row_abs);
    }
    let step = 0.95 / l_bound.max(1e-12);

    out_weights.fill(0.0);
    for _ in 0..DEFAULT_PG_MAX_ITERS {
        for i in 0..n {
            let mut sigma_w = 0.0;
            for (j, &wj) in out_weights.iter().enumerate() {
                sigma_w += covariance[(i, j)] * wj;
            }
            candidate[i] = out_weights[i] + step * (excess[i] - sigma_w);
        }
        project_nonneg_capped_simplex(
            &mut candidate[..n],
            max_gross_leverage,
            &mut sort_scratch[..n],
        );

        let mut max_delta = 0.0_f64;
        for (dst, &cand) in out_weights.iter_mut().zip(candidate[..n].iter()) {
            max_delta = max_delta.max((cand - *dst).abs());
            *dst = cand;
        }
        if max_delta <= PG_CONVERGENCE_TOL {
            break;
        }
    }
}

#[doc = include_str!("../../../../../../docs/math_specs/snippets/kelly_long_only_pg.md")]
#[verified_engine::verified]
pub fn solve_long_only_kelly_into(
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
    max_gross_leverage: Option<f64>,
    out_weights: &mut [f64],
    workspace: &mut PortfolioKellyWorkspace,
) -> Result<PortfolioKellyMetrics, KellyError> {
    let n = validate_covariance_inputs(expected_returns, covariance, risk_free_rate)?;
    if out_weights.len() != n {
        return Err(KellyError::DimensionMismatch {
            expected: n,
            actual: out_weights.len(),
        });
    }
    if let Some(cap) = max_gross_leverage
        && (!cap.is_finite() || cap < 0.0)
    {
        return Err(KellyError::InvalidParameter {
            reason: "max_gross_leverage must be finite and non-negative",
        });
    }
    workspace.check_capacity(n)?;
    let s = workspace.slices_mut(n);
    cholesky_into_flat(covariance, n, s.sym_flat, s.chol_flat)?;
    for (dst, &mu_i) in s.excess.iter_mut().zip(expected_returns) {
        *dst = mu_i - risk_free_rate;
    }

    run_projected_gradient_loop(
        covariance,
        n,
        max_gross_leverage,
        out_weights,
        s.excess,
        s.candidate,
        s.sort_scratch,
    );
    Ok(compute_metrics_unchecked(
        out_weights,
        expected_returns,
        covariance,
        risk_free_rate,
    ))
}

/// Computes long-only Kelly weights via projected gradient ascent (`max_gross_leverage = Some(1.0)` for unlevered simplex).
#[verified_engine::verified]
pub fn long_only_kelly_projected_gradient(
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
    max_gross_leverage: Option<f64>,
) -> Result<PortfolioKellySolution, KellyError> {
    let n = expected_returns.len();
    let mut weights = vec![0.0; n];
    let mut ws = PortfolioKellyWorkspace::with_capacity(n.max(1));
    let metrics = solve_long_only_kelly_into(
        expected_returns,
        covariance,
        risk_free_rate,
        max_gross_leverage,
        &mut weights,
        &mut ws,
    )?;
    Ok(PortfolioKellySolution { weights, metrics })
}

// [cite:clinical_trials_statistics]
