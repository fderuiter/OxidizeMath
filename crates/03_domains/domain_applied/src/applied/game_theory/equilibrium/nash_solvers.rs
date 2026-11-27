//! Zero-allocation bimatrix Lemke-Howson and polymatrix regret-matching Nash solvers.

use super::Correspondence;
use engine_opt::nash::{
    BimatrixNashStats, PolymatrixConfig, PolymatrixNashStats, nash_bimatrix_workspace_len,
    nash_polymatrix_workspace_len, solve_bimatrix_lemke_howson, solve_polymatrix_regret_matching,
};
use engine_opt::{OptError, SolverWorkspace};
use nalgebra::DVector;

/// Returns the `SolverWorkspace<f64>` capacity required by [`solve_bimatrix_nash_into`].
#[inline]
#[must_use]
#[verified_engine::verified]
pub const fn bimatrix_nash_workspace_len(rows: usize, cols: usize) -> usize {
    nash_bimatrix_workspace_len(rows, cols)
}

/// Returns the `SolverWorkspace<f64>` capacity required by [`solve_polymatrix_nash_into`].
#[inline]
#[must_use]
#[verified_engine::verified]
pub const fn polymatrix_nash_workspace_len(total_actions: usize) -> usize {
    nash_polymatrix_workspace_len(total_actions)
}

/// Computes an exact two-player bimatrix Nash equilibrium $(x^\star, y^\star)$ via Lemke-Howson pivoting.
///
/// $$
/// x^{\star T} A y^\star \ge e_i^T A y^\star \;\; \forall i, \qquad
/// x^{\star T} B y^\star \ge x^{\star T} B e_j \;\; \forall j
/// $$
///
/// Reference: Lemke & Howson (1964), DOI: 10.1137/0112033;
/// [`docs/math_specs/domain_applied.md`](../../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Positive affine payoff shifting and lexicographic minimum-ratio tie-breaking prevent degeneracy cycling
/// while slicing all tableau memory from `ws` with zero heap allocations.
///
/// # Domain of Validity
/// Requires `rows >= 1`, `cols >= 1`, row-major finite `payoff_a` and `payoff_b` (`rows * cols`),
/// `strat_a` (`rows`), `strat_b` (`cols`), and `ws.len() >= bimatrix_nash_workspace_len(rows, cols)`.
#[verified_engine::verified]
pub fn solve_bimatrix_nash_into(
    payoff_a: &[f64],
    payoff_b: &[f64],
    rows: usize,
    cols: usize,
    strat_a: &mut [f64],
    strat_b: &mut [f64],
    ws: &mut SolverWorkspace<'_, f64>,
) -> Result<BimatrixNashStats, OptError> {
    solve_bimatrix_lemke_howson(payoff_a, payoff_b, rows, cols, strat_a, strat_b, ws)
}

/// Computes an $N$-player polymatrix strategy profile via zero-allocation `CFR+` regret matching.
///
/// $$
/// R_p^{t}(a) = \bigl[R_p^{t-1}(a) + u_p(a, \sigma_{-p}^t) - u_p(\sigma^t)\bigr]_+, \qquad
/// \sigma_p^{t+1}(a) \propto \max\bigl(R_p^t(a), 0\bigr)
/// $$
///
/// Reference: Hart & Mas-Colell (2000), DOI: 10.1111/1468-0262.00153;
/// [`docs/math_specs/domain_applied.md`](../../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Linear-weighted positive regret matching maintains simplex feasibility at every step without heap allocation.
///
/// # Domain of Validity
/// Requires `action_counts.len() >= 2`, `payoff_matrix.len() == M * M` ($M = \sum_p m_p$),
/// `avg_strategies_out.len() == M`, and `ws.len() >= polymatrix_nash_workspace_len(M)`.
#[verified_engine::verified]
pub fn solve_polymatrix_nash_into(
    action_counts: &[usize],
    payoff_matrix: &[f64],
    cfg: &PolymatrixConfig,
    avg_strategies_out: &mut [f64],
    ws: &mut SolverWorkspace<'_, f64>,
) -> Result<PolymatrixNashStats, OptError> {
    solve_polymatrix_regret_matching(action_counts, payoff_matrix, cfg, avg_strategies_out, ws)
}

/// Zero-allocation bimatrix best-response [`Correspondence`] verifying joint mixed-strategy profiles `[x; y]`.
#[derive(Debug, Clone, Copy)]
pub struct BimatrixBestResponse<'a> {
    /// Row-major $m \times n$ payoff matrix $A$ for player 1.
    pub payoff_a: &'a [f64],
    /// Row-major $m \times n$ payoff matrix $B$ for player 2.
    pub payoff_b: &'a [f64],
    /// Number of pure strategies $m$ for player 1.
    pub rows: usize,
    /// Number of pure strategies $n$ for player 2.
    pub cols: usize,
    /// Maximum unilateral deviation regret tolerance $\varepsilon \ge 0$.
    pub tolerance: f64,
}

impl<'a> BimatrixBestResponse<'a> {
    /// Evaluates the exact unilateral best-response regret of `(strat_a, strat_b)` with zero heap allocations.
    #[verified_engine::verified]
    pub fn unilateral_regret(&self, strat_a: &[f64], strat_b: &[f64]) -> f64 {
        if strat_a.len() != self.rows || strat_b.len() != self.cols {
            return f64::INFINITY;
        }
        let sum_a: f64 = strat_a.iter().copied().sum();
        let sum_b: f64 = strat_b.iter().copied().sum();
        if (sum_a - 1.0).abs() > self.tolerance.max(1.0e-12)
            || (sum_b - 1.0).abs() > self.tolerance.max(1.0e-12)
            || strat_a.iter().any(|&p| !p.is_finite() || p < -1.0e-12)
            || strat_b.iter().any(|&p| !p.is_finite() || p < -1.0e-12)
        {
            return f64::INFINITY;
        }
        let mut exp_a = 0.0;
        let mut best_a = f64::NEG_INFINITY;
        for (i, &sa_i) in strat_a.iter().enumerate().take(self.rows) {
            let mut u = 0.0;
            for (j, &sb_j) in strat_b.iter().enumerate().take(self.cols) {
                u += self.payoff_a[i * self.cols + j] * sb_j;
            }
            best_a = best_a.max(u);
            exp_a += sa_i * u;
        }
        let mut exp_b = 0.0;
        let mut best_b = f64::NEG_INFINITY;
        for (j, &sb_j) in strat_b.iter().enumerate().take(self.cols) {
            let mut u = 0.0;
            for (i, &sa_i) in strat_a.iter().enumerate().take(self.rows) {
                u += sa_i * self.payoff_b[i * self.cols + j];
            }
            best_b = best_b.max(u);
            exp_b += sb_j * u;
        }
        (best_a - exp_a).max(best_b - exp_b).max(0.0)
    }
}

impl Correspondence for BimatrixBestResponse<'_> {
    #[verified_engine::verified]
    fn is_in_image(&self, source: &DVector<f64>, target: &DVector<f64>) -> bool {
        let total = self.rows + self.cols;
        if source.len() != total || target.len() != total {
            return false;
        }
        let src = source.as_slice();
        let tgt = target.as_slice();
        let y = &src[self.rows..];
        let x = &src[..self.rows];
        let tx = &tgt[..self.rows];
        let ty = &tgt[self.rows..];
        let sum_tx: f64 = tx.iter().copied().sum();
        let sum_ty: f64 = ty.iter().copied().sum();
        if (sum_tx - 1.0).abs() > self.tolerance.max(1.0e-12)
            || (sum_ty - 1.0).abs() > self.tolerance.max(1.0e-12)
            || tx.iter().any(|&p| !p.is_finite() || p < -1.0e-12)
            || ty.iter().any(|&p| !p.is_finite() || p < -1.0e-12)
        {
            return false;
        }
        let mut best_a = f64::NEG_INFINITY;
        let mut tgt_a = 0.0;
        for (i, &tx_i) in tx.iter().enumerate().take(self.rows) {
            let mut u = 0.0;
            for (j, &y_j) in y.iter().enumerate().take(self.cols) {
                u += self.payoff_a[i * self.cols + j] * y_j;
            }
            best_a = best_a.max(u);
            tgt_a += tx_i * u;
        }
        let mut best_b = f64::NEG_INFINITY;
        let mut tgt_b = 0.0;
        for (j, &ty_j) in ty.iter().enumerate().take(self.cols) {
            let mut u = 0.0;
            for (i, &x_i) in x.iter().enumerate().take(self.rows) {
                u += x_i * self.payoff_b[i * self.cols + j];
            }
            best_b = best_b.max(u);
            tgt_b += ty_j * u;
        }
        (best_a - tgt_a) <= self.tolerance && (best_b - tgt_b) <= self.tolerance
    }
}
