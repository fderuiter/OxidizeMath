// [cite:engine_opt]
//! Lexicographic Lemke-Howson complementary pivoting for two-player bimatrix Nash equilibria.

use crate::error::OptError;
use math_scalars::workspace::SolverWorkspace;

const PIVOT_COL_TOL: f64 = 1.0e-11;
const LEX_TIE_TOL: f64 = 1.0e-12;

/// Equilibrium telemetry returned by [`solve_bimatrix_lemke_howson`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BimatrixNashStats {
    /// Number of complementary Gauss-Jordan pivots executed.
    pub pivots: usize,
    /// Equilibrium expected payoff $x^{\star T} A y^\star$ for row player 1.
    pub payoff_a: f64,
    /// Equilibrium expected payoff $x^{\star T} B y^\star$ for column player 2.
    pub payoff_b: f64,
    /// Worst-case unilateral deviation regret $\max(\varepsilon_1, \varepsilon_2)$.
    pub max_regret: f64,
}

/// Returns the exact `SolverWorkspace<f64>` capacity required by [`solve_bimatrix_lemke_howson`].
#[inline]
#[must_use]
pub const fn nash_bimatrix_workspace_len(m: usize, n: usize) -> usize {
    let total_labels = m.saturating_add(n);
    let cols = total_labels.saturating_add(1);
    let tab1 = m.saturating_mul(cols);
    let tab2 = n.saturating_mul(cols);
    tab1.saturating_add(tab2).saturating_add(total_labels)
}

/// Computes an exact mixed-strategy Nash equilibrium $(x^\star, y^\star)$ of a two-player bimatrix game $(A, B)$.
///
/// $$
/// x^{\star T} A y^\star \ge e_i^T A y^\star \;\; \forall i, \qquad
/// x^{\star T} B y^\star \ge x^{\star T} B e_j \;\; \forall j
/// $$
///
/// Reference: Lemke & Howson (1964), DOI: 10.1137/0112033; von Stengel (2002), Handbook of Game Theory.
///
/// # Stability
/// Shifts $(A, B)$ strictly positive without altering best-response correspondences and breaks minimum-ratio ties
/// lexicographically across the embedded identity columns $(I_m, I_n)$, preventing cycling on degenerate games.
///
/// # Domain of Validity
/// Requires `m >= 1`, `n >= 1`, row-major finite `payoff_a` (`m * n`) and `payoff_b` (`m * n`),
/// `strat_a` (`m`), `strat_b` (`n`), and `ws.len() >= nash_bimatrix_workspace_len(m, n)`.
#[verified_engine::verified]
pub fn solve_bimatrix_lemke_howson(
    payoff_a: &[f64],
    payoff_b: &[f64],
    m: usize,
    n: usize,
    strat_a: &mut [f64],
    strat_b: &mut [f64],
    ws: &mut SolverWorkspace<'_, f64>,
) -> Result<BimatrixNashStats, OptError> {
    validate_bimatrix_inputs(payoff_a, payoff_b, m, n, strat_a.len(), strat_b.len())?;
    let req = nash_bimatrix_workspace_len(m, n);
    if ws.len() < req {
        return Err(OptError::WorkspaceTooSmall {
            required: req,
            available: ws.len(),
        });
    }
    let cols = 1 + m + n;
    let (tab1, mut r1) = ws.try_split(m * cols)?;
    let (tab2, mut r2) = r1.try_split(n * cols)?;
    let (basis_f64, _) = r2.try_split(m + n)?;
    let (basis1, basis2) = basis_f64.split_at_mut(m);

    init_shifted_tableaux(payoff_a, payoff_b, m, n, tab1, tab2, basis1, basis2);

    let max_pivots = (m + n).saturating_mul(m + n).saturating_add(64).max(128);
    let dropped = 0usize;
    let mut leaving = pivot_tableau(tab2, basis2, n, cols, 1 + m, dropped)?;
    let mut pivots = 1usize;
    let mut enter_in_t1 = true;

    while leaving != dropped {
        if pivots >= max_pivots {
            return Err(OptError::ConvergenceFailed { iterations: pivots });
        }
        leaving = if enter_in_t1 {
            pivot_tableau(tab1, basis1, m, cols, 1, leaving)?
        } else {
            pivot_tableau(tab2, basis2, n, cols, 1 + m, leaving)?
        };
        enter_in_t1 = !enter_in_t1;
        pivots += 1;
    }

    extract_normalized_strategies(tab1, basis1, tab2, basis2, m, n, cols, strat_a, strat_b)?;
    Ok(compute_bimatrix_stats(
        payoff_a, payoff_b, m, n, strat_a, strat_b, pivots,
    ))
}

#[verified_engine::verified]
fn validate_bimatrix_inputs(
    payoff_a: &[f64],
    payoff_b: &[f64],
    m: usize,
    n: usize,
    sa_len: usize,
    sb_len: usize,
) -> Result<(), OptError> {
    let mn = m.checked_mul(n).ok_or(OptError::InvalidDimension)?;
    if m == 0
        || n == 0
        || payoff_a.len() != mn
        || payoff_b.len() != mn
        || sa_len != m
        || sb_len != n
    {
        return Err(OptError::InvalidDimension);
    }
    if payoff_a.iter().any(|v| !v.is_finite()) || payoff_b.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn init_shifted_tableaux(
    payoff_a: &[f64],
    payoff_b: &[f64],
    m: usize,
    n: usize,
    tab1: &mut [f64],
    tab2: &mut [f64],
    basis1: &mut [f64],
    basis2: &mut [f64],
) {
    let mut min_val = f64::INFINITY;
    for &v in payoff_a.iter().chain(payoff_b.iter()) {
        min_val = min_val.min(v);
    }
    let shift = if min_val <= 0.5 { 1.0 - min_val } else { 0.0 };
    let cols = 1 + m + n;

    tab1.fill(0.0);
    for i in 0..m {
        tab1[i * cols] = 1.0;
        tab1[i * cols + 1 + i] = 1.0;
        for j in 0..n {
            tab1[i * cols + 1 + m + j] = payoff_a[i * n + j] + shift;
        }
        basis1[i] = i as f64;
    }

    tab2.fill(0.0);
    for j in 0..n {
        tab2[j * cols] = 1.0;
        for i in 0..m {
            tab2[j * cols + 1 + i] = payoff_b[i * n + j] + shift;
        }
        tab2[j * cols + 1 + m + j] = 1.0;
        basis2[j] = (m + j) as f64;
    }
}

#[verified_engine::verified]
fn pivot_tableau(
    tab: &mut [f64],
    basis: &mut [f64],
    rows: usize,
    cols: usize,
    lex_start: usize,
    enter_label: usize,
) -> Result<usize, OptError> {
    let enter_col = 1 + enter_label;
    let mut best_row: Option<usize> = None;
    for r in 0..rows {
        let a_r = tab[r * cols + enter_col];
        if a_r <= PIVOT_COL_TOL {
            continue;
        }
        if let Some(b) = best_row {
            if is_lex_smaller(tab, cols, rows, lex_start, enter_col, r, b) {
                best_row = Some(r);
            }
        } else {
            best_row = Some(r);
        }
    }
    let piv_r = best_row.ok_or(OptError::NumericalInstability)?;
    let piv_val = tab[piv_r * cols + enter_col];
    let inv_piv = 1.0 / piv_val;
    for c in 0..cols {
        tab[piv_r * cols + c] *= inv_piv;
    }
    tab[piv_r * cols + enter_col] = 1.0;

    for r in 0..rows {
        if r == piv_r {
            continue;
        }
        let factor = tab[r * cols + enter_col];
        if factor.abs() > 0.0 {
            for c in 0..cols {
                tab[r * cols + c] -= factor * tab[piv_r * cols + c];
            }
            tab[r * cols + enter_col] = 0.0;
        }
    }

    let leaving = basis[piv_r] as usize;
    basis[piv_r] = enter_label as f64;
    Ok(leaving)
}

#[verified_engine::verified]
fn is_lex_smaller(
    tab: &[f64],
    cols: usize,
    rows: usize,
    lex_start: usize,
    enter_col: usize,
    cand: usize,
    best: usize,
) -> bool {
    let a_c = tab[cand * cols + enter_col];
    let a_b = tab[best * cols + enter_col];
    let lhs0 = tab[cand * cols] * a_b;
    let rhs0 = tab[best * cols] * a_c;
    let scale0 = 1.0_f64.max(lhs0.abs()).max(rhs0.abs());
    if (lhs0 - rhs0).abs() > LEX_TIE_TOL * scale0 {
        return lhs0 < rhs0;
    }
    for q in 0..rows {
        let col = lex_start + q;
        let lhs = tab[cand * cols + col] * a_b;
        let rhs = tab[best * cols + col] * a_c;
        let scale = 1.0_f64.max(lhs.abs()).max(rhs.abs());
        if (lhs - rhs).abs() > LEX_TIE_TOL * scale {
            return lhs < rhs;
        }
    }
    cand < best
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn extract_normalized_strategies(
    tab1: &[f64],
    basis1: &[f64],
    tab2: &[f64],
    basis2: &[f64],
    m: usize,
    n: usize,
    cols: usize,
    strat_a: &mut [f64],
    strat_b: &mut [f64],
) -> Result<(), OptError> {
    strat_a.fill(0.0);
    strat_b.fill(0.0);
    for r in 0..n {
        let lbl = basis2[r] as usize;
        if lbl < m {
            strat_a[lbl] = tab2[r * cols].max(0.0);
        }
    }
    for r in 0..m {
        let lbl = basis1[r] as usize;
        if lbl >= m && lbl < m + n {
            strat_b[lbl - m] = tab1[r * cols].max(0.0);
        }
    }
    normalize_simplex(strat_a)?;
    normalize_simplex(strat_b)
}

#[verified_engine::verified]
fn normalize_simplex(p: &mut [f64]) -> Result<(), OptError> {
    let sum: f64 = p.iter().copied().sum();
    if !sum.is_finite() || sum <= 1.0e-15 {
        return Err(OptError::NumericalInstability);
    }
    let inv = 1.0 / sum;
    for slot in p.iter_mut() {
        *slot *= inv;
    }
    Ok(())
}

#[verified_engine::verified]
fn compute_bimatrix_stats(
    payoff_a: &[f64],
    payoff_b: &[f64],
    m: usize,
    n: usize,
    strat_a: &[f64],
    strat_b: &[f64],
    pivots: usize,
) -> BimatrixNashStats {
    let mut exp_a = 0.0;
    let mut best_a = f64::NEG_INFINITY;
    for i in 0..m {
        let mut row_u = 0.0;
        for j in 0..n {
            row_u += payoff_a[i * n + j] * strat_b[j];
        }
        best_a = best_a.max(row_u);
        exp_a += strat_a[i] * row_u;
    }

    let mut exp_b = 0.0;
    let mut best_b = f64::NEG_INFINITY;
    for j in 0..n {
        let mut col_u = 0.0;
        for i in 0..m {
            col_u += strat_a[i] * payoff_b[i * n + j];
        }
        best_b = best_b.max(col_u);
        exp_b += strat_b[j] * col_u;
    }
    let reg = (best_a - exp_a).max(best_b - exp_b).max(0.0);
    BimatrixNashStats {
        pivots,
        payoff_a: exp_a,
        payoff_b: exp_b,
        max_regret: reg,
    }
}
