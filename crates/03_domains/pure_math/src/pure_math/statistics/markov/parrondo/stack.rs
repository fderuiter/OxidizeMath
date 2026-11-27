//! Bounded periodic game schedules and zero-allocation 9-state stack Markov solver
//! for modulo-$M$ Parrondo Brownian ratchets.

use super::{ALLOWED_MODULI, MAX_MODULUS, ParrondoGame, ParrondoSchedule, Result};
use crate::error::MarkovError;
use math_commons::stochastic::SolverWorkspace;
use nalgebra::{DMatrix, DVector};

#[verified_engine::verified]
pub(super) fn solve_stationary_exact(
    ws: &mut SolverWorkspace,
    transition: &DMatrix<f64>,
) -> Result<DVector<f64>> {
    let m = transition.nrows();
    debug_assert_eq!(m, transition.ncols());
    let staged = ws.operator_staging_mut(m);
    for r in 0..m {
        for c in 0..m {
            staged[math_commons::math_kernel::types::flatten_2d_index(c, r, m)] =
                transition[(r, c)];
        }
    }
    ws.solve_staged_dtmc_stationary_slice(m)
        .map(DVector::from_column_slice)
        .ok_or_else(|| MarkovError::SingularMatrix {
            context: "Parrondo transition matrix lacks a unique stationary distribution"
                .to_string(),
        })
}

#[verified_engine::verified]
pub(super) fn evaluate_schedule_cycle(
    p_a: &DMatrix<f64>,
    p_b: &DMatrix<f64>,
    gain_vec_a: &DVector<f64>,
    gain_vec_b: &DVector<f64>,
    pi_cycle_start: &DVector<f64>,
    schedule: &ParrondoSchedule,
) -> (f64, DVector<f64>) {
    let (mut dist, mut accum_pi) = (pi_cycle_start.clone(), DVector::zeros(pi_cycle_start.len()));
    let (mut total_gain, p_a_t, p_b_t) = (0.0, p_a.transpose(), p_b.transpose());
    for &step in schedule.steps() {
        accum_pi += &dist;
        let (gain_v, p_t) = match step {
            ParrondoGame::GameA => (gain_vec_a, &p_a_t),
            ParrondoGame::GameB => (gain_vec_b, &p_b_t),
        };
        total_gain += dist.dot(gain_v);
        dist = p_t * &dist;
    }
    let inv_len = 1.0 / (schedule.len() as f64);
    (total_gain * inv_len, accum_pi * inv_len)
}

/// Stack-resident stationary drift and trap-occupancy summary matching [`super::solve_parrondo_ratchet`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackParrondoSummary {
    /// Stationary expected step gain of isolated Game A.
    pub gain_a: f64,
    /// Stationary expected step gain of isolated Game B.
    pub gain_b: f64,
    /// Stationary expected step gain of 50/50 random mixture.
    pub gain_mix: f64,
    /// Cycle-averaged stationary step gain of the periodic schedule.
    pub gain_schedule: f64,
    /// Stationary trap residue `r = 0` probability under isolated Game B.
    pub pi_b_trap: f64,
    /// Cycle-averaged trap residue `r = 0` probability under the schedule.
    pub pi_sched_trap: f64,
    /// `true` iff `gain_a < 0`, `gain_b < 0`, and `gain_schedule > 0`.
    pub is_paradox: bool,
}

type StackMatrix = [[f64; MAX_MODULUS]; MAX_MODULUS];
type StackVec = [f64; MAX_MODULUS];

#[verified_engine::verified]
fn build_transition_stack(m: usize, p_zero: f64, p_other: f64) -> StackMatrix {
    let mut mat = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for (s, row) in mat.iter_mut().enumerate().take(m) {
        let p_win = if s == 0 { p_zero } else { p_other };
        row[(s + 1) % m] += p_win;
        row[(s + m - 1) % m] += 1.0 - p_win;
    }
    mat
}

#[verified_engine::verified]
fn mat_mul_stack(lhs: &StackMatrix, rhs: &StackMatrix, m: usize) -> StackMatrix {
    let mut out = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for i in 0..m {
        for k in 0..m {
            let a = lhs[i][k];
            if a != 0.0 {
                for j in 0..m {
                    out[i][j] += a * rhs[k][j];
                }
            }
        }
    }
    out
}

#[verified_engine::verified]
fn row_vec_mul_stack(vec: &StackVec, mat: &StackMatrix, m: usize) -> StackVec {
    let mut out = [0.0; MAX_MODULUS];
    for i in 0..m {
        let v = vec[i];
        if v != 0.0 {
            for j in 0..m {
                out[j] += v * mat[i][j];
            }
        }
    }
    out
}

#[verified_engine::verified]
fn power_iter_stationary_stack(mat: &StackMatrix, m: usize) -> StackVec {
    let mut pi = [0.0; MAX_MODULUS];
    let inv = 1.0 / (m as f64);
    for v in pi.iter_mut().take(m) {
        *v = inv;
    }
    for _ in 0..256 {
        let stepped = row_vec_mul_stack(&pi, mat, m);
        let mut diff = 0.0_f64;
        for s in 0..m {
            let next = 0.5 * pi[s] + 0.5 * stepped[s];
            diff = diff.max((next - pi[s]).abs());
            pi[s] = next;
        }
        if diff < 1e-14 {
            break;
        }
    }
    pi
}

#[verified_engine::verified]
fn solve_stationary_stack(mat: &StackMatrix, m: usize) -> StackVec {
    let mut aug = [[0.0; MAX_MODULUS + 1]; MAX_MODULUS];
    for r in 0..(m - 1) {
        for c in 0..m {
            aug[r][c] = mat[c][r] - if r == c { 1.0 } else { 0.0 };
        }
    }
    aug[m - 1][..m].fill(1.0);
    aug[m - 1][m] = 1.0;

    for col in 0..m {
        let mut pivot = col;
        let mut best = aug[col][col].abs();
        for (r, row) in aug.iter().enumerate().take(m).skip(col + 1) {
            let cand = row[col].abs();
            if cand > best {
                best = cand;
                pivot = r;
            }
        }
        if best < 1e-13 {
            return power_iter_stationary_stack(mat, m);
        }
        if pivot != col {
            aug.swap(col, pivot);
        }
        let pivot_row = aug[col];
        let inv_diag = 1.0 / pivot_row[col];
        for row in aug.iter_mut().take(m).skip(col + 1) {
            let factor = row[col] * inv_diag;
            if factor != 0.0 {
                for (dst, &piv) in row[col..=m].iter_mut().zip(&pivot_row[col..=m]) {
                    *dst -= factor * piv;
                }
            }
        }
    }

    let mut sol = [0.0; MAX_MODULUS];
    for r in (0..m).rev() {
        let mut rhs = aug[r][m];
        for c in (r + 1)..m {
            rhs -= aug[r][c] * sol[c];
        }
        sol[r] = (rhs / aug[r][r]).max(0.0);
    }
    let sum: f64 = sol.iter().take(m).sum();
    if sum > 0.0 {
        for v in sol.iter_mut().take(m) {
            *v /= sum;
        }
    }
    sol
}

#[inline]
#[verified_engine::verified]
fn step_gain_b_stack(pi: &StackVec, m: usize, p_b0: f64, p_b1: f64) -> f64 {
    let mut p_win = 0.0;
    for (s, &weight) in pi.iter().enumerate().take(m) {
        p_win += weight * if s == 0 { p_b0 } else { p_b1 };
    }
    2.0 * p_win - 1.0
}

#[verified_engine::verified]
fn evaluate_cycle_schedule_stack(
    mat_a: &StackMatrix,
    mat_b: &StackMatrix,
    m: usize,
    gain_a: f64,
    p_b0: f64,
    p_b1: f64,
    schedule: &[ParrondoGame],
) -> (f64, f64) {
    let mut mat_cycle = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for (i, row) in mat_cycle.iter_mut().enumerate().take(m) {
        row[i] = 1.0;
    }
    for &g in schedule {
        let step_mat = match g {
            ParrondoGame::GameA => mat_a,
            ParrondoGame::GameB => mat_b,
        };
        mat_cycle = mat_mul_stack(&mat_cycle, step_mat, m);
    }

    let mut pi_step = solve_stationary_stack(&mat_cycle, m);
    let mut pi_sched_trap = 0.0;
    let mut gain_sum = 0.0;
    for &g in schedule {
        pi_sched_trap += pi_step[0];
        match g {
            ParrondoGame::GameA => {
                gain_sum += gain_a;
                pi_step = row_vec_mul_stack(&pi_step, mat_a, m);
            }
            ParrondoGame::GameB => {
                gain_sum += step_gain_b_stack(&pi_step, m, p_b0, p_b1);
                pi_step = row_vec_mul_stack(&pi_step, mat_b, m);
            }
        }
    }
    let inv_len = 1.0 / (schedule.len() as f64);
    (gain_sum * inv_len, pi_sched_trap * inv_len)
}

/// Solves the modulo-`M` (`M <= 9`) Parrondo Markov chain entirely on the stack with zero heap allocations.
///
/// Computes stationary residue distributions $\pi P = \pi$, $\sum_{r=0}^{M-1} \pi_r = 1$ for isolated
/// operators $P_A, P_B$, equal mixture $P_{\text{mix}} = \tfrac{1}{2}(P_A + P_B)$, and periodic
/// product $P_{\text{sched}} = \prod_{k=1}^{L} P_{s_k}$, yielding asymptotic expected capital gain
/// $$g_{\text{sched}} = \frac{1}{L} \sum_{k=0}^{L-1} \sum_{r=0}^{M-1} \pi^{(k)}_r \bigl(2 p_{s_k}(r) - 1\bigr)$$
/// per Harmer & Abbott (1999), *Nature* 402, 864 (`DOI: 10.1038/47220`).
///
/// # Stability
/// Partial-pivoted Gaussian elimination on the $(M \times (M+1))$ augmented simplex system with
/// automatic damped power-iteration fallback (`|piv| < 1e-13`) guarantees non-negative normalized
/// stationary measures in $O(M^3 + L M^2)$ operations without heap allocation.
///
/// # Domain of Validity
/// Requires odd modulus $M \in \{3, 5, 7, 9\}$, non-empty `schedule`, and finite perturbed branch
/// win probabilities $p_A, p_{B0}, p_{B1} \in (0, 1)$; returns `None` outside this domain.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::markov::parrondo::{
///     ParrondoSchedule, solve_parrondo_ratchet_stack,
/// };
///
/// let sched = ParrondoSchedule::abb();
/// let summary = solve_parrondo_ratchet_stack(3, 0.5, 0.1, 0.75, 0.005, sched.steps()).unwrap();
/// assert!(summary.is_paradox);
/// assert!((summary.gain_schedule - 0.057431).abs() < 1e-4);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn solve_parrondo_ratchet_stack(
    modulus: usize,
    p_a_base: f64,
    p_b0_base: f64,
    p_b1_base: f64,
    epsilon: f64,
    schedule: &[ParrondoGame],
) -> Option<StackParrondoSummary> {
    if !ALLOWED_MODULI.contains(&modulus) || schedule.is_empty() {
        return None;
    }
    let (p_a, p_b0, p_b1) = (p_a_base - epsilon, p_b0_base - epsilon, p_b1_base - epsilon);
    if ![p_a, p_b0, p_b1]
        .into_iter()
        .all(|p| p.is_finite() && p > 0.0 && p < 1.0)
    {
        return None;
    }

    let m = modulus;
    let mat_a = build_transition_stack(m, p_a, p_a);
    let mat_b = build_transition_stack(m, p_b0, p_b1);
    let mut mat_mix = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for r in 0..m {
        for c in 0..m {
            mat_mix[r][c] = 0.5 * (mat_a[r][c] + mat_b[r][c]);
        }
    }

    let gain_a = 2.0 * p_a - 1.0;
    let pi_b = solve_stationary_stack(&mat_b, m);
    let gain_b = step_gain_b_stack(&pi_b, m, p_b0, p_b1);
    let pi_mix = solve_stationary_stack(&mat_mix, m);
    let gain_mix = 0.5 * gain_a + 0.5 * step_gain_b_stack(&pi_mix, m, p_b0, p_b1);
    let (gain_schedule, pi_sched_trap) =
        evaluate_cycle_schedule_stack(&mat_a, &mat_b, m, gain_a, p_b0, p_b1, schedule);
    debug_assert!(gain_schedule.is_finite());

    Some(StackParrondoSummary {
        gain_a,
        gain_b,
        gain_mix,
        gain_schedule,
        pi_b_trap: pi_b[0],
        pi_sched_trap,
        is_paradox: gain_a < 0.0 && gain_b < 0.0 && gain_schedule > 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_solve_parrondo_ratchet_stack_zero_alloc_matches_exact() {
        let sched = ParrondoSchedule::abb();
        let (summary, _) = VerifiedEngine::run_verified(|| {
            solve_parrondo_ratchet_stack(3, 0.5, 0.1, 0.75, 0.005, sched.steps()).unwrap()
        });
        assert!(summary.is_paradox);
        assert!((summary.gain_a - (-0.01)).abs() < 1e-6);
        assert!((summary.gain_schedule - 0.057431).abs() < 1e-4);
    }
}

// [cite:clinical_trials_statistics]
