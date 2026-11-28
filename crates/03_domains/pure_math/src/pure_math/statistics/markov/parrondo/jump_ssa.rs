//! Capital- and history-dependent Parrondo jump-ratchet Gillespie SSA and stack drift solvers.

use super::Result;
use crate::pure_math::analysis::stochastic::{StochasticError, StochasticSystem};
use crate::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
use crate::pure_math::statistics::markov::tensor::{TimeIndex, TransitionTensor};
use nalgebra::{DMatrix, DVector};
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Parrondo ratchet state-space formulation (`Mod-3` capital vs. 4-state outcome history).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParrondoVariant {
    /// Modulo-3 capital-dependent coin biases (`X_t mod 3`).
    CapitalDependent,
    /// Four-state two-step outcome history (`LL`, `LW`, `WL`, `WW`).
    HistoryDependent,
}

/// Discrete walker state evolved across ensemble rounds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParrondoWalker {
    /// Cumulative bankroll $X_t$.
    pub capital: i32,
    /// Active Markov state index (`0..3` or `0..4`).
    pub state: u8,
}

/// Stack-allocated continuous-time state evolved by [`crate::pure_math::analysis::stochastic::GillespieSolver`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParrondoSsaState {
    /// Cumulative bankroll.
    pub capital: i64,
    /// Current Markov state index.
    pub markov_state: usize,
    /// Active game (`0` = Game A, `1` = Game B).
    pub active_game: u8,
}

/// Zero-allocation continuous-time jump ratchet implementing [`StochasticSystem`].
#[derive(Clone, Copy, Debug)]
pub struct ParrondoGillespieSystem {
    /// Per-state win probabilities under Game A.
    pub win_a: [f64; 4],
    /// Per-state win probabilities under Game B.
    pub win_b: [f64; 4],
    /// Capital- or history-dependent state transition rule.
    pub variant: ParrondoVariant,
    /// Continuous-time jump rate from Game A to Game B.
    pub switch_ab: f64,
    /// Continuous-time jump rate from Game B to Game A.
    pub switch_ba: f64,
}

impl StochasticSystem<ParrondoSsaState> for ParrondoGillespieSystem {
    #[verified_engine::verified]
    fn propensities(&self, state: &ParrondoSsaState, out: &mut Vec<f64>) {
        let idx = state.markov_state.min(3);
        let p_win = if state.active_game == 0 {
            self.win_a[idx]
        } else {
            self.win_b[idx]
        };
        out.push(p_win);
        out.push((1.0 - p_win).max(0.0));
        out.push(if state.active_game == 0 {
            self.switch_ab
        } else {
            0.0
        });
        out.push(if state.active_game == 1 {
            self.switch_ba
        } else {
            0.0
        });
    }

    #[verified_engine::verified]
    fn react(
        &self,
        state: &mut ParrondoSsaState,
        idx: usize,
    ) -> std::result::Result<(), StochasticError> {
        match idx {
            0 => {
                state.capital += 1;
                state.markov_state = next_parrondo_state(self.variant, state.markov_state, true);
                Ok(())
            }
            1 => {
                state.capital -= 1;
                state.markov_state = next_parrondo_state(self.variant, state.markov_state, false);
                Ok(())
            }
            2 => {
                state.active_game = 1;
                Ok(())
            }
            3 => {
                state.active_game = 0;
                Ok(())
            }
            other => Err(StochasticError::InvalidReactionIndex(other)),
        }
    }
}

/// Analytical drift summary computed from [`MarkovChain`] and [`TransitionTensor`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ParrondoJumpMetrics {
    /// Exact stationary drift per turn for pure Game A.
    pub drift_a: f64,
    /// Exact stationary drift per turn for pure Game B.
    pub drift_b: f64,
    /// Exact stationary drift per turn for periodic `ABB`.
    pub drift_abb: f64,
    /// Exact stationary drift per turn for random mixture.
    pub drift_mix: f64,
    /// Entry `(0, 1)` of `TransitionTensor::average_transition` across `[0, 2]`.
    pub tensor_avg_p01: f64,
    /// Mid-cycle interpolated entry `(0, 1)` at `TimeIndex(0.5)`.
    pub tensor_mid_p01: f64,
}

/// Computes the next Markov state index under `variant` after a win (`won = true`) or loss (`won = false`).
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn next_parrondo_state(variant: ParrondoVariant, cur: usize, won: bool) -> usize {
    match variant {
        ParrondoVariant::CapitalDependent => (cur + if won { 1 } else { 2 }) % 3,
        ParrondoVariant::HistoryDependent => {
            usize::from(cur == 1 || cur == 3) * 2 + usize::from(won)
        }
    }
}

/// Returns `(win_a, win_b, num_states)` for `variant` with coin disadvantage `eps`.
#[must_use]
#[verified_engine::verified]
pub fn parrondo_variant_win_probs(
    variant: ParrondoVariant,
    eps: f64,
) -> ([f64; 4], [f64; 4], usize) {
    let win_a = [(0.5 - eps).clamp(0.01, 0.99); 4];
    match variant {
        ParrondoVariant::CapitalDependent => {
            let b0 = (0.10 - eps).clamp(0.01, 0.99);
            let b1 = (0.75 - eps).clamp(0.01, 0.99);
            (win_a, [b0, b1, b1, b1], 3)
        }
        ParrondoVariant::HistoryDependent => (
            win_a,
            [
                (0.90 - eps).clamp(0.01, 0.99),
                (0.25 - eps).clamp(0.01, 0.99),
                (0.25 - eps).clamp(0.01, 0.99),
                (0.70 - eps).clamp(0.01, 0.99),
            ],
            4,
        ),
    }
}

/// Evaluates `(drift_abb, tensor_avg_p01, tensor_mid_p01)` via [`TransitionTensor`] and [`MarkovChain`].
#[verified_engine::verified]
pub fn compute_abb_tensor_metrics(
    p_a: &DMatrix<f64>,
    p_b: &DMatrix<f64>,
    win_a: &[f64; 4],
    win_b: &[f64; 4],
) -> Result<(f64, f64, f64)> {
    let n = p_a.nrows();
    let t_min = TimeIndex::new(0.0)?;
    let t_mid = TimeIndex::new(0.5)?;
    let t_one = TimeIndex::new(1.0)?;
    let t_max = TimeIndex::new(2.0)?;
    let mut tensor = TransitionTensor::new(n, t_min, t_max);
    tensor.add_time_slice(t_min, p_a.clone())?;
    tensor.add_time_slice(t_one, p_b.clone())?;
    tensor.add_time_slice(t_max, p_b.clone())?;

    let s0 = tensor.transition_matrix_at(t_min)?;
    let s1 = tensor.transition_matrix_at(t_one)?;
    let s2 = tensor.transition_matrix_at(t_max)?;
    let chain_abb = MarkovChain::new(&s0 * &s1 * &s2, vec![StateType::Transient; n])?;
    let pi0 = chain_abb.stationary_distribution().ok_or_else(|| {
        crate::error::MarkovError::SingularMatrix {
            context: "ABB cycle lacks unique stationary distribution".to_string(),
        }
    })?;
    let pi1 = s0.transpose() * &pi0;
    let pi2 = s1.transpose() * &pi1;
    let gain = |pi: &DVector<f64>, w: &[f64; 4]| -> f64 {
        (0..n).map(|s| pi[s] * (2.0 * w[s] - 1.0)).sum()
    };
    let drift_abb = (gain(&pi0, win_a) + gain(&pi1, win_b) + gain(&pi2, win_b)) / 3.0;
    let avg = tensor.average_transition(t_min, t_max, 3)?;
    let mid = tensor.transition_matrix_at(t_mid)?;
    Ok((drift_abb, avg[(0, 1)], mid[(0, 1)]))
}

/// Computes exact stationary drifts and [`TransitionTensor`] observables for `variant`.
#[verified_engine::verified]
pub fn compute_parrondo_jump_metrics(
    variant: ParrondoVariant,
    bias_epsilon: f64,
    mix_gamma: f64,
) -> Result<ParrondoJumpMetrics> {
    let (win_a, win_b, n) = parrondo_variant_win_probs(variant, bias_epsilon);
    let mut win_mix = [0.0; 4];
    for i in 0..n {
        win_mix[i] = mix_gamma * win_a[i] + (1.0 - mix_gamma) * win_b[i];
    }
    let build_mat = |probs: &[f64; 4]| -> DMatrix<f64> {
        let mut mat = DMatrix::zeros(n, n);
        for s in 0..n {
            mat[(s, next_parrondo_state(variant, s, true))] += probs[s];
            mat[(s, next_parrondo_state(variant, s, false))] += 1.0 - probs[s];
        }
        mat
    };
    let drift_of = |mat: &DMatrix<f64>, probs: &[f64; 4]| -> Result<f64> {
        let chain = MarkovChain::new(mat.clone(), vec![StateType::Transient; n])?;
        let pi = chain.stationary_distribution().ok_or_else(|| {
            crate::error::MarkovError::SingularMatrix {
                context: "Parrondo chain lacks unique stationary distribution".to_string(),
            }
        })?;
        Ok((0..n).map(|s| pi[s] * (2.0 * probs[s] - 1.0)).sum())
    };
    let p_a = build_mat(&win_a);
    let p_b = build_mat(&win_b);
    let p_mix = build_mat(&win_mix);
    let (drift_abb, tensor_avg_p01, tensor_mid_p01) =
        compute_abb_tensor_metrics(&p_a, &p_b, &win_a, &win_b)?;
    Ok(ParrondoJumpMetrics {
        drift_a: drift_of(&p_a, &win_a)?,
        drift_b: drift_of(&p_b, &win_b)?,
        drift_abb,
        drift_mix: drift_of(&p_mix, &win_mix)?,
        tensor_avg_p01,
        tensor_mid_p01,
    })
}

/// Advances an ensemble of [`ParrondoWalker`] states by one round under `probs` with zero heap allocations.
#[verified_engine::verified]
pub fn advance_parrondo_walkers(
    walkers: &mut [ParrondoWalker],
    rng: &mut OxidizeRng,
    variant: ParrondoVariant,
    probs: &[f64; 4],
) -> f64 {
    let mut sum = 0.0;
    for w in walkers.iter_mut() {
        let won = rng.r#gen::<f64>() < probs[usize::from(w.state).min(3)];
        w.capital += if won { 1 } else { -1 };
        w.state = next_parrondo_state(variant, usize::from(w.state), won) as u8;
        sum += f64::from(w.capital);
    }
    sum / (walkers.len().max(1) as f64)
}

/// Advances an ensemble of [`ParrondoWalker`] states under a `gamma`-weighted random mixture.
#[verified_engine::verified]
pub fn advance_parrondo_mix_walkers(
    walkers: &mut [ParrondoWalker],
    rng: &mut OxidizeRng,
    variant: ParrondoVariant,
    win_a: &[f64; 4],
    win_b: &[f64; 4],
    gamma: f64,
) -> f64 {
    let mut sum = 0.0;
    for w in walkers.iter_mut() {
        let idx = usize::from(w.state).min(3);
        let p = if rng.r#gen::<f64>() < gamma {
            win_a[idx]
        } else {
            win_b[idx]
        };
        let won = rng.r#gen::<f64>() < p;
        w.capital += if won { 1 } else { -1 };
        w.state = next_parrondo_state(variant, idx, won) as u8;
        sum += f64::from(w.capital);
    }
    sum / (walkers.len().max(1) as f64)
}

/// Evaluates `(drift_a, drift_b, drift_mix, drift_abb, drift_ssa)` strictly on `[f64; 8]` stack arrays.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::markov::parrondo::{
///     ParrondoVariant, evaluate_jump_ratchet_stack,
/// };
///
/// let (da, db, dmix, dabb, dssa) =
///     evaluate_jump_ratchet_stack(ParrondoVariant::CapitalDependent, 0.005, 0.5, 1.2);
/// assert!(da < 0.0 && db < 0.0);
/// assert!(dmix > 0.0 && dabb > 0.0 && dssa > 0.0);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn evaluate_jump_ratchet_stack(
    variant: ParrondoVariant,
    eps: f64,
    gamma: f64,
    lambda: f64,
) -> (f64, f64, f64, f64, f64) {
    let (win_a, win_b, n) = parrondo_variant_win_probs(variant, eps.clamp(0.0, 0.20));
    let g = gamma.clamp(0.0, 1.0);
    let mut win_mix = [0.0; 4];
    for s in 0..n {
        win_mix[s] = g * win_a[s] + (1.0 - g) * win_b[s];
    }
    let pa = build_dtmc_4(variant, &win_a, n);
    let pb = build_dtmc_4(variant, &win_b, n);
    let pmix = build_dtmc_4(variant, &win_mix, n);
    let drift_a = 2.0 * win_a[0] - 1.0;
    let pi_b = solve_linear_8(&dtmc_to_generator_8(&pb, n), n);
    let pi_mix = solve_linear_8(&dtmc_to_generator_8(&pmix, n), n);
    let drift_b: f64 = (0..n).map(|s| pi_b[s] * (2.0 * win_b[s] - 1.0)).sum();
    let drift_mix: f64 = (0..n).map(|s| pi_mix[s] * (2.0 * win_mix[s] - 1.0)).sum();

    let p_abb = mul_4(&mul_4(&pa, &pb, n), &pb, n);
    let pi0 = solve_linear_8(&dtmc_to_generator_8(&p_abb, n), n);
    let pi1 = step_4(&pa, &pi0, n);
    let pi2 = step_4(&pb, &pi1, n);
    let g0: f64 = (0..n).map(|s| pi0[s] * (2.0 * win_a[s] - 1.0)).sum();
    let g1: f64 = (0..n).map(|s| pi1[s] * (2.0 * win_b[s] - 1.0)).sum();
    let g2: f64 = (0..n).map(|s| pi2[s] * (2.0 * win_b[s] - 1.0)).sum();
    let drift_abb = (g0 + g1 + g2) / 3.0;

    let q_ssa = build_ssa_generator_8(variant, &win_a, &win_b, n, lambda.clamp(0.05, 10.0));
    let pi_ssa = solve_linear_8(&q_ssa, 2 * n);
    let mut drift_ssa = 0.0;
    for s in 0..n {
        drift_ssa += pi_ssa[s] * (2.0 * win_a[s] - 1.0);
        drift_ssa += pi_ssa[n + s] * (2.0 * win_b[s] - 1.0);
    }
    debug_assert!(drift_ssa.is_finite());
    (drift_a, drift_b, drift_mix, drift_abb, drift_ssa)
}

#[verified_engine::verified]
fn build_dtmc_4(variant: ParrondoVariant, probs: &[f64; 4], n: usize) -> [[f64; 4]; 4] {
    let mut mat = [[0.0; 4]; 4];
    for s in 0..n {
        mat[s][next_parrondo_state(variant, s, true)] += probs[s];
        mat[s][next_parrondo_state(variant, s, false)] += 1.0 - probs[s];
    }
    mat
}

#[verified_engine::verified]
fn mul_4(lhs: &[[f64; 4]; 4], rhs: &[[f64; 4]; 4], n: usize) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                out[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }
    out
}

#[verified_engine::verified]
fn step_4(mat: &[[f64; 4]; 4], dist: &[f64; 8], n: usize) -> [f64; 8] {
    let mut next = [0.0; 8];
    for i in 0..n {
        for j in 0..n {
            next[j] += dist[i] * mat[i][j];
        }
    }
    next
}

#[verified_engine::verified]
fn dtmc_to_generator_8(p: &[[f64; 4]; 4], n: usize) -> [[f64; 8]; 8] {
    let mut q = [[0.0; 8]; 8];
    for i in 0..n {
        for j in 0..n {
            q[i][j] = p[i][j] - if i == j { 1.0 } else { 0.0 };
        }
    }
    q
}

#[verified_engine::verified]
fn build_ssa_generator_8(
    variant: ParrondoVariant,
    win_a: &[f64; 4],
    win_b: &[f64; 4],
    n: usize,
    lambda: f64,
) -> [[f64; 8]; 8] {
    let mut q = [[0.0; 8]; 8];
    let sw_ab = lambda * 1.6;
    let sw_ba = lambda * 0.8;
    for s in 0..n {
        let win_s = next_parrondo_state(variant, s, true);
        let loss_s = next_parrondo_state(variant, s, false);
        q[s][s] -= 1.0 + sw_ab;
        q[s][win_s] += win_a[s];
        q[s][loss_s] += 1.0 - win_a[s];
        q[s][n + s] += sw_ab;

        let b_idx = n + s;
        q[b_idx][b_idx] -= 1.0 + sw_ba;
        q[b_idx][n + win_s] += win_b[s];
        q[b_idx][n + loss_s] += 1.0 - win_b[s];
        q[b_idx][s] += sw_ba;
    }
    q
}

#[verified_engine::verified]
fn solve_linear_8(q: &[[f64; 8]; 8], m: usize) -> [f64; 8] {
    let mut aug = [[0.0_f64; 9]; 8];
    for r in 0..(m - 1) {
        for c in 0..m {
            aug[r][c] = q[c][r];
        }
    }
    for val in aug[m - 1].iter_mut().take(m) {
        *val = 1.0;
    }
    aug[m - 1][m] = 1.0;

    for col in 0..m {
        let mut pivot = col;
        let mut best = aug[col][col].abs();
        for (row, candidate) in aug.iter().enumerate().take(m).skip(col + 1) {
            let mag = candidate[col].abs();
            if mag > best {
                best = mag;
                pivot = row;
            }
        }
        if best <= 1e-14 {
            let mut uni = [0.0; 8];
            for v in uni.iter_mut().take(m) {
                *v = 1.0 / (m as f64);
            }
            return uni;
        }
        if pivot != col {
            aug.swap(col, pivot);
        }
        let inv = 1.0 / aug[col][col];
        let pivot_row = aug[col];
        for row_vals in aug.iter_mut().take(m).skip(col + 1) {
            let factor = row_vals[col] * inv;
            for (dst, &src) in row_vals[col..=m].iter_mut().zip(&pivot_row[col..=m]) {
                *dst -= factor * src;
            }
        }
    }

    let mut pi = [0.0; 8];
    for r in (0..m).rev() {
        let mut rhs = aug[r][m];
        for c in (r + 1)..m {
            rhs -= aug[r][c] * pi[c];
        }
        pi[r] = (rhs / aug[r][r]).max(0.0);
    }
    let sum: f64 = pi.iter().take(m).sum();
    if sum > 1e-12 && sum.is_finite() {
        for val in pi.iter_mut().take(m) {
            *val /= sum;
        }
    }
    pi
}

#[cfg(test)]
#[path = "jump_ssa_tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]
