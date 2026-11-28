// [cite:engine_opt]
//! Finite-horizon backward induction and infinite-horizon Howard policy iteration (`lu_solve`) over `ScalarWorkspace`.

use crate::error::OptError;
use engine_linalg::{lu_solve, lu_workspace_len, mat_idx};
use math_scalars::workspace::{ScalarWorkspace, SolverWorkspace};

const STACK_PIVOT_CAP: usize = 512;
const POLICY_TIE_EPS: f64 = 1.0e-13;

/// Zero-allocation discrete Markov Decision Process transition visitor.
///
/// $$
/// Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a)\,\bigl[R(s, a, s') + (1 - \mathbf{1}_{\text{done}})\gamma V^\pi(s')\bigr]
/// $$
///
/// Reference: Puterman (1994), DOI: 10.1002/9780470316887; Howard (1960).
///
/// # Stability
/// Transition probabilities visited for any `(state, action)` pair must be non-negative and sum to at most `1.0`.
///
/// # Domain of Validity
/// Valid for finite state spaces `num_states() >= 1` and finite action spaces `num_actions() >= 1`.
pub trait DiscreteMdp {
    /// Total number of discrete states $S \ge 1$.
    fn num_states(&self) -> usize;
    /// Number of discrete actions $A \ge 1$.
    fn num_actions(&self) -> usize;
    /// Returns `true` if `state` is absorbing/terminal ($V(s) = 0$).
    fn is_terminal(&self, _state: usize) -> bool {
        false
    }
    /// Calls `visitor(prob, next_state, reward, done)` for every reachable outcome of `(state, action)`.
    fn visit_transitions<F>(&self, state: usize, action: usize, visitor: F)
    where
        F: FnMut(f64, usize, f64, bool);
}

/// Flat row-major tabular MDP view wrapping transition probabilities `P[s, a, s']` and expected rewards `R[s, a]`.
#[derive(Debug, Clone, Copy)]
pub struct DenseTabularMdp<'a> {
    /// Number of states $S$.
    pub n_states: usize,
    /// Number of actions $A$.
    pub n_actions: usize,
    /// Row-major tensor `transitions[(s * n_actions + a) * n_states + next_s]`.
    pub transitions: &'a [f64],
    /// Row-major matrix `rewards[s * n_actions + a]`.
    pub rewards: &'a [f64],
    /// Optional terminal state flags (`len == n_states`).
    pub terminal_mask: Option<&'a [bool]>,
}

impl DiscreteMdp for DenseTabularMdp<'_> {
    #[inline]
    fn num_states(&self) -> usize {
        self.n_states
    }

    #[inline]
    fn num_actions(&self) -> usize {
        self.n_actions
    }

    #[inline]
    fn is_terminal(&self, state: usize) -> bool {
        self.terminal_mask
            .and_then(|m| m.get(state).copied())
            .unwrap_or(false)
    }

    fn visit_transitions<F>(&self, state: usize, action: usize, mut visitor: F)
    where
        F: FnMut(f64, usize, f64, bool),
    {
        let r = self.rewards[state * self.n_actions + action];
        let base = (state * self.n_actions + action) * self.n_states;
        for next_s in 0..self.n_states {
            let p = self.transitions[base + next_s];
            if p > 0.0 {
                visitor(p, next_s, r, self.is_terminal(next_s));
            }
        }
    }
}

/// Configuration for infinite-horizon Howard policy iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HowardDpConfig {
    /// Discount factor $\gamma \in [0, 1)$.
    pub gamma: f64,
    /// Maximum policy-evaluation / policy-improvement sweeps.
    pub max_sweeps: usize,
    /// Stopping threshold on Bellman optimality residual $\|T^\star V - V\|_\infty$.
    pub tolerance: f64,
}

impl Default for HowardDpConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            max_sweeps: 32,
            tolerance: 1.0e-10,
        }
    }
}

/// Convergence telemetry returned by [`solve_infinite_horizon_dp`] and [`solve_finite_horizon_dp`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DpStats {
    /// Number of Howard policy sweeps or backward-induction stages executed.
    pub sweeps: usize,
    /// Final Bellman optimality residual $\max_s |\max_a Q(s, a) - V(s)|$.
    pub bellman_residual: f64,
    /// Whether optimality criteria were satisfied.
    pub converged: bool,
}

/// Returns the exact `SolverWorkspace<f64>` capacity required by [`solve_infinite_horizon_dp`].
#[inline]
#[must_use]
pub const fn dp_infinite_workspace_len(n_states: usize, _n_actions: usize) -> usize {
    let sys = n_states.saturating_mul(n_states);
    let lu = lu_workspace_len(n_states);
    sys.saturating_add(n_states).saturating_add(lu)
}

/// Returns the exact `SolverWorkspace<f64>` capacity required by [`solve_finite_horizon_dp`].
#[inline]
#[must_use]
pub const fn dp_finite_workspace_len(n_states: usize, n_actions: usize) -> usize {
    n_states.saturating_mul(n_actions).saturating_add(n_states)
}

/// Solves an infinite-horizon discounted MDP via Howard policy iteration using `lu_solve` over `SolverWorkspace`.
///
/// $$
/// (I - \gamma P_\pi) V^\pi = r_\pi, \qquad \pi_{k+1}(s) \in \arg\max_{a} \sum_{s'} P(s' \mid s, a)\bigl[R(s, a, s') + \gamma V^{\pi_k}(s')\bigr]
/// $$
///
/// Reference: Puterman (1994), DOI: 10.1002/9780470316887; Higham (2002), DOI: 10.1137/1.9780898718027.
///
/// # Stability
/// For $\gamma \in [0, 1)$, $I - \gamma P_\pi$ is strictly row diagonally dominant (nonsingular M-matrix);
/// `lu_solve` with partial pivoting evaluates $V^\pi$ to machine precision in $O(S^3)$ without contraction slowdowns as $\gamma \to 1^-$.
///
/// # Domain of Validity
/// Requires `1 <= n_states <= 512` (use [`solve_infinite_horizon_dp_with_pivots`] for larger state spaces),
/// `gamma` in `[0.0, 1.0)`, `values_out` (`n_states`), `q_table_out` (`n_states * n_actions`), `policy_out` (`n_states`),
/// and `ws.len() >= dp_infinite_workspace_len(n_states, n_actions)`.
#[verified_engine::verified]
pub fn solve_infinite_horizon_dp<M: DiscreteMdp>(
    mdp: &M,
    cfg: &HowardDpConfig,
    values_out: &mut [f64],
    q_table_out: &mut [f64],
    policy_out: &mut [usize],
    ws: &mut ScalarWorkspace<'_, f64>,
) -> Result<DpStats, OptError> {
    let n = mdp.num_states();
    if n > STACK_PIVOT_CAP {
        return Err(OptError::InvalidDimension);
    }
    let mut stack_pivots = [0usize; STACK_PIVOT_CAP];
    solve_infinite_horizon_dp_with_pivots(
        mdp,
        cfg,
        values_out,
        q_table_out,
        policy_out,
        &mut stack_pivots[..n],
        ws,
    )
}

/// Solves an infinite-horizon discounted MDP via Howard policy iteration with caller-supplied `pivots` (`n_states`).
///
/// $$
/// (I - \gamma P_\pi) V^\pi = r_\pi, \qquad \pi^\star(s) \in \arg\max_a Q^\star(s, a)
/// $$
///
/// Reference: Puterman (1994), DOI: 10.1002/9780470316887.
///
/// # Stability
/// Exact policy evaluation via `engine_linalg::lu_solve` paired with monotone policy improvement converges
/// in finitely many sweeps independent of $(1 - \gamma)^{-1}$.
///
/// # Domain of Validity
/// Requires `n_states >= 1`, `n_actions >= 1`, `gamma` in `[0.0, 1.0)`, `pivots.len() == n_states`, and
/// `ws.len() >= dp_infinite_workspace_len(n_states, n_actions)`.
#[verified_engine::verified]
pub fn solve_infinite_horizon_dp_with_pivots<M: DiscreteMdp>(
    mdp: &M,
    cfg: &HowardDpConfig,
    values_out: &mut [f64],
    q_table_out: &mut [f64],
    policy_out: &mut [usize],
    pivots: &mut [usize],
    ws: &mut ScalarWorkspace<'_, f64>,
) -> Result<DpStats, OptError> {
    let n = mdp.num_states();
    let na = mdp.num_actions();
    validate_howard_inputs(
        cfg,
        n,
        na,
        values_out.len(),
        q_table_out.len(),
        policy_out.len(),
        pivots.len(),
    )?;
    let req = dp_infinite_workspace_len(n, na);
    if ws.len() < req {
        return Err(OptError::WorkspaceTooSmall {
            required: req,
            available: ws.len(),
        });
    }
    let (sys_mat, mut r1) = ws.try_split(n * n)?;
    let (rhs, mut lu_ws) = r1.try_split(n)?;

    init_immediate_greedy_policy(mdp, policy_out);
    let mut stats = DpStats {
        sweeps: 0,
        bellman_residual: f64::INFINITY,
        converged: false,
    };

    for sweep in 0..cfg.max_sweeps {
        evaluate_policy_linear_system(
            mdp, cfg.gamma, policy_out, sys_mat, rhs, values_out, pivots, &mut lu_ws,
        )?;
        let (residual, changed) =
            improve_policy_greedy(mdp, cfg.gamma, values_out, q_table_out, policy_out);
        stats.sweeps = sweep + 1;
        stats.bellman_residual = residual;
        if !changed || residual <= cfg.tolerance {
            if changed {
                evaluate_policy_linear_system(
                    mdp, cfg.gamma, policy_out, sys_mat, rhs, values_out, pivots, &mut lu_ws,
                )?;
                let (final_res, _) =
                    improve_policy_greedy(mdp, cfg.gamma, values_out, q_table_out, policy_out);
                stats.bellman_residual = final_res;
            }
            stats.converged = stats.bellman_residual <= cfg.tolerance || !changed;
            break;
        }
    }
    Ok(stats)
}

/// Solves a finite-horizon MDP by exact backward induction over `horizon` stages into `values_out` and `policy_out`.
///
/// $$
/// V_t(s) = \max_{a} \sum_{s'} P(s' \mid s, a)\bigl[R(s, a, s') + \gamma V_{t+1}(s')\bigr], \qquad t = H - 1, \dots, 0
/// $$
///
/// Reference: Bertsekas (2012), *Dynamic Programming and Optimal Control*; Puterman (1994), DOI: 10.1002/9780470316887.
///
/// # Stability
/// Backward induction is an exact contraction-free dynamic program terminating in `horizon` stages.
///
/// # Domain of Validity
/// Requires `horizon >= 1`, `gamma` in `[0.0, 1.0]`, `terminal_values.len() == n_states`,
/// `values_out.len() == (horizon + 1) * n_states`, `policy_out.len() == horizon * n_states`,
/// and `ws.len() >= dp_finite_workspace_len(n_states, n_actions)`.
#[verified_engine::verified]
pub fn solve_finite_horizon_dp<M: DiscreteMdp>(
    mdp: &M,
    horizon: usize,
    gamma: f64,
    terminal_values: &[f64],
    values_out: &mut [f64],
    policy_out: &mut [usize],
    ws: &mut ScalarWorkspace<'_, f64>,
) -> Result<DpStats, OptError> {
    let n = mdp.num_states();
    let na = mdp.num_actions();
    if n == 0
        || na == 0
        || horizon == 0
        || terminal_values.len() != n
        || values_out.len() != (horizon + 1) * n
        || policy_out.len() != horizon * n
    {
        return Err(OptError::InvalidDimension);
    }
    if !gamma.is_finite() || terminal_values.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput);
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(OptError::InvalidParameter);
    }
    let req = dp_finite_workspace_len(n, na);
    if ws.len() < req {
        return Err(OptError::WorkspaceTooSmall {
            required: req,
            available: ws.len(),
        });
    }
    let (q_stage, _) = ws.try_split(n * na)?;

    values_out[horizon * n..(horizon + 1) * n].copy_from_slice(terminal_values);
    for s in 0..n {
        if mdp.is_terminal(s) {
            values_out[mat_idx(horizon, s, n)] = 0.0;
        }
    }

    for t in (0..horizon).rev() {
        let (head, tail) = values_out.split_at_mut((t + 1) * n);
        let v_curr = &mut head[t * n..(t + 1) * n];
        let v_next = &tail[..n];
        let pi_t = &mut policy_out[t * n..(t + 1) * n];
        for s in 0..n {
            if mdp.is_terminal(s) {
                v_curr[s] = 0.0;
                pi_t[s] = 0;
                continue;
            }
            let mut best_a = 0usize;
            let mut best_q = f64::NEG_INFINITY;
            for a in 0..na {
                let q = action_q_value(mdp, gamma, v_next, s, a);
                q_stage[mat_idx(s, a, na)] = q;
                if q > best_q {
                    best_q = q;
                    best_a = a;
                }
            }
            v_curr[s] = best_q;
            pi_t[s] = best_a;
        }
    }
    Ok(DpStats {
        sweeps: horizon,
        bellman_residual: 0.0,
        converged: true,
    })
}

#[verified_engine::verified]
fn validate_howard_inputs(
    cfg: &HowardDpConfig,
    n: usize,
    na: usize,
    v_len: usize,
    q_len: usize,
    pi_len: usize,
    piv_len: usize,
) -> Result<(), OptError> {
    if n == 0
        || na == 0
        || v_len != n
        || q_len != n * na
        || pi_len != n
        || piv_len != n
        || cfg.max_sweeps == 0
    {
        return Err(OptError::InvalidDimension);
    }
    if !cfg.gamma.is_finite() || !cfg.tolerance.is_finite() {
        return Err(OptError::NonFiniteInput);
    }
    if cfg.gamma < 0.0 || cfg.gamma >= 1.0 || cfg.tolerance < 0.0 {
        return Err(OptError::InvalidParameter);
    }
    Ok(())
}

#[verified_engine::verified]
fn init_immediate_greedy_policy<M: DiscreteMdp>(mdp: &M, policy_out: &mut [usize]) {
    let na = mdp.num_actions();
    for (s, slot) in policy_out.iter_mut().enumerate() {
        if mdp.is_terminal(s) {
            *slot = 0;
            continue;
        }
        let mut best_a = 0usize;
        let mut best_r = f64::NEG_INFINITY;
        for a in 0..na {
            let mut exp_r = 0.0;
            mdp.visit_transitions(s, a, |p, _, r, _| {
                exp_r += p * r;
            });
            if exp_r > best_r {
                best_r = exp_r;
                best_a = a;
            }
        }
        *slot = best_a;
    }
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn evaluate_policy_linear_system<M: DiscreteMdp>(
    mdp: &M,
    gamma: f64,
    policy: &[usize],
    sys_mat: &mut [f64],
    rhs: &mut [f64],
    values_out: &mut [f64],
    pivots: &mut [usize],
    lu_ws: &mut ScalarWorkspace<'_, f64>,
) -> Result<(), OptError> {
    let n = mdp.num_states();
    sys_mat.fill(0.0);
    rhs.fill(0.0);
    for s in 0..n {
        sys_mat[mat_idx(s, s, n)] = 1.0;
        if mdp.is_terminal(s) {
            continue;
        }
        let a = policy[s].min(mdp.num_actions() - 1);
        mdp.visit_transitions(s, a, |prob, next_s, reward, done| {
            rhs[s] += prob * reward;
            if !done && !mdp.is_terminal(next_s) {
                sys_mat[mat_idx(s, next_s, n)] -= gamma * prob;
            }
        });
    }
    let mut sub_ws = lu_ws.reborrow();
    lu_solve(sys_mat, rhs, values_out, n, pivots, &mut sub_ws)?;
    Ok(())
}

#[verified_engine::verified]
fn improve_policy_greedy<M: DiscreteMdp>(
    mdp: &M,
    gamma: f64,
    values: &[f64],
    q_table: &mut [f64],
    policy: &mut [usize],
) -> (f64, bool) {
    let n = mdp.num_states();
    let na = mdp.num_actions();
    let mut max_res = 0.0_f64;
    let mut changed = false;

    for s in 0..n {
        if mdp.is_terminal(s) {
            q_table[s * na..(s + 1) * na].fill(0.0);
            policy[s] = 0;
            continue;
        }
        let curr_a = policy[s].min(na - 1);
        let mut best_a = curr_a;
        let mut best_q = action_q_value(mdp, gamma, values, s, curr_a);
        for a in 0..na {
            let q = action_q_value(mdp, gamma, values, s, a);
            q_table[mat_idx(s, a, na)] = q;
            if q > best_q + POLICY_TIE_EPS {
                best_q = q;
                best_a = a;
            }
        }
        let max_q = q_table[s * na..(s + 1) * na]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        max_res = max_res.max((max_q - values[s]).abs());
        if best_a != curr_a {
            policy[s] = best_a;
            changed = true;
        }
    }
    (max_res, changed)
}

#[verified_engine::verified]
fn action_q_value<M: DiscreteMdp>(
    mdp: &M,
    gamma: f64,
    values: &[f64],
    state: usize,
    action: usize,
) -> f64 {
    let mut q = 0.0;
    mdp.visit_transitions(state, action, |prob, next_s, reward, done| {
        let cont = if done || mdp.is_terminal(next_s) {
            0.0
        } else {
            gamma * values[next_s]
        };
        q += prob * (reward + cont);
    });
    q
}
