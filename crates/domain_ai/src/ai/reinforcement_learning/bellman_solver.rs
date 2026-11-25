//! Exact Bellman value-iteration and policy-evaluation drivers over [`SculptedGridMdp`].

// [cite:algorithmic_information_rust]

use super::algorithms::softmax_action_probs;
use super::bellman::{
    action_value_bellman_equation, bellman_optimality_value, state_value_bellman_equation,
};
use super::maze::{
    ALL_MOVES, DenseRlWorkspace, GridCellRole, SculptedGridMdp, index_to_move, move_to_index,
};
use super::types::{MarkovDecisionProcess, Policy};
use super::{GridState, Move};
use math_commons::math_kernel::types::flatten_2d_index;

/// Dense 4-action stochastic policy $\pi(a \mid s)$ over a 2D [`SculptedGridMdp`].
#[derive(Clone, Debug, PartialEq)]
pub struct TabularGridPolicy {
    width: usize,
    height: usize,
    probs: Vec<[f64; 4]>,
}

impl TabularGridPolicy {
    /// Builds a uniform random policy ($\pi(a \mid s) = 0.25$) across `width x height` cells.
    #[must_use]
    pub fn uniform(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            probs: vec![[0.25; 4]; width * height],
        }
    }

    /// Builds a greedy policy with tie-splitting from dense action values $Q(s, a)$.
    #[must_use]
    pub fn from_q_table(width: usize, height: usize, q_table: &[[f64; 4]]) -> Self {
        let count = width * height;
        let mut probs = vec![[0.25; 4]; count];
        for (i, q_row) in q_table.iter().take(count).enumerate() {
            let max_q = q_row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut ties = 0usize;
            for &q in q_row {
                if (q - max_q).abs() <= 1e-12 {
                    ties += 1;
                }
            }
            if ties > 0 {
                let share = 1.0 / (ties as f64);
                let mut dist = [0.0; 4];
                for k in 0..4 {
                    if (q_row[k] - max_q).abs() <= 1e-12 {
                        dist[k] = share;
                    }
                }
                probs[i] = dist;
            }
        }
        Self {
            width,
            height,
            probs,
        }
    }

    /// Builds a Boltzmann softmax policy $\pi_\theta(a \mid s)$ from dense preference logits.
    #[must_use]
    pub fn from_softmax_logits(width: usize, height: usize, logits: &[[f64; 4]]) -> Self {
        let count = width * height;
        let mut probs = vec![[0.25; 4]; count];
        for (i, row) in logits.iter().take(count).enumerate() {
            probs[i] = softmax_action_probs(*row);
        }
        Self {
            width,
            height,
            probs,
        }
    }

    /// Returns the 4-action probability distribution at `state`.
    #[inline]
    #[must_use]
    pub fn action_probabilities(&self, state: GridState) -> [f64; 4] {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return [0.25; 4];
        }
        self.probs[flatten_2d_index(state.x as usize, state.y as usize, self.width)]
    }
}

impl Policy<GridState, Move> for TabularGridPolicy {
    #[verified_engine::verified]
    fn probability(&self, state: &GridState, action: &Move) -> f64 {
        self.action_probabilities(*state)[move_to_index(*action)]
    }

    #[verified_engine::verified]
    fn sample(&self, state: &GridState) -> Move {
        let dist = self.action_probabilities(*state);
        let mut best_idx = 0usize;
        for idx in 1..4 {
            if dist[idx] > dist[best_idx] {
                best_idx = idx;
            }
        }
        index_to_move(best_idx)
    }
}

/// Exact Bellman optimality and policy-evaluation fields over a [`SculptedGridMdp`].
#[derive(Clone, Debug, PartialEq)]
pub struct BellmanSolution {
    /// Grid column count.
    pub width: usize,
    /// Grid row count.
    pub height: usize,
    /// Exact optimal state values $V^*(s)$ from `bellman_optimality_value`.
    pub v_star: Vec<f64>,
    /// Exact optimal action values $Q^*(s, a)$ from `action_value_bellman_equation`.
    pub q_star: Vec<[f64; 4]>,
    /// Exact state values $V_\pi(s)$ under a reference policy from `state_value_bellman_equation`.
    pub v_pi: Vec<f64>,
    /// Optimal greedy action $\arg\max_a Q^*(s, a)$ for each non-terminal passable cell.
    pub optimal_policy: Vec<Option<Move>>,
    /// Number of value-iteration sweeps performed to converge.
    pub sweeps: usize,
    /// Final maximum Bellman residual across the grid at termination.
    pub max_delta: f64,
}

impl BellmanSolution {
    /// Solves $V^*(s)$ and $Q^*(s, a)$ via value iteration calling [`bellman_optimality_value`]
    /// and [`action_value_bellman_equation`], plus uniform baseline $V_\pi(s)$ via [`state_value_bellman_equation`].
    #[must_use]
    #[verified_engine::verified]
    pub fn solve(mdp: &SculptedGridMdp, tolerance: f64, max_sweeps: usize) -> Self {
        let w = mdp.width as usize;
        let h = mdp.height as usize;
        let count = w * h;
        let tol = tolerance.max(1e-12);
        let cap_sweeps = max_sweeps.clamp(1, 1_000);

        let mut v_curr = vec![0.0_f64; count];
        let mut v_next = vec![0.0_f64; count];
        let mut sweeps = 0usize;
        let mut max_delta = 0.0_f64;

        for sweep in 0..cap_sweeps {
            max_delta = 0.0;
            for y in 0..h {
                for x in 0..w {
                    let st = GridState {
                        x: x as i32,
                        y: y as i32,
                    };
                    let idx = flatten_2d_index(x, y, w);
                    if mdp.is_terminal(&st) || mdp.cell_role(st) == GridCellRole::SolidWall {
                        v_next[idx] = 0.0;
                        continue;
                    }
                    let (reach, len) = mdp.reachable_next_states(st);
                    let updated = bellman_optimality_value(mdp, &st, &reach[..len], |ns| {
                        v_curr[flatten_2d_index(ns.x as usize, ns.y as usize, w)]
                    });
                    max_delta = max_delta.max((updated - v_curr[idx]).abs());
                    v_next[idx] = updated;
                }
            }
            std::mem::swap(&mut v_curr, &mut v_next);
            sweeps = sweep + 1;
            if max_delta <= tol {
                break;
            }
        }

        let (q_star, optimal_policy) = compute_q_star_and_greedy(mdp, &v_curr);
        let mut sol = Self {
            width: w,
            height: h,
            v_star: v_curr,
            q_star,
            v_pi: vec![0.0; count],
            optimal_policy,
            sweeps,
            max_delta,
        };
        let uniform = TabularGridPolicy::uniform(w, h);
        sol.evaluate_policy(mdp, &uniform, tol, cap_sweeps.min(150));
        sol
    }

    /// Evaluates $V_\pi(s)$ in place for `policy` by iterating [`state_value_bellman_equation`].
    #[verified_engine::verified]
    pub fn evaluate_policy(
        &mut self,
        mdp: &SculptedGridMdp,
        policy: &TabularGridPolicy,
        tolerance: f64,
        max_sweeps: usize,
    ) {
        let w = self.width;
        let h = self.height;
        let count = w * h;
        if self.v_pi.len() != count {
            self.v_pi.resize(count, 0.0);
        }
        self.v_pi.fill(0.0);
        let mut next = vec![0.0_f64; count];
        let tol = tolerance.max(1e-12);
        let cap_sweeps = max_sweeps.clamp(1, 1_000);

        for _ in 0..cap_sweeps {
            let mut max_delta = 0.0_f64;
            for y in 0..h {
                for x in 0..w {
                    let st = GridState {
                        x: x as i32,
                        y: y as i32,
                    };
                    let idx = flatten_2d_index(x, y, w);
                    if mdp.is_terminal(&st) || mdp.cell_role(st) == GridCellRole::SolidWall {
                        next[idx] = 0.0;
                        continue;
                    }
                    let (reach, len) = mdp.reachable_next_states(st);
                    let val = state_value_bellman_equation(mdp, policy, &st, &reach[..len], |ns| {
                        self.v_pi[flatten_2d_index(ns.x as usize, ns.y as usize, w)]
                    });
                    max_delta = max_delta.max((val - self.v_pi[idx]).abs());
                    next[idx] = val;
                }
            }
            self.v_pi.copy_from_slice(&next);
            if max_delta <= tol {
                break;
            }
        }
    }

    /// Reads $V^*(s)$ at `state`.
    #[inline]
    #[must_use]
    pub fn v_star_at(&self, state: GridState) -> f64 {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return 0.0;
        }
        self.v_star[flatten_2d_index(state.x as usize, state.y as usize, self.width)]
    }

    /// Reads $V_\pi(s)$ at `state`.
    #[inline]
    #[must_use]
    pub fn v_pi_at(&self, state: GridState) -> f64 {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return 0.0;
        }
        self.v_pi[flatten_2d_index(state.x as usize, state.y as usize, self.width)]
    }

    /// Reads $Q^*(s, a)$ at `(state, action)`.
    #[inline]
    #[must_use]
    pub fn q_star_at(&self, state: GridState, action: Move) -> f64 {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return 0.0;
        }
        self.q_star[flatten_2d_index(state.x as usize, state.y as usize, self.width)]
            [move_to_index(action)]
    }

    /// Returns the optimal greedy action $\arg\max_a Q^*(s, a)$ at `state`, if non-terminal.
    #[inline]
    #[must_use]
    pub fn optimal_action_at(&self, state: GridState) -> Option<Move> {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return None;
        }
        self.optimal_policy[flatten_2d_index(state.x as usize, state.y as usize, self.width)]
    }

    /// Returns the pointwise Bellman optimality residual $|\max_a Q(s, a) - V^*(s)|$ at `state`.
    #[inline]
    #[must_use]
    pub fn residual_at(&self, state: GridState, empirical_estimate: f64) -> f64 {
        (empirical_estimate - self.v_star_at(state)).abs()
    }

    /// Computes `(mean_residual, max_residual)` of $|\max_a Q(s, a) - V^*(s)|$ over passable non-terminal cells.
    #[must_use]
    pub fn q_residual_summary(&self, mdp: &SculptedGridMdp, ws: &DenseRlWorkspace) -> (f64, f64) {
        let mut sum = 0.0_f64;
        let mut max_err = 0.0_f64;
        let mut n = 0usize;
        for y in 0..(mdp.height as usize).min(self.height).min(ws.height) {
            for x in 0..(mdp.width as usize).min(self.width).min(ws.width) {
                let st = GridState {
                    x: x as i32,
                    y: y as i32,
                };
                if mdp.is_terminal(&st) || mdp.cell_role(st) == GridCellRole::SolidWall {
                    continue;
                }
                let err = self.residual_at(st, ws.max_q(st));
                sum += err;
                max_err = max_err.max(err);
                n += 1;
            }
        }
        if n == 0 {
            (0.0, 0.0)
        } else {
            (sum / (n as f64), max_err)
        }
    }
}

fn compute_q_star_and_greedy(
    mdp: &SculptedGridMdp,
    v_star: &[f64],
) -> (Vec<[f64; 4]>, Vec<Option<Move>>) {
    let w = mdp.width as usize;
    let h = mdp.height as usize;
    let count = w * h;
    let mut q_star = vec![[0.0_f64; 4]; count];
    let mut greedy = vec![None; count];

    for y in 0..h {
        for x in 0..w {
            let st = GridState {
                x: x as i32,
                y: y as i32,
            };
            let idx = flatten_2d_index(x, y, w);
            if mdp.is_terminal(&st) || mdp.cell_role(st) == GridCellRole::SolidWall {
                continue;
            }
            let (reach, len) = mdp.reachable_next_states(st);
            let mut best_a = Move::Up;
            let mut best_q = f64::NEG_INFINITY;
            for action in ALL_MOVES {
                let q = action_value_bellman_equation(mdp, &st, &action, &reach[..len], |ns| {
                    v_star[flatten_2d_index(ns.x as usize, ns.y as usize, w)]
                });
                q_star[idx][move_to_index(action)] = q;
                if q > best_q {
                    best_q = q;
                    best_a = action;
                }
            }
            greedy[idx] = Some(best_a);
        }
    }
    (q_star, greedy)
}
