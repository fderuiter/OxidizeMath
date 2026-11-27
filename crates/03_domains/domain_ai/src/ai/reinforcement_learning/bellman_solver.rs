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
use oxidize_core::rng::OxidizeRng;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum safety sweep ceiling for large high-$\gamma$ grids (`20x20`, $\gamma \le 0.999$).
pub const MAX_BELLMAN_SWEEPS: usize = 20_000;

/// Dense 4-action stochastic policy $\pi(a \mid s)$ over a 2D [`SculptedGridMdp`].
#[derive(Debug)]
pub struct TabularGridPolicy {
    width: usize,
    height: usize,
    probs: Vec<[f64; 4]>,
    sample_seq: AtomicUsize,
}

impl Clone for TabularGridPolicy {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            probs: self.probs.clone(),
            sample_seq: AtomicUsize::new(self.sample_seq.load(Ordering::Relaxed)),
        }
    }
}

impl TabularGridPolicy {
    /// Builds a uniform random policy ($\pi(a \mid s) = 0.25$) across `width x height` cells.
    #[must_use]
    pub fn uniform(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            probs: vec![[0.25; 4]; width * height],
            sample_seq: AtomicUsize::new(0),
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
            sample_seq: AtomicUsize::new(0),
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
            sample_seq: AtomicUsize::new(0),
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

    /// Samples an action from $\pi(\cdot \mid s)$ using a caller-provided [`OxidizeRng`].
    #[must_use]
    #[verified_engine::verified]
    pub fn sample_with_rng(&self, state: &GridState, rng: &mut OxidizeRng) -> Move {
        let dist = self.action_probabilities(*state);
        let total: f64 = dist
            .iter()
            .copied()
            .filter(|p| p.is_finite() && *p > 0.0)
            .sum();
        if total <= 0.0 {
            return ALL_MOVES[rng.bounded_usize(4)];
        }
        let u = rng.uniform_closed_open::<f64>() * total;
        let mut cdf = 0.0_f64;
        let mut last_pos = 0usize;
        for (idx, &p) in dist.iter().enumerate() {
            if p.is_finite() && p > 0.0 {
                last_pos = idx;
                cdf += p;
                if u < cdf {
                    return index_to_move(idx);
                }
            }
        }
        index_to_move(last_pos)
    }
}

impl Policy<GridState, Move> for TabularGridPolicy {
    #[verified_engine::verified]
    fn probability(&self, state: &GridState, action: &Move) -> f64 {
        self.action_probabilities(*state)[move_to_index(*action)]
    }

    #[verified_engine::verified]
    fn sample(&self, state: &GridState) -> Move {
        let seq = self.sample_seq.fetch_add(1, Ordering::Relaxed) as u64;
        let state_stream = (u64::from(state.x as u32) << 32) | u64::from(state.y as u32);
        let mut rng = OxidizeRng::default().with_stream(state_stream);
        rng.set_word_pos(u128::from(seq) * 4);
        self.sample_with_rng(state, &mut rng)
    }
}

/// Exact Bellman optimality and policy-evaluation fields over a [`SculptedGridMdp`].
#[derive(Clone, Debug)]
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
    /// Computes a contraction-scaled sweep budget $\lceil \ln(\text{tol}) / \ln(\gamma) \rceil + 2(W + H)$
    /// bounded within `[250, MAX_BELLMAN_SWEEPS]` for large high-$\gamma$ grids.
    #[must_use]
    pub fn recommended_sweeps(mdp: &SculptedGridMdp, tolerance: f64) -> usize {
        let gamma = mdp.gamma.clamp(0.0, 0.9999);
        let tol = tolerance.clamp(1e-12, 1e-1);
        let diam = (mdp.width as usize)
            .saturating_add(mdp.height as usize)
            .max(4);
        let contraction_horizon = if gamma <= 1e-6 {
            diam * 2
        } else {
            (tol.ln() / gamma.ln()).ceil().max(1.0) as usize
        };
        contraction_horizon
            .saturating_add(diam * 2)
            .clamp(250, MAX_BELLMAN_SWEEPS)
    }

    /// Solves $V^*(s)$ and $Q^*(s, a)$ via symmetric Gauss-Seidel value iteration calling
    /// [`bellman_optimality_value`] and [`action_value_bellman_equation`], plus uniform baseline
    /// $V_\pi(s)$ via [`state_value_bellman_equation`].
    #[must_use]
    #[verified_engine::verified]
    pub fn solve(mdp: &SculptedGridMdp, tolerance: f64, max_sweeps: usize) -> Self {
        let w = mdp.width as usize;
        let h = mdp.height as usize;
        let count = w * h;
        let tol = tolerance.max(1e-12);
        let cap_sweeps = max_sweeps.clamp(1, MAX_BELLMAN_SWEEPS);

        let mut v_curr = vec![0.0_f64; count];
        let mut sweeps = 0usize;
        let mut max_delta = 0.0_f64;

        for sweep in 0..cap_sweeps {
            max_delta = 0.0;
            let reverse = sweep % 2 == 1;
            for step_y in 0..h {
                let y = if reverse { h - 1 - step_y } else { step_y };
                for step_x in 0..w {
                    let x = if reverse { w - 1 - step_x } else { step_x };
                    let st = GridState {
                        x: x as i32,
                        y: y as i32,
                    };
                    let idx = flatten_2d_index(x, y, w);
                    if mdp.is_terminal(&st) || mdp.cell_role(st) == GridCellRole::SolidWall {
                        v_curr[idx] = 0.0;
                        continue;
                    }
                    let (reach, len) = mdp.reachable_next_states(st);
                    let updated = bellman_optimality_value(mdp, &st, &reach[..len], |ns| {
                        v_curr[flatten_2d_index(ns.x as usize, ns.y as usize, w)]
                    });
                    max_delta = max_delta.max((updated - v_curr[idx]).abs());
                    v_curr[idx] = updated;
                }
            }
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
        sol.evaluate_policy(mdp, &uniform, tol, cap_sweeps);
        sol
    }

    /// Evaluates $V_\pi(s)$ in place for `policy` via symmetric Gauss-Seidel sweeps of [`state_value_bellman_equation`].
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
        let tol = tolerance.max(1e-12);
        let cap_sweeps = max_sweeps.clamp(1, MAX_BELLMAN_SWEEPS);

        for sweep in 0..cap_sweeps {
            let mut max_delta = 0.0_f64;
            let reverse = sweep % 2 == 1;
            for step_y in 0..h {
                let y = if reverse { h - 1 - step_y } else { step_y };
                for step_x in 0..w {
                    let x = if reverse { w - 1 - step_x } else { step_x };
                    let st = GridState {
                        x: x as i32,
                        y: y as i32,
                    };
                    let idx = flatten_2d_index(x, y, w);
                    if mdp.is_terminal(&st) || mdp.cell_role(st) == GridCellRole::SolidWall {
                        self.v_pi[idx] = 0.0;
                        continue;
                    }
                    let (reach, len) = mdp.reachable_next_states(st);
                    let val = state_value_bellman_equation(mdp, policy, &st, &reach[..len], |ns| {
                        self.v_pi[flatten_2d_index(ns.x as usize, ns.y as usize, w)]
                    });
                    max_delta = max_delta.max((val - self.v_pi[idx]).abs());
                    self.v_pi[idx] = val;
                }
            }
            if max_delta <= tol {
                break;
            }
        }
    }

    /// Evaluates $Q_\pi(s, a) = \sum_{s'} P(s' \mid s, a)[R(s, a, s') + \gamma V_\pi(s')]$ from `v_pi`.
    #[must_use]
    pub fn q_pi_at(&self, mdp: &SculptedGridMdp, state: GridState, action: Move) -> f64 {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
            || mdp.is_terminal(&state)
            || mdp.cell_role(state) == GridCellRole::SolidWall
        {
            return 0.0;
        }
        let (reach, len) = mdp.reachable_next_states(state);
        action_value_bellman_equation(mdp, &state, &action, &reach[..len], |ns| self.v_pi_at(*ns))
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
