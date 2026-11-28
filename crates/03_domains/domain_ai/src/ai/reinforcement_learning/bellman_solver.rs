//! Exact Howard policy-iteration and LU policy-evaluation drivers over [`SculptedGridMdp`].

// [cite:algorithmic_information_rust]

mod adapter;
mod policy;
#[cfg(test)]
mod tests;

pub use self::policy::TabularGridPolicy;

use self::adapter::GridDiscreteMdp;
use super::bellman::action_value_bellman_equation;
use super::maze::{
    ALL_MOVES, DenseRlWorkspace, GridCellRole, MAX_RL_GRID_DIM, SculptedGridMdp, index_to_move,
    move_to_index,
};
use super::types::MarkovDecisionProcess;
use super::{GridState, Move};
use engine_linalg::lu_solve;
use engine_opt::dp::{
    DiscreteMdp, DpStats, HowardDpConfig, dp_infinite_workspace_len, solve_infinite_horizon_dp,
};
use engine_opt::{OptError, ScalarWorkspace};
use math_commons::math_kernel::types::flatten_2d_index;

/// Maximum safety sweep ceiling for large high-$\gamma$ grids (`20x20`, $\gamma \le 0.999$).
pub const MAX_BELLMAN_SWEEPS: usize = 20_000;

/// Maximum number of grid cells (`20 x 20 = 400`) stored inline without heap `Vec` buffers.
pub const MAX_RL_GRID_CELLS: usize = (MAX_RL_GRID_DIM as usize) * (MAX_RL_GRID_DIM as usize);

const INLINE_STACK_STATES: usize = 100;
const INLINE_STACK_WS_LEN: usize = dp_infinite_workspace_len(INLINE_STACK_STATES, 4);

/// Exact `f64` workspace length required by [`BellmanSolution::solve_with_workspace`].
#[inline]
#[must_use]
#[verified_engine::verified]
pub const fn bellman_grid_workspace_len(width: usize, height: usize) -> usize {
    dp_infinite_workspace_len(width.saturating_mul(height), 4)
}

/// Exact Howard policy-iteration optimality and LU policy-evaluation fields over a [`SculptedGridMdp`].
///
/// Stores all grid value, action-value, and policy tables inline (`<= 400` cells) without heap `Vec` buffers.
#[derive(Clone, Debug)]
pub struct BellmanSolution {
    /// Grid column count.
    pub width: usize,
    /// Grid row count.
    pub height: usize,
    /// Exact optimal state values $V^*(s)$ computed by Howard policy iteration.
    pub v_star: [f64; MAX_RL_GRID_CELLS],
    /// Exact optimal action values $Q^*(s, a)$ computed at convergence.
    pub q_star: [[f64; 4]; MAX_RL_GRID_CELLS],
    /// Exact state values $V_\pi(s)$ under a reference policy solved via `(I - \gamma P_\pi) V_\pi = r_\pi`.
    pub v_pi: [f64; MAX_RL_GRID_CELLS],
    /// Optimal greedy action $\arg\max_a Q^*(s, a)$ for each non-terminal passable cell.
    pub optimal_policy: [Option<Move>; MAX_RL_GRID_CELLS],
    /// Number of Howard policy-iteration sweeps performed to converge.
    pub sweeps: usize,
    /// Final maximum Bellman optimality residual $\|T^\star V^* - V^*\|_\infty$ across the grid.
    pub max_delta: f64,
}

impl BellmanSolution {
    /// Creates an empty zeroed `BellmanSolution` for `width x height`.
    #[must_use]
    #[verified_engine::verified]
    pub const fn empty(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            v_star: [0.0; MAX_RL_GRID_CELLS],
            q_star: [[0.0; 4]; MAX_RL_GRID_CELLS],
            v_pi: [0.0; MAX_RL_GRID_CELLS],
            optimal_policy: [None; MAX_RL_GRID_CELLS],
            sweeps: 0,
            max_delta: 0.0,
        }
    }

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

    /// Solves $V^*(s)$, $Q^*(s, a)$, and uniform $V_\pi(s)$ into `self` via Howard policy iteration over `ws` with zero heap allocations.
    #[verified_engine::verified]
    pub fn solve_into(
        &mut self,
        mdp: &SculptedGridMdp,
        tolerance: f64,
        max_sweeps: usize,
        ws: &mut ScalarWorkspace<'_, f64>,
    ) -> Result<DpStats, OptError> {
        let adapter = GridDiscreteMdp::new(mdp);
        let count = adapter.count;
        self.width = adapter.width;
        self.height = mdp.height.max(0) as usize;
        self.v_star.fill(0.0);
        self.q_star.fill([0.0; 4]);
        self.v_pi.fill(0.0);
        self.optimal_policy.fill(None);

        if count == 0 {
            self.sweeps = 0;
            self.max_delta = 0.0;
            return Ok(DpStats {
                sweeps: 0,
                bellman_residual: 0.0,
                converged: true,
            });
        }

        let eff_gamma = mdp.gamma.clamp(0.0, 1.0 - 1.0e-9);
        let cfg = HowardDpConfig {
            gamma: eff_gamma,
            max_sweeps: max_sweeps.clamp(1, 64),
            tolerance: tolerance.max(1.0e-13),
        };
        let mut flat_q = [0.0_f64; MAX_RL_GRID_CELLS * 4];
        let mut pol_idx = [0usize; MAX_RL_GRID_CELLS];

        let stats = solve_infinite_horizon_dp(
            &adapter,
            &cfg,
            &mut self.v_star[..count],
            &mut flat_q[..count * 4],
            &mut pol_idx[..count],
            ws,
        )?;

        for idx in 0..count {
            let row = [
                flat_q[idx * 4],
                flat_q[idx * 4 + 1],
                flat_q[idx * 4 + 2],
                flat_q[idx * 4 + 3],
            ];
            self.q_star[idx] = row;
            if !adapter.is_terminal(idx) {
                let mut best_a = 0usize;
                for a in 1..4 {
                    if row[a] > row[best_a] {
                        best_a = a;
                    }
                }
                self.optimal_policy[idx] = Some(index_to_move(best_a));
            }
        }
        self.sweeps = stats.sweeps;
        self.max_delta = stats.bellman_residual;

        let uniform = TabularGridPolicy::uniform(self.width, self.height);
        self.evaluate_policy_with_workspace(mdp, &uniform, ws)?;
        Ok(stats)
    }

    /// Constructs a solved `BellmanSolution` over caller-provided `ScalarWorkspace` with zero heap allocations.
    #[verified_engine::verified]
    pub fn solve_with_workspace(
        mdp: &SculptedGridMdp,
        tolerance: f64,
        max_sweeps: usize,
        ws: &mut ScalarWorkspace<'_, f64>,
    ) -> Result<Self, OptError> {
        let mut sol = Self::empty(mdp.width.max(0) as usize, mdp.height.max(0) as usize);
        sol.solve_into(mdp, tolerance, max_sweeps, ws)?;
        Ok(sol)
    }

    /// Solves $V^*(s)$, $Q^*(s, a)$, and uniform $V_\pi(s)$ via Howard policy iteration.
    ///
    /// Uses stack workspace directly for grids up to `10 x 10` (`<= 100` states) so standard solves
    /// perform **zero heap allocations**.
    #[must_use]
    pub fn solve(mdp: &SculptedGridMdp, tolerance: f64, max_sweeps: usize) -> Self {
        let w = mdp.width.max(0) as usize;
        let h = mdp.height.max(0) as usize;
        let mut sol = Self::empty(w, h);
        if w * h <= INLINE_STACK_STATES {
            let mut stack_buf = [0.0_f64; INLINE_STACK_WS_LEN];
            let mut ws = ScalarWorkspace::new(&mut stack_buf);
            let _ = sol.solve_into(mdp, tolerance, max_sweeps, &mut ws);
        } else {
            let mut heap_buf = vec![0.0_f64; bellman_grid_workspace_len(w, h)];
            let mut ws = ScalarWorkspace::new(&mut heap_buf);
            let _ = sol.solve_into(mdp, tolerance, max_sweeps, &mut ws);
        }
        sol
    }

    /// Evaluates $V_\pi(s)$ exactly in place via `(I - \gamma P_\pi) V_\pi = r_\pi` using `lu_solve` over `ws`.
    #[verified_engine::verified]
    pub fn evaluate_policy_with_workspace(
        &mut self,
        mdp: &SculptedGridMdp,
        policy: &TabularGridPolicy,
        ws: &mut ScalarWorkspace<'_, f64>,
    ) -> Result<(), OptError> {
        let adapter = GridDiscreteMdp::new(mdp);
        let n = adapter.count;
        self.v_pi.fill(0.0);
        if n == 0 {
            return Ok(());
        }
        let req = bellman_grid_workspace_len(adapter.width, mdp.height.max(0) as usize);
        if ws.len() < req {
            return Err(OptError::WorkspaceTooSmall {
                required: req,
                available: ws.len(),
            });
        }
        let (sys, mut r1) = ws.try_split(n * n)?;
        let (rhs, mut lu_ws) = r1.try_split(n)?;
        sys.fill(0.0);
        rhs.fill(0.0);
        let eff_gamma = mdp.gamma.clamp(0.0, 1.0 - 1.0e-9);

        for s in 0..n {
            sys[flatten_2d_index(s, s, n)] = 1.0;
            if adapter.is_terminal(s) {
                continue;
            }
            let st = adapter.idx_to_state(s);
            let dist = policy.action_probabilities(st);
            for (a_idx, &act) in ALL_MOVES.iter().enumerate() {
                let pa = dist[a_idx];
                if pa <= 0.0 {
                    continue;
                }
                for (next_st, p) in mdp.transition_outcomes(st, act) {
                    if p <= 0.0 {
                        continue;
                    }
                    let next_idx =
                        flatten_2d_index(next_st.x as usize, next_st.y as usize, adapter.width);
                    rhs[s] += pa * p * mdp.reward(&st, &act, &next_st);
                    if !adapter.is_terminal(next_idx) {
                        sys[flatten_2d_index(next_idx, s, n)] -= eff_gamma * pa * p;
                    }
                }
            }
        }

        let mut pivots = [0usize; MAX_RL_GRID_CELLS];
        lu_solve(
            sys,
            rhs,
            &mut self.v_pi[..n],
            n,
            &mut pivots[..n],
            &mut lu_ws,
        )?;
        Ok(())
    }

    /// Evaluates $V_\pi(s)$ in place for `policy` via exact LU policy evaluation.
    pub fn evaluate_policy(
        &mut self,
        mdp: &SculptedGridMdp,
        policy: &TabularGridPolicy,
        _tolerance: f64,
        _max_sweeps: usize,
    ) {
        let w = mdp.width.max(0) as usize;
        let h = mdp.height.max(0) as usize;
        if w * h <= INLINE_STACK_STATES {
            let mut stack_buf = [0.0_f64; INLINE_STACK_WS_LEN];
            let mut ws = ScalarWorkspace::new(&mut stack_buf);
            let _ = self.evaluate_policy_with_workspace(mdp, policy, &mut ws);
        } else {
            let mut heap_buf = vec![0.0_f64; bellman_grid_workspace_len(w, h)];
            let mut ws = ScalarWorkspace::new(&mut heap_buf);
            let _ = self.evaluate_policy_with_workspace(mdp, policy, &mut ws);
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
