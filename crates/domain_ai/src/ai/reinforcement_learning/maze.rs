//! Sculptable 2D GridWorld Markov Decision Process with solid bounce-back walls,
//! stochastic directional wind currents, and pre-allocated dense workspace buffers.

// [cite:algorithmic_information_rust]

use super::{GridState, GridWorldEnv, MarkovDecisionProcess, Move};
use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Minimum supported grid width or height for [`SculptedGridMdp`].
pub const MIN_RL_GRID_DIM: i32 = 2;

/// Maximum supported grid width or height for [`SculptedGridMdp`].
pub const MAX_RL_GRID_DIM: i32 = 20;

pub use super::workspace::{ALL_MOVES, DenseRlWorkspace, index_to_move, move_to_index};

/// Discrete spatial role assigned to each cell of a [`SculptedGridMdp`].
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GridCellRole {
    /// Traversable floor cell (`R = -0.1` step cost).
    #[default]
    Free = 0,
    /// Impermeable obstacle that bounces the agent back to its prior cell.
    SolidWall = 1,
    /// Terminal positive-reward target (`R = +10.0`).
    Goal = 2,
    /// Terminal negative-reward hazard (`R = -10.0`).
    Trap = 3,
    /// Upward crosswind cell (`dy = -1` with probability `wind_slip_prob`).
    WindUp = 4,
    /// Downward crosswind cell (`dy = +1` with probability `wind_slip_prob`).
    WindDown = 5,
    /// Leftward crosswind cell (`dx = -1` with probability `wind_slip_prob`).
    WindLeft = 6,
    /// Rightward crosswind cell (`dx = +1` with probability `wind_slip_prob`).
    WindRight = 7,
}

impl GridCellRole {
    /// Encodes this cell role as a compact `u8` tag for RLE serialization.
    #[inline]
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decodes a `u8` tag produced by [`Self::to_u8`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Free),
            1 => Some(Self::SolidWall),
            2 => Some(Self::Goal),
            3 => Some(Self::Trap),
            4 => Some(Self::WindUp),
            5 => Some(Self::WindDown),
            6 => Some(Self::WindLeft),
            7 => Some(Self::WindRight),
            _ => None,
        }
    }

    /// Returns the `(dx, dy)` wind drift vector when this cell carries wind.
    #[inline]
    #[must_use]
    pub const fn wind_delta(self) -> Option<(i32, i32)> {
        match self {
            Self::WindUp => Some((0, -1)),
            Self::WindDown => Some((0, 1)),
            Self::WindLeft => Some((-1, 0)),
            Self::WindRight => Some((1, 0)),
            Self::Free | Self::SolidWall | Self::Goal | Self::Trap => None,
        }
    }

    /// Stereo pan hint (`[-1.0, 1.0]`) associated with wind drift direction.
    #[inline]
    #[must_use]
    pub const fn wind_pan(self) -> f32 {
        match self {
            Self::WindLeft => -0.75,
            Self::WindRight => 0.75,
            Self::WindUp => -0.25,
            Self::WindDown => 0.25,
            Self::Free | Self::SolidWall | Self::Goal | Self::Trap => 0.0,
        }
    }
}

/// Sculptable 2D GridWorld MDP supporting solid bounce-back walls and stochastic wind drift.
#[derive(Clone, Debug, PartialEq)]
pub struct SculptedGridMdp {
    /// Grid column count (`2..=20`).
    pub width: i32,
    /// Grid row count (`2..=20`).
    pub height: i32,
    /// Agent spawn cell at the start of each episode.
    pub start: GridState,
    /// Primary goal cell (`GridCellRole::Goal`).
    pub goal: GridState,
    /// Discount factor $\gamma \in [0, 1]$.
    pub gamma: f64,
    /// Probability $p_{\text{wind}} \in [0, 1]$ that a wind cell applies an extra 1-cell drift.
    pub wind_slip_prob: f64,
    cells: Vec<GridCellRole>,
}

impl Default for SculptedGridMdp {
    fn default() -> Self {
        Self::default_with_traps(5, 5, 0.9)
    }
}

impl From<&GridWorldEnv> for SculptedGridMdp {
    #[verified_engine::verified]
    fn from(env: &GridWorldEnv) -> Self {
        let width = env.width.clamp(MIN_RL_GRID_DIM, MAX_RL_GRID_DIM);
        let height = env.height.clamp(MIN_RL_GRID_DIM, MAX_RL_GRID_DIM);
        let mut mdp = Self::new(width, height, env.gamma, 0.0);
        mdp.cells.fill(GridCellRole::Free);
        mdp.start = GridState {
            x: env.start.x.clamp(0, width - 1),
            y: env.start.y.clamp(0, height - 1),
        };
        mdp.goal = GridState {
            x: env.goal.x.clamp(0, width - 1),
            y: env.goal.y.clamp(0, height - 1),
        };
        for trap in &env.traps {
            if mdp.in_bounds(*trap) && *trap != mdp.start && *trap != mdp.goal {
                let idx = mdp.cell_index(*trap);
                mdp.cells[idx] = GridCellRole::Trap;
            }
        }
        let goal_idx = mdp.cell_index(mdp.goal);
        mdp.cells[goal_idx] = GridCellRole::Goal;
        mdp
    }
}

impl SculptedGridMdp {
    /// Creates an empty `width x height` sculpted grid with `start = (0, 0)` and `goal = (width - 1, height - 1)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(width: i32, height: i32, gamma: f64, wind_slip_prob: f64) -> Self {
        let w = width.clamp(MIN_RL_GRID_DIM, MAX_RL_GRID_DIM);
        let h = height.clamp(MIN_RL_GRID_DIM, MAX_RL_GRID_DIM);
        let start = GridState { x: 0, y: 0 };
        let goal = GridState { x: w - 1, y: h - 1 };
        let mut cells = vec![GridCellRole::Free; (w as usize) * (h as usize)];
        cells[flatten_2d_index(goal.x as usize, goal.y as usize, w as usize)] = GridCellRole::Goal;
        let clean_gamma = if gamma.is_finite() {
            gamma.clamp(0.0, 1.0)
        } else {
            0.9
        };
        let clean_wind = if wind_slip_prob.is_finite() {
            wind_slip_prob.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            width: w,
            height: h,
            start,
            goal,
            gamma: clean_gamma,
            wind_slip_prob: clean_wind,
            cells,
        }
    }

    /// Builds the standard baseline GridWorld layout with hazards at `(2, 2)` and `(3, 2)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn default_with_traps(width: i32, height: i32, gamma: f64) -> Self {
        let mut mdp = Self::new(width, height, gamma, 0.0);
        for candidate in [GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }] {
            if mdp.in_bounds(candidate) && candidate != mdp.start && candidate != mdp.goal {
                let idx = mdp.cell_index(candidate);
                mdp.cells[idx] = GridCellRole::Trap;
            }
        }
        mdp
    }

    /// Returns `true` when the grid matches [`Self::default_with_traps`] with zero wind slip.
    #[must_use]
    pub fn is_default_layout(&self) -> bool {
        if self.wind_slip_prob.abs() > 1e-12 {
            return false;
        }
        let baseline = Self::default_with_traps(self.width, self.height, self.gamma);
        self.start == baseline.start && self.goal == baseline.goal && self.cells == baseline.cells
    }

    /// Returns `true` if `state` lies within `0..width` and `0..height`.
    #[inline]
    #[must_use]
    pub const fn in_bounds(&self, state: GridState) -> bool {
        state.x >= 0 && state.x < self.width && state.y >= 0 && state.y < self.height
    }

    /// Computes the row-major flat index for an in-bounds `state`.
    #[inline]
    #[must_use]
    pub fn cell_index(&self, state: GridState) -> usize {
        flatten_2d_index(state.x as usize, state.y as usize, self.width as usize)
    }

    /// Reads the [`GridCellRole`] at `state` (`SolidWall` if out of bounds).
    #[inline]
    #[must_use]
    pub fn cell_role(&self, state: GridState) -> GridCellRole {
        if !self.in_bounds(state) {
            return GridCellRole::SolidWall;
        }
        self.cells[self.cell_index(state)]
    }

    /// Assigns `role` to `state`, keeping `start` passable and maintaining a unique `Goal` cell.
    #[verified_engine::verified]
    pub fn set_cell_role(&mut self, state: GridState, role: GridCellRole) {
        if !self.in_bounds(state) {
            return;
        }
        if state == self.start
            && matches!(
                role,
                GridCellRole::SolidWall | GridCellRole::Goal | GridCellRole::Trap
            )
        {
            return;
        }
        if role == GridCellRole::Goal {
            let old_goal_idx = self.cell_index(self.goal);
            if self.cells[old_goal_idx] == GridCellRole::Goal {
                self.cells[old_goal_idx] = GridCellRole::Free;
            }
            self.goal = state;
            let new_idx = self.cell_index(state);
            self.cells[new_idx] = GridCellRole::Goal;
            return;
        }
        if state == self.goal {
            return;
        }
        let idx = self.cell_index(state);
        self.cells[idx] = role;
    }

    /// Exports the row-major cell mask as `u8` role tags for compact RLE encoding.
    #[must_use]
    pub fn cells_to_u8(&self) -> Vec<u8> {
        self.cells
            .iter()
            .copied()
            .map(GridCellRole::to_u8)
            .collect()
    }

    /// Restores row-major `u8` role tags, validating length, tag bounds, `start`, and `goal`.
    pub fn apply_cells_u8(&mut self, raw: &[u8]) -> Result<(), String> {
        let expected = (self.width as usize) * (self.height as usize);
        if raw.len() != expected {
            return Err(format!("Cell mask length {} != {expected}", raw.len()));
        }
        let mut parsed = Vec::with_capacity(expected);
        let mut found_goal = None;
        for (i, &code) in raw.iter().enumerate() {
            let role = GridCellRole::from_u8(code)
                .ok_or_else(|| format!("Invalid GridCellRole tag {code} at cell {i}"))?;
            if role == GridCellRole::Goal && found_goal.is_none() {
                found_goal = Some(GridState {
                    x: (i % (self.width as usize)) as i32,
                    y: (i / (self.width as usize)) as i32,
                });
                parsed.push(GridCellRole::Goal);
            } else if role == GridCellRole::Goal {
                parsed.push(GridCellRole::Free);
            } else {
                parsed.push(role);
            }
        }
        self.cells = parsed;
        let g = found_goal.unwrap_or(GridState {
            x: self.width - 1,
            y: self.height - 1,
        });
        self.goal = g;
        let g_idx = self.cell_index(g);
        self.cells[g_idx] = GridCellRole::Goal;
        if self.start == self.goal {
            self.start = GridState { x: 0, y: 0 };
        }
        let s_idx = self.cell_index(self.start);
        if matches!(
            self.cells[s_idx],
            GridCellRole::SolidWall | GridCellRole::Trap
        ) {
            self.cells[s_idx] = GridCellRole::Free;
        }
        Ok(())
    }

    #[inline]
    fn move_delta(action: Move) -> (i32, i32) {
        match action {
            Move::Up => (0, -1),
            Move::Down => (0, 1),
            Move::Left => (-1, 0),
            Move::Right => (1, 0),
        }
    }

    #[inline]
    fn step_delta(&self, from: GridState, dx: i32, dy: i32) -> GridState {
        let candidate = GridState {
            x: from.x + dx,
            y: from.y + dy,
        };
        if self.in_bounds(candidate) && self.cell_role(candidate) != GridCellRole::SolidWall {
            candidate
        } else {
            from
        }
    }

    /// Applies `action` without wind slip, bouncing off grid borders and [`GridCellRole::SolidWall`] cells.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn step_deterministic(&self, current_state: GridState, action: Move) -> GridState {
        if !self.in_bounds(current_state)
            || self.is_terminal(&current_state)
            || self.cell_role(current_state) == GridCellRole::SolidWall
        {
            return current_state;
        }
        let (dx, dy) = Self::move_delta(action);
        self.step_delta(current_state, dx, dy)
    }

    /// Returns up to two `(next_state, probability)` outcomes whose probabilities sum to `1.0`.
    #[inline]
    #[must_use]
    pub fn transition_outcomes(
        &self,
        current_state: GridState,
        action: Move,
    ) -> [(GridState, f64); 2] {
        let role = self.cell_role(current_state);
        if !self.in_bounds(current_state)
            || matches!(
                role,
                GridCellRole::Goal | GridCellRole::Trap | GridCellRole::SolidWall
            )
        {
            return [(current_state, 1.0), (current_state, 0.0)];
        }
        let primary = self.step_deterministic(current_state, action);
        let p_wind = self.wind_slip_prob.clamp(0.0, 1.0);
        if let Some((wx, wy)) = role.wind_delta()
            && p_wind > 0.0
        {
            let slipped = if self.is_terminal(&primary) {
                primary
            } else {
                self.step_delta(primary, wx, wy)
            };
            if slipped != primary {
                return [(primary, 1.0 - p_wind), (slipped, p_wind)];
            }
        }
        [(primary, 1.0), (primary, 0.0)]
    }

    /// Collects all deduplicated 1-step reachable states from `state` onto the stack (`<= 9` states).
    #[must_use]
    pub fn reachable_next_states(&self, state: GridState) -> ([GridState; 9], usize) {
        let mut buf = [state; 9];
        let mut len = 1usize;
        for action in ALL_MOVES {
            for (cand, prob) in self.transition_outcomes(state, action) {
                if prob > 0.0 && !buf[..len].contains(&cand) && len < buf.len() {
                    buf[len] = cand;
                    len += 1;
                }
            }
        }
        (buf, len)
    }

    /// Samples one environment transition with zero heap allocations, returning `(next_state, reward, wind_slipped, wind_pan)`.
    #[inline]
    pub fn sample_step(
        &self,
        current_state: GridState,
        action: Move,
        rng: &mut OxidizeRng,
    ) -> (GridState, f64, bool, f32) {
        let role = self.cell_role(current_state);
        let outcomes = self.transition_outcomes(current_state, action);
        let u01 = ((rng.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
        let (next_state, slipped) = if outcomes[1].1 > 0.0 && u01 >= outcomes[0].1 {
            (outcomes[1].0, true)
        } else {
            (outcomes[0].0, false)
        };
        let reward = self.reward(&current_state, &action, &next_state);
        let pan = if slipped { role.wind_pan() } else { 0.0 };
        (next_state, reward, slipped, pan)
    }
}

impl MarkovDecisionProcess for SculptedGridMdp {
    type S = GridState;
    type A = Move;

    #[verified_engine::verified]
    fn transition_probability(
        &self,
        next_state: &Self::S,
        current_state: &Self::S,
        action: &Self::A,
    ) -> f64 {
        let outcomes = self.transition_outcomes(*current_state, *action);
        let mut prob = 0.0;
        for (cand, p) in outcomes {
            if p > 0.0 && cand == *next_state {
                prob += p;
            }
        }
        prob
    }

    #[verified_engine::verified]
    fn reward(&self, current_state: &Self::S, _action: &Self::A, next_state: &Self::S) -> f64 {
        if self.is_terminal(current_state)
            || self.cell_role(*current_state) == GridCellRole::SolidWall
        {
            return 0.0;
        }
        match self.cell_role(*next_state) {
            GridCellRole::Goal => 10.0,
            GridCellRole::Trap => -10.0,
            _ => -0.1,
        }
    }

    #[verified_engine::verified]
    fn actions(&self, state: &Self::S) -> Vec<Self::A> {
        if !self.in_bounds(*state)
            || self.is_terminal(state)
            || self.cell_role(*state) == GridCellRole::SolidWall
        {
            Vec::new()
        } else {
            ALL_MOVES.to_vec()
        }
    }

    #[verified_engine::verified]
    fn discount_factor(&self) -> f64 {
        self.gamma
    }

    #[verified_engine::verified]
    fn is_terminal(&self, state: &Self::S) -> bool {
        matches!(
            self.cell_role(*state),
            GridCellRole::Goal | GridCellRole::Trap
        )
    }
}
