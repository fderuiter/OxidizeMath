//! Pre-allocated dense RL workspace tables for zero-allocation tabular and softmax rollouts.

// [cite:algorithmic_information_rust]

use super::maze::{GridCellRole, MAX_RL_GRID_DIM, MIN_RL_GRID_DIM, SculptedGridMdp};
use super::{GridState, Move};
use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Canonical 4-action ordering `[Up, Down, Left, Right]` used by dense RL tables.
pub const ALL_MOVES: [Move; 4] = [Move::Up, Move::Down, Move::Left, Move::Right];

/// Maps a [`Move`] action to its dense table index in `0..4`.
#[inline]
#[must_use]
pub const fn move_to_index(action: Move) -> usize {
    match action {
        Move::Up => 0,
        Move::Down => 1,
        Move::Left => 2,
        Move::Right => 3,
    }
}

/// Maps a dense table index `0..4` back to its [`Move`] action.
#[inline]
#[must_use]
pub const fn index_to_move(idx: usize) -> Move {
    match idx & 3 {
        0 => Move::Up,
        1 => Move::Down,
        2 => Move::Left,
        _ => Move::Right,
    }
}

/// Pre-allocated dense RL buffers enabling `0`-allocation Q-learning and policy-gradient rollouts.
#[derive(Clone, Debug)]
pub struct DenseRlWorkspace {
    /// Active grid column count.
    pub width: usize,
    /// Active grid row count.
    pub height: usize,
    /// Row-major `GridState` lookup table of length `width * height`.
    pub states: Vec<GridState>,
    /// Dense action-value table $Q(s, a)$ indexed by `flatten_2d_index(x, y, width)` and `move_to_index(a)`.
    pub q_table: Vec<[f64; 4]>,
    /// Dense softmax policy preferences $\theta(s, a)$ for REINFORCE policy gradients.
    pub theta_logits: Vec<[f64; 4]>,
    /// Baseline state-value critic $V(s)$ used for low-variance REINFORCE advantage estimation.
    pub state_values: Vec<f64>,
    /// Pre-allocated episode state-index trajectory buffer.
    pub ep_states: Vec<usize>,
    /// Pre-allocated episode action-index trajectory buffer.
    pub ep_actions: Vec<usize>,
    /// Pre-allocated episode reward trajectory buffer.
    pub ep_rewards: Vec<f64>,
    /// Pre-allocated per-state policy-gradient accumulation scratch buffer for batch episodic REINFORCE.
    pub(crate) pg_delta_buf: Vec<[f64; 4]>,
    /// Pre-allocated per-state critic accumulation scratch buffer for batch episodic REINFORCE.
    pub(crate) critic_delta_buf: Vec<f64>,
    /// Maximum steps per episode before automatic episode truncation (`>= 4`).
    pub max_episode_steps: usize,
    /// Current agent grid coordinate.
    pub current_state: GridState,
    /// Steps elapsed in the active episode.
    pub episode_steps: usize,
    /// Undiscounted cumulative reward in the active episode.
    pub episode_return: f64,
    /// Total completed episodes since the last reset.
    pub episodes_completed: u32,
    /// Deterministic PRNG used by exploration and stochastic wind sampling.
    pub rng: OxidizeRng,
}

impl DenseRlWorkspace {
    /// Allocates all dense RL tables and episode trajectory buffers up front.
    #[must_use]
    pub fn new(width: usize, height: usize, max_episode_steps: usize, seed: Option<u64>) -> Self {
        let w = width.clamp(MIN_RL_GRID_DIM as usize, MAX_RL_GRID_DIM as usize);
        let h = height.clamp(MIN_RL_GRID_DIM as usize, MAX_RL_GRID_DIM as usize);
        let cap = max_episode_steps.max(4);
        let count = w * h;
        let mut states = vec![GridState { x: 0, y: 0 }; count];
        for y in 0..h {
            for x in 0..w {
                states[flatten_2d_index(x, y, w)] = GridState {
                    x: x as i32,
                    y: y as i32,
                };
            }
        }
        let rng = seed.map_or_else(OxidizeRng::default, OxidizeRng::new);
        Self {
            width: w,
            height: h,
            states,
            q_table: vec![[0.0; 4]; count],
            theta_logits: vec![[0.0; 4]; count],
            state_values: vec![0.0; count],
            ep_states: Vec::with_capacity(cap),
            ep_actions: Vec::with_capacity(cap),
            ep_rewards: Vec::with_capacity(cap),
            pg_delta_buf: vec![[0.0; 4]; count],
            critic_delta_buf: vec![0.0; count],
            max_episode_steps: cap,
            current_state: GridState { x: 0, y: 0 },
            episode_steps: 0,
            episode_return: 0.0,
            episodes_completed: 0,
            rng,
        }
    }

    /// Clears learned parameters and episode counters in place without reallocating buffers.
    pub fn reset_learning(&mut self, start: GridState) {
        self.q_table.fill([0.0; 4]);
        self.theta_logits.fill([0.0; 4]);
        self.state_values.fill(0.0);
        self.ep_states.clear();
        self.ep_actions.clear();
        self.ep_rewards.clear();
        self.pg_delta_buf.fill([0.0; 4]);
        self.critic_delta_buf.fill(0.0);
        self.current_state = start;
        self.episode_steps = 0;
        self.episode_return = 0.0;
        self.episodes_completed = 0;
    }

    /// Synchronizes workspace dimensions and keeps `current_state` on a valid passable cell.
    pub fn sync_topology(&mut self, mdp: &SculptedGridMdp) {
        let w = mdp.width as usize;
        let h = mdp.height as usize;
        if w != self.width || h != self.height {
            let seed_next = self.rng.next_u64();
            *self = Self::new(w, h, self.max_episode_steps, Some(seed_next));
            self.current_state = mdp.start;
            return;
        }
        for y in 0..h {
            for x in 0..w {
                let st = GridState {
                    x: x as i32,
                    y: y as i32,
                };
                if mdp.cell_role(st) == GridCellRole::SolidWall {
                    let idx = flatten_2d_index(x, y, w);
                    self.q_table[idx] = [0.0; 4];
                    self.theta_logits[idx] = [0.0; 4];
                    self.state_values[idx] = 0.0;
                }
            }
        }
        if !mdp.in_bounds(self.current_state)
            || mdp.cell_role(self.current_state) == GridCellRole::SolidWall
        {
            self.current_state = mdp.start;
            self.ep_states.clear();
            self.ep_actions.clear();
            self.ep_rewards.clear();
            self.episode_steps = 0;
            self.episode_return = 0.0;
        }
    }

    /// Reads $Q(s, a)$ from the dense table (`0.0` if out of bounds).
    #[inline]
    #[must_use]
    pub fn q_value(&self, state: GridState, action: Move) -> f64 {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return 0.0;
        }
        let idx = flatten_2d_index(state.x as usize, state.y as usize, self.width);
        self.q_table[idx][move_to_index(action)]
    }

    /// Returns $\max_a Q(s, a)$ for `state`.
    #[inline]
    #[must_use]
    pub fn max_q(&self, state: GridState) -> f64 {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return 0.0;
        }
        let idx = flatten_2d_index(state.x as usize, state.y as usize, self.width);
        self.q_table[idx]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Alias returning $\max_a Q(s, a)$ for `state`.
    #[inline]
    #[must_use]
    pub fn max_q_at(&self, state: GridState) -> f64 {
        self.max_q(state)
    }

    /// Returns the critic state-value estimate $V(s)$ for `state`.
    #[inline]
    #[must_use]
    pub fn state_value_at(&self, state: GridState) -> f64 {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return 0.0;
        }
        self.state_values[flatten_2d_index(state.x as usize, state.y as usize, self.width)]
    }

    /// Resizes workspace buffers if dimensions changed and clears all learned parameters.
    pub fn resize_and_clear(&mut self, width: usize, height: usize) {
        let w = width.clamp(MIN_RL_GRID_DIM as usize, MAX_RL_GRID_DIM as usize);
        let h = height.clamp(MIN_RL_GRID_DIM as usize, MAX_RL_GRID_DIM as usize);
        if w != self.width || h != self.height {
            let seed_next = self.rng.next_u64();
            *self = Self::new(w, h, self.max_episode_steps, Some(seed_next));
        } else {
            self.reset_learning(GridState { x: 0, y: 0 });
        }
    }

    /// Resets per-episode trajectory buffers and places `current_state` at `mdp.start`.
    pub fn reset_episode(&mut self, mdp: &SculptedGridMdp) {
        self.sync_topology(mdp);
        self.current_state = mdp.start;
        self.ep_states.clear();
        self.ep_actions.clear();
        self.ep_rewards.clear();
        self.episode_steps = 0;
        self.episode_return = 0.0;
    }

    /// Returns the greedy $\arg\max_a Q(s, a)$ action when learned values are non-uniform.
    #[must_use]
    pub fn greedy_q_action(&self, state: GridState) -> Option<Move> {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return None;
        }
        let row = self.q_table[flatten_2d_index(state.x as usize, state.y as usize, self.width)];
        if row.iter().all(|v| v.abs() <= 1e-12) {
            return None;
        }
        let mut best = 0usize;
        for i in 1..4 {
            if row[i] > row[best] {
                best = i;
            }
        }
        Some(index_to_move(best))
    }

    /// Returns the greedy $\arg\max_a \theta(s, a)$ softmax policy action when preferences are non-uniform.
    #[must_use]
    pub fn greedy_policy_action(&self, state: GridState) -> Option<Move> {
        if state.x < 0
            || state.y < 0
            || (state.x as usize) >= self.width
            || (state.y as usize) >= self.height
        {
            return None;
        }
        let row =
            self.theta_logits[flatten_2d_index(state.x as usize, state.y as usize, self.width)];
        if row.iter().all(|v| v.abs() <= 1e-12) {
            return None;
        }
        let mut best = 0usize;
        for i in 1..4 {
            if row[i] > row[best] {
                best = i;
            }
        }
        Some(index_to_move(best))
    }

    /// Returns `true` when all learned Q-values, policy logits, and critic values are finite.
    #[must_use]
    pub fn is_finite_state(&self) -> bool {
        self.episode_return.is_finite()
            && self
                .q_table
                .iter()
                .all(|row| row.iter().all(|v| v.is_finite()))
            && self
                .theta_logits
                .iter()
                .all(|row| row.iter().all(|v| v.is_finite()))
            && self.state_values.iter().all(|v| v.is_finite())
    }
}
