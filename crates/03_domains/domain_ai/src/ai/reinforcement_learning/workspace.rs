//! Pre-allocated dense RL workspace tables backed by a single [`OwnedByteArena`].

// [cite:algorithmic_information_rust]

use super::maze::{GridCellRole, MAX_RL_GRID_DIM, MIN_RL_GRID_DIM, SculptedGridMdp};
use super::{GridState, Move};
use math_commons::math_kernel::types::flatten_2d_index;
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};
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

#[derive(Copy, Clone, Debug)]
struct DenseRlRegions {
    states: ArenaRegion<GridState>,
    q_table: ArenaRegion<[f64; 4]>,
    theta_logits: ArenaRegion<[f64; 4]>,
    state_values: ArenaRegion<f64>,
    ep_states: ArenaRegion<usize>,
    ep_actions: ArenaRegion<usize>,
    ep_rewards: ArenaRegion<f64>,
}

fn build_dense_rl_arena(count: usize, cap: usize) -> (OwnedByteArena, DenseRlRegions) {
    let mut layout = ArenaLayout::new();
    let states = layout.reserve::<GridState>(count).expect("states layout");
    let q_table = layout.reserve::<[f64; 4]>(count).expect("q_table layout");
    let theta_logits = layout
        .reserve::<[f64; 4]>(count)
        .expect("theta_logits layout");
    let state_values = layout.reserve::<f64>(count).expect("state_values layout");
    let ep_states = layout.reserve::<usize>(cap).expect("ep_states layout");
    let ep_actions = layout.reserve::<usize>(cap).expect("ep_actions layout");
    let ep_rewards = layout.reserve::<f64>(cap).expect("ep_rewards layout");
    let arena = OwnedByteArena::from_layout(&layout);
    (
        arena,
        DenseRlRegions {
            states,
            q_table,
            theta_logits,
            state_values,
            ep_states,
            ep_actions,
            ep_rewards,
        },
    )
}

pub(super) type ReinforceMutSlices<'a> = (
    &'a mut [[f64; 4]],
    &'a mut [[f64; 4]],
    &'a mut [f64],
    &'a [usize],
    &'a [usize],
    &'a [f64],
);

/// Pre-allocated dense RL buffers backed by one [`OwnedByteArena`] for `0`-allocation rollouts.
#[derive(Clone, Debug)]
pub struct DenseRlWorkspace {
    /// Active grid column count.
    pub width: usize,
    /// Active grid row count.
    pub height: usize,
    arena: OwnedByteArena,
    regions: DenseRlRegions,
    ep_len: usize,
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
    /// Allocates all dense RL tables and episode trajectory buffers up front in one [`OwnedByteArena`].
    #[must_use]
    pub fn new(width: usize, height: usize, max_episode_steps: usize, seed: Option<u64>) -> Self {
        let w = width.clamp(MIN_RL_GRID_DIM as usize, MAX_RL_GRID_DIM as usize);
        let h = height.clamp(MIN_RL_GRID_DIM as usize, MAX_RL_GRID_DIM as usize);
        let cap = max_episode_steps.max(4);
        let count = w * h;
        let (mut arena, regions) = build_dense_rl_arena(count, cap);
        let states_slice = arena.region_slice_mut(regions.states);
        for y in 0..h {
            for x in 0..w {
                states_slice[flatten_2d_index(x, y, w)] = GridState {
                    x: x as i32,
                    y: y as i32,
                };
            }
        }
        let rng = seed.map_or_else(OxidizeRng::default, OxidizeRng::new);
        Self {
            width: w,
            height: h,
            arena,
            regions,
            ep_len: 0,
            max_episode_steps: cap,
            current_state: GridState { x: 0, y: 0 },
            episode_steps: 0,
            episode_return: 0.0,
            episodes_completed: 0,
            rng,
        }
    }

    /// Row-major `GridState` lookup table of length `width * height`.
    #[inline]
    #[must_use]
    pub fn states(&self) -> &[GridState] {
        self.arena.region_slice(self.regions.states)
    }

    /// Immutable view of dense action-value table $Q(s, a)$.
    #[inline]
    #[must_use]
    pub fn q_table(&self) -> &[[f64; 4]] {
        self.arena.region_slice(self.regions.q_table)
    }

    /// Mutable view of dense action-value table $Q(s, a)$.
    #[inline]
    pub fn q_table_mut(&mut self) -> &mut [[f64; 4]] {
        self.arena.region_slice_mut(self.regions.q_table)
    }

    /// Immutable view of dense softmax policy preferences $\theta(s, a)$.
    #[inline]
    #[must_use]
    pub fn theta_logits(&self) -> &[[f64; 4]] {
        self.arena.region_slice(self.regions.theta_logits)
    }

    /// Mutable view of dense softmax policy preferences $\theta(s, a)$.
    #[inline]
    pub fn theta_logits_mut(&mut self) -> &mut [[f64; 4]] {
        self.arena.region_slice_mut(self.regions.theta_logits)
    }

    /// Immutable view of critic baseline state values $V(s)$.
    #[inline]
    #[must_use]
    pub fn state_values(&self) -> &[f64] {
        self.arena.region_slice(self.regions.state_values)
    }

    /// Mutable view of critic baseline state values $V(s)$.
    #[inline]
    pub fn state_values_mut(&mut self) -> &mut [f64] {
        self.arena.region_slice_mut(self.regions.state_values)
    }

    /// Active trajectory step count recorded in the current episode.
    #[inline]
    #[must_use]
    pub const fn ep_len(&self) -> usize {
        self.ep_len
    }

    /// Recorded episode state indices for the active episode (`len == ep_len()`).
    #[inline]
    #[must_use]
    pub fn ep_states(&self) -> &[usize] {
        &self.arena.region_slice(self.regions.ep_states)[..self.ep_len]
    }

    /// Recorded episode action indices for the active episode (`len == ep_len()`).
    #[inline]
    #[must_use]
    pub fn ep_actions(&self) -> &[usize] {
        &self.arena.region_slice(self.regions.ep_actions)[..self.ep_len]
    }

    /// Recorded episode scalar rewards for the active episode (`len == ep_len()`).
    #[inline]
    #[must_use]
    pub fn ep_rewards(&self) -> &[f64] {
        &self.arena.region_slice(self.regions.ep_rewards)[..self.ep_len]
    }

    /// Clears recorded episode trajectory slices in O(1) without touching capacity.
    #[inline]
    pub fn clear_episode_trajectory(&mut self) {
        self.ep_len = 0;
    }

    /// Appends `(state_idx, action_idx, reward)` if `ep_len() < max_episode_steps`.
    #[inline]
    pub fn push_episode_step(&mut self, state_idx: usize, action_idx: usize, reward: f64) {
        if self.ep_len < self.max_episode_steps {
            let t = self.ep_len;
            self.arena.region_slice_mut(self.regions.ep_states)[t] = state_idx;
            self.arena.region_slice_mut(self.regions.ep_actions)[t] = action_idx;
            self.arena.region_slice_mut(self.regions.ep_rewards)[t] = reward;
            self.ep_len = t + 1;
        }
    }

    #[inline]
    pub(super) fn split_reinforce_buffers_mut(&mut self) -> ReinforceMutSlices<'_> {
        let len = self.ep_len;
        let regs = self.regions;
        let mut cursor = self.arena.raw_cursor();
        let _states = cursor.region_mut(regs.states);
        let q_table = cursor.region_mut(regs.q_table);
        let theta_logits = cursor.region_mut(regs.theta_logits);
        let state_values = cursor.region_mut(regs.state_values);
        let ep_states = &cursor.region_mut(regs.ep_states)[..len];
        let ep_actions = &cursor.region_mut(regs.ep_actions)[..len];
        let ep_rewards = &cursor.region_mut(regs.ep_rewards)[..len];
        (
            q_table,
            theta_logits,
            state_values,
            ep_states,
            ep_actions,
            ep_rewards,
        )
    }

    /// Clears learned parameters and episode counters in place without reallocating buffers.
    pub fn reset_learning(&mut self, start: GridState) {
        self.q_table_mut().fill([0.0; 4]);
        self.theta_logits_mut().fill([0.0; 4]);
        self.state_values_mut().fill(0.0);
        self.ep_len = 0;
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
                    self.q_table_mut()[idx] = [0.0; 4];
                    self.theta_logits_mut()[idx] = [0.0; 4];
                    self.state_values_mut()[idx] = 0.0;
                }
            }
        }
        if !mdp.in_bounds(self.current_state)
            || mdp.cell_role(self.current_state) == GridCellRole::SolidWall
        {
            self.current_state = mdp.start;
            self.ep_len = 0;
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
        self.q_table()[idx][move_to_index(action)]
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
        self.q_table()[idx]
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
        self.state_values()[flatten_2d_index(state.x as usize, state.y as usize, self.width)]
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
        self.ep_len = 0;
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
        let row = self.q_table()[flatten_2d_index(state.x as usize, state.y as usize, self.width)];
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
            self.theta_logits()[flatten_2d_index(state.x as usize, state.y as usize, self.width)];
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
                .q_table()
                .iter()
                .all(|row| row.iter().all(|v| v.is_finite()))
            && self
                .theta_logits()
                .iter()
                .all(|row| row.iter().all(|v| v.is_finite()))
            && self.state_values().iter().all(|v| v.is_finite())
    }
}
