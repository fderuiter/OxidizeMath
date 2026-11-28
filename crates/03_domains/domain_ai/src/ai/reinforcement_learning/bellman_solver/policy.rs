//! Zero-allocation inline stochastic policy [`TabularGridPolicy`] over [`SculptedGridMdp`](super::super::SculptedGridMdp).

// [cite:algorithmic_information_rust]

use super::super::algorithms::softmax_action_probs;
use super::super::maze::{ALL_MOVES, index_to_move, move_to_index};
use super::super::types::Policy;
use super::super::{GridState, Move};
use super::MAX_RL_GRID_CELLS;
use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Dense 4-action stochastic policy $\pi(a \mid s)$ over a 2D [`SculptedGridMdp`](super::super::SculptedGridMdp) stored inline without heap allocations.
#[derive(Debug)]
pub struct TabularGridPolicy {
    width: usize,
    height: usize,
    probs: [[f64; 4]; MAX_RL_GRID_CELLS],
    sample_seq: AtomicUsize,
}

impl Clone for TabularGridPolicy {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            probs: self.probs,
            sample_seq: AtomicUsize::new(self.sample_seq.load(Ordering::Relaxed)),
        }
    }
}

impl TabularGridPolicy {
    /// Builds a uniform random policy ($\pi(a \mid s) = 0.25$) across `width x height` cells with zero heap allocations.
    #[must_use]
    #[verified_engine::verified]
    pub fn uniform(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            probs: [[0.25; 4]; MAX_RL_GRID_CELLS],
            sample_seq: AtomicUsize::new(0),
        }
    }

    /// Builds a greedy policy with tie-splitting from dense action values $Q(s, a)$ with zero heap allocations.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_q_table(width: usize, height: usize, q_table: &[[f64; 4]]) -> Self {
        let count = (width * height).min(MAX_RL_GRID_CELLS);
        let mut probs = [[0.25; 4]; MAX_RL_GRID_CELLS];
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

    /// Builds a Boltzmann softmax policy $\pi_\theta(a \mid s)$ from dense preference logits with zero heap allocations.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_softmax_logits(width: usize, height: usize, logits: &[[f64; 4]]) -> Self {
        let count = (width * height).min(MAX_RL_GRID_CELLS);
        let mut probs = [[0.25; 4]; MAX_RL_GRID_CELLS];
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
    #[verified_engine::verified]
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
