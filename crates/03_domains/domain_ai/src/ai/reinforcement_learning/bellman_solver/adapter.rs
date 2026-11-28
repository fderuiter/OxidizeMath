//! Zero-allocation [`DiscreteMdp`] adapter over [`SculptedGridMdp`].

// [cite:algorithmic_information_rust]

use super::super::GridState;
use super::super::maze::{GridCellRole, SculptedGridMdp, index_to_move};
use super::super::types::MarkovDecisionProcess;
use super::MAX_RL_GRID_CELLS;
use engine_opt::dp::DiscreteMdp;
use math_commons::math_kernel::types::flatten_2d_index;

pub(super) struct GridDiscreteMdp<'a> {
    mdp: &'a SculptedGridMdp,
    pub(super) width: usize,
    pub(super) count: usize,
}

impl<'a> GridDiscreteMdp<'a> {
    #[inline]
    #[verified_engine::verified]
    pub(super) fn new(mdp: &'a SculptedGridMdp) -> Self {
        let width = mdp.width.max(0) as usize;
        let height = mdp.height.max(0) as usize;
        Self {
            mdp,
            width,
            count: (width * height).min(MAX_RL_GRID_CELLS),
        }
    }

    #[inline]
    #[verified_engine::verified]
    pub(super) fn idx_to_state(&self, state_idx: usize) -> GridState {
        GridState {
            x: (state_idx % self.width) as i32,
            y: (state_idx / self.width) as i32,
        }
    }
}

impl DiscreteMdp for GridDiscreteMdp<'_> {
    #[inline]
    #[verified_engine::verified]
    fn num_states(&self) -> usize {
        self.count
    }

    #[inline]
    #[verified_engine::verified]
    fn num_actions(&self) -> usize {
        4
    }

    #[inline]
    #[verified_engine::verified]
    fn is_terminal(&self, state: usize) -> bool {
        let st = self.idx_to_state(state);
        self.mdp.is_terminal(&st) || self.mdp.cell_role(st) == GridCellRole::SolidWall
    }

    #[verified_engine::verified]
    fn visit_transitions<F>(&self, state: usize, action: usize, mut visitor: F)
    where
        F: FnMut(f64, usize, f64, bool),
    {
        if self.is_terminal(state) {
            return;
        }
        let st = self.idx_to_state(state);
        let act = index_to_move(action);
        for (next_st, prob) in self.mdp.transition_outcomes(st, act) {
            if prob > 0.0 {
                let next_idx = flatten_2d_index(next_st.x as usize, next_st.y as usize, self.width);
                let reward = self.mdp.reward(&st, &act, &next_st);
                visitor(prob, next_idx, reward, self.is_terminal(next_idx));
            }
        }
    }
}
