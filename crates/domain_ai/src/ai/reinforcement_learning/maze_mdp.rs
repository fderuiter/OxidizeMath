//! `MarkovDecisionProcess` implementation for [`SculptedGridMdp`].

use super::{ALL_MOVES, GridCellRole, GridState, MarkovDecisionProcess, Move, SculptedGridMdp};

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

    #[verified_engine::verified(alloc_ok = "returns available action vector per trait contract")]
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
