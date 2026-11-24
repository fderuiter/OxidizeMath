use super::{Action, MarkovDecisionProcess, State};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct GridState {
    #[allow(missing_docs)]
    pub x: i32,
    #[allow(missing_docs)]
    pub y: i32,
}

impl State for GridState {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Move {
    #[allow(missing_docs)]
    Up,
    #[allow(missing_docs)]
    Down,
    #[allow(missing_docs)]
    Left,
    #[allow(missing_docs)]
    Right,
}

impl Action for Move {}

#[allow(missing_docs)]
pub struct GridWorldEnv {
    #[allow(missing_docs)]
    pub width: i32,
    #[allow(missing_docs)]
    pub height: i32,
    #[allow(missing_docs)]
    pub goal: GridState,
    #[allow(missing_docs)]
    pub start: GridState,
    #[allow(missing_docs)]
    pub traps: Vec<GridState>,
    /// Impassable wall cells that block agent transitions in [`Self::step`].
    pub walls: Vec<GridState>,
    #[allow(missing_docs)]
    pub gamma: f64,
}

impl GridWorldEnv {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn step(&self, current_state: &GridState, action: &Move) -> GridState {
        let mut expected_next = *current_state;
        match action {
            Move::Up => expected_next.y -= 1,
            Move::Down => expected_next.y += 1,
            Move::Left => expected_next.x -= 1,
            Move::Right => expected_next.x += 1,
        }

        let is_valid = expected_next.x >= 0
            && expected_next.x < self.width
            && expected_next.y >= 0
            && expected_next.y < self.height
            && !self.walls.contains(&expected_next);

        if is_valid {
            expected_next
        } else {
            *current_state
        }
    }
}

impl MarkovDecisionProcess for GridWorldEnv {
    type S = GridState;
    type A = Move;

    #[verified_engine::verified]
    fn transition_probability(
        &self,
        next_state: &Self::S,
        current_state: &Self::S,
        action: &Self::A,
    ) -> f64 {
        let actual_next = self.step(current_state, action);
        if *next_state == actual_next { 1.0 } else { 0.0 }
    }

    #[verified_engine::verified]
    fn reward(&self, _current_state: &Self::S, _action: &Self::A, next_state: &Self::S) -> f64 {
        if *next_state == self.goal {
            10.0
        } else if self.traps.contains(next_state) {
            -10.0
        } else {
            -0.1
        }
    }

    #[verified_engine::verified]
    fn actions(&self, state: &Self::S) -> Vec<Self::A> {
        if self.is_terminal(state) {
            vec![]
        } else {
            vec![Move::Up, Move::Down, Move::Left, Move::Right]
        }
    }

    #[verified_engine::verified]
    fn discount_factor(&self) -> f64 {
        self.gamma
    }

    #[verified_engine::verified]
    fn is_terminal(&self, state: &Self::S) -> bool {
        *state == self.goal || self.traps.contains(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_world_walls_block_transitions_with_zero_allocations() {
        let env = GridWorldEnv {
            width: 5,
            height: 5,
            start: GridState { x: 0, y: 0 },
            goal: GridState { x: 4, y: 4 },
            traps: vec![GridState { x: 2, y: 2 }],
            walls: vec![GridState { x: 1, y: 0 }, GridState { x: 0, y: 1 }],
            gamma: 0.9,
        };
        let origin = GridState { x: 0, y: 0 };
        let (blocked_right, metrics) =
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                env.step(&origin, &Move::Right)
            });
        assert!(metrics.function_calls > 0);
        assert_eq!(blocked_right, origin);
        assert_eq!(env.step(&origin, &Move::Down), origin);
        let free_cell = GridState { x: 2, y: 0 };
        assert_eq!(env.step(&free_cell, &Move::Right), GridState { x: 3, y: 0 });
    }
}
