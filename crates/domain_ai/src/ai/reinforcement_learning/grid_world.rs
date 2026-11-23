use super::{Action, MarkovDecisionProcess, State};

/// Discrete 2D integer cell coordinate in [`GridWorldEnv`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridState {
    /// Horizontal column index ($0 \le x < \text{width}$).
    pub x: i32,
    /// Vertical row index ($0 \le y < \text{height}$).
    pub y: i32,
}

impl State for GridState {}

/// Cardinal grid movement action in [`GridWorldEnv`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Move {
    /// Decrements row coordinate `y` by 1.
    Up,
    /// Increments row coordinate `y` by 1.
    Down,
    /// Decrements column coordinate `x` by 1.
    Left,
    /// Increments column coordinate `x` by 1.
    Right,
}

impl Action for Move {}

/// Deterministic bounded 2D grid-world Markov Decision Process with goal and trap cells.
pub struct GridWorldEnv {
    /// Grid width in cells.
    pub width: i32,
    /// Grid height in cells.
    pub height: i32,
    /// Terminal goal state yielding $+10.0$ reward.
    pub goal: GridState,
    /// Initial agent spawn state.
    pub start: GridState,
    /// Terminal hazard states yielding $-10.0$ reward.
    pub traps: Vec<GridState>,
    /// Future-reward discount factor $\gamma \in [0, 1]$.
    pub gamma: f64,
}

impl GridWorldEnv {
    /// Computes the deterministic next [`GridState`] after applying `action`, clamping at grid boundaries.
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
            && expected_next.y < self.height;

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
