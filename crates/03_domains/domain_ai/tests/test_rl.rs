//! Integration tests for test rl.
#[cfg(test)]
mod tests {
    use domain_ai::ai::reinforcement_learning::algorithms::TabularQAgent;
    use domain_ai::ai::reinforcement_learning::bellman::state_value_bellman_equation;
    use domain_ai::ai::reinforcement_learning::types::{
        Action, MarkovDecisionProcess, Policy, State,
    };
    use math_commons::primitives::UnitInterval;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum GridState {
        Start,
        Path,
        Goal,
        Trap,
    }

    impl State for GridState {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum Move {
        Forward,
        Stay,
    }

    impl Action for Move {}

    struct SimpleGridWorld {
        gamma: f64,
    }

    impl MarkovDecisionProcess for SimpleGridWorld {
        type S = GridState;
        type A = Move;

        #[verified_engine::verified]
        fn transition_probability(
            &self,
            next_state: &Self::S,
            current_state: &Self::S,
            action: &Self::A,
        ) -> f64 {
            match (current_state, action, next_state) {
                (GridState::Start, Move::Forward, GridState::Path) => 1.0,
                (GridState::Start, Move::Stay, GridState::Start) => 1.0,
                (GridState::Path, Move::Forward, GridState::Goal) => 0.8, // Stochasticity
                (GridState::Path, Move::Forward, GridState::Trap) => 0.2,
                (GridState::Path, Move::Stay, GridState::Path) => 1.0,
                (GridState::Goal, _, GridState::Goal) => 1.0, // Terminal
                (GridState::Trap, _, GridState::Trap) => 1.0, // Terminal
                _ => 0.0,
            }
        }

        #[verified_engine::verified]
        fn reward(&self, current_state: &Self::S, _action: &Self::A, next_state: &Self::S) -> f64 {
            if *current_state == GridState::Path && *next_state == GridState::Goal {
                10.0
            } else if *current_state == GridState::Path && *next_state == GridState::Trap {
                -10.0
            } else {
                0.0
            }
        }

        #[verified_engine::verified]
        fn actions(&self, state: &Self::S) -> Vec<Self::A> {
            match state {
                GridState::Goal | GridState::Trap => vec![],
                _ => vec![Move::Forward, Move::Stay],
            }
        }

        #[verified_engine::verified]
        fn discount_factor(&self) -> f64 {
            self.gamma
        }

        #[verified_engine::verified]
        fn is_terminal(&self, state: &Self::S) -> bool {
            matches!(state, GridState::Goal | GridState::Trap)
        }
    }

    struct RandomPolicy;
    impl Policy<GridState, Move> for RandomPolicy {
        #[verified_engine::verified]
        fn probability(&self, _state: &GridState, _action: &Move) -> f64 {
            0.5
        }
        #[verified_engine::verified]
        fn sample(&self, _state: &GridState) -> Move {
            Move::Forward // Simplified
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_bellman_equations() {
        let env = SimpleGridWorld { gamma: 0.9 };
        let policy = RandomPolicy;

        // Mock Value Function V(s)
        let v_func = |s: &GridState| match s {
            GridState::Goal => 0.0, // Terminal
            GridState::Trap => 0.0, // Terminal
            _ => 1.0,               // Arbitrary guess
        };

        // Test V(s) calculation for Start
        // Actions: Forward -> Path (prob 1.0), Stay -> Start (prob 1.0)
        // V(Start) = 0.5 * [P(Path|Start, Fwd)*(R + g*V(Path))] + 0.5 * [P(Start|Start, Stay)*(R + g*V(Start))]
        //          = 0.5 * [1.0 * (0 + 0.9*1.0)] + 0.5 * [1.0 * (0 + 0.9*1.0)]
        //          = 0.5 * 0.9 + 0.5 * 0.9 = 0.9
        let v_start = state_value_bellman_equation(
            &env,
            &policy,
            &GridState::Start,
            &[
                GridState::Start,
                GridState::Path,
                GridState::Goal,
                GridState::Trap,
            ],
            v_func,
        );

        assert!(
            (v_start - 0.9).abs() < math_commons::registry::TOLERANCE_FAST,
            "V(Start) should be 0.9, got {}",
            v_start
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_q_learning_agent() {
        let mut agent = TabularQAgent::new(
            UnitInterval::new(0.1).unwrap(),
            UnitInterval::new(0.9).unwrap(),
            UnitInterval::new(0.1).unwrap(),
            None,
        );
        let state = GridState::Start;
        let action = Move::Forward;
        let next_state = GridState::Path;
        let reward = 0.0;

        // Q(Start, Forward) init 0.0
        // Update: Q += alpha * (R + gamma * max_a' Q(next, a') - Q)
        // Next state actions: Forward, Stay. Q values are 0.0. Max is 0.0.
        // Q += 0.1 * (0.0 + 0.9 * 0.0 - 0.0) = 0.0

        agent.update(
            &state,
            &action,
            reward,
            &next_state,
            &[Move::Forward, Move::Stay],
        );
        assert_eq!(agent.get_q_value(&state, &action), 0.0);

        // Now assume we found gold at the next step
        agent.update(
            &GridState::Path,
            &Move::Forward,
            10.0,
            &GridState::Goal,
            &[],
        );
        // Q(Path, Fwd) += 0.1 * (10.0 + 0.9*0 - 0) = 1.0
        assert!(
            (agent.get_q_value(&GridState::Path, &Move::Forward) - 1.0).abs()
                < math_commons::registry::TOLERANCE_FAST
        );

        // Now update Start again
        // Max Q(Path) is now 1.0 (from Forward action)
        // Q(Start, Fwd) += 0.1 * (0.0 + 0.9 * 1.0 - 0.0) = 0.09
        agent.update(
            &state,
            &action,
            reward,
            &next_state,
            &[Move::Forward, Move::Stay],
        );
        assert!(
            (agent.get_q_value(&state, &action) - 0.09).abs()
                < math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_deterministic_action_selection() {
        // Epsilon 0.5 to trigger both exploration and exploitation
        let mut agent1 = TabularQAgent::new(
            UnitInterval::new(0.1).unwrap(),
            UnitInterval::new(0.9).unwrap(),
            UnitInterval::new(0.5).unwrap(),
            Some(42),
        );
        let mut agent2 = TabularQAgent::new(
            UnitInterval::new(0.1).unwrap(),
            UnitInterval::new(0.9).unwrap(),
            UnitInterval::new(0.5).unwrap(),
            Some(42),
        );

        let state = GridState::Start;
        let actions = [Move::Forward, Move::Stay];

        for _ in 0..20 {
            let action1 = agent1.select_action(&state, &actions);
            let action2 = agent2.select_action(&state, &actions);
            assert_eq!(
                action1, action2,
                "Actions should be identical for same seed"
            );
        }
    }
}
