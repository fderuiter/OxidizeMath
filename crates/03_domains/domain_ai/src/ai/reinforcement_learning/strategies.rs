use super::types::{Action, State};
use math_commons::primitives::UnitInterval;
use oxidize_core::rng::{sample_bounded_usize, sample_closed_open};

/// Strategy for selecting an action based on Q-values.
pub trait ExplorationStrategy<S, A>: Send {
    /// Chooses an action from `available_actions` given aligned `q_values` and `rng`.
    #[verified_engine::verified]
    fn select_action(
        &self,
        state: &S,
        available_actions: &[A],
        q_values: &[f64],
        rng: &mut dyn rand::RngCore,
    ) -> Option<A>
    where
        S: State,
        A: Action;
}

/// Epsilon-Greedy Exploration Strategy.
/// Selects a random action with probability epsilon, and the best action with probability 1-epsilon.
pub struct EpsilonGreedy {
    epsilon: UnitInterval,
}

impl EpsilonGreedy {
    /// Creates an $\epsilon$-greedy policy with exploration probability `epsilon` $\in [0, 1]$.
    #[verified_engine::verified]
    pub fn new(epsilon: UnitInterval) -> Self {
        Self { epsilon }
    }
}

impl<S, A> ExplorationStrategy<S, A> for EpsilonGreedy
where
    S: State,
    A: Action,
{
    #[verified_engine::verified]
    fn select_action(
        &self,
        _state: &S,
        available_actions: &[A],
        q_values: &[f64],
        rng: &mut dyn rand::RngCore,
    ) -> Option<A> {
        if available_actions.is_empty() {
            return None;
        }

        if sample_closed_open::<f64, _>(rng) < self.epsilon.value() {
            let index = sample_bounded_usize(rng, available_actions.len());
            return Some(available_actions[index].clone());
        }

        let bound = available_actions.len().min(q_values.len());
        if bound == 0 {
            let index = sample_bounded_usize(rng, available_actions.len());
            return Some(available_actions[index].clone());
        }

        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (idx, &val) in q_values.iter().take(bound).enumerate() {
            if val.is_finite() && val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        debug_assert!(best_idx < available_actions.len());
        Some(available_actions[best_idx].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::reinforcement_learning::{GridState, Move};
    use oxidize_core::rng::OxidizeRng;

    #[test]
    #[verified_engine::verified]
    fn test_epsilon_greedy_bounds_oversized_nan_and_empty_q_slices() {
        let greedy = EpsilonGreedy::new(UnitInterval::new(0.0).expect("zero eps"));
        let mut rng = OxidizeRng::new(7);
        let state = GridState { x: 0, y: 0 };

        // Oversized q_values where max lies past available_actions.len() must stay in bounds.
        let actions = [Move::Up, Move::Down];
        let oversized_q = [0.2_f64, 0.5, 99.0, 120.0];
        assert_eq!(
            greedy.select_action(&state, &actions, &oversized_q, &mut rng),
            Some(Move::Down)
        );

        // NaN entries must not shadow finite action values or panic.
        let nan_q = [f64::NAN, -0.5_f64];
        assert_eq!(
            greedy.select_action(&state, &actions, &nan_q, &mut rng),
            Some(Move::Down)
        );

        // Empty q_values with non-empty available_actions still returns a valid bounded action.
        let picked = greedy.select_action(&state, &actions, &[], &mut rng);
        assert!(picked == Some(Move::Up) || picked == Some(Move::Down));
        let empty_actions: &[Move] = &[];
        assert_eq!(
            greedy.select_action(&state, empty_actions, &oversized_q, &mut rng),
            None
        );
    }
}
