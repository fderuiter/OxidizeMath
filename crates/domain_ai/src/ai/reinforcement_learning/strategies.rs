use super::types::{Action, State};
use math_commons::primitives::UnitInterval;
use rand::Rng;

/// Strategy for selecting an action based on Q-values.
pub trait ExplorationStrategy<S, A>: Send {
    #[verified_engine::verified]
    #[allow(missing_docs)]
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
    #[allow(missing_docs)]
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

        if rng.r#gen::<f64>() < self.epsilon.value() {
            // Explore: Random action
            let index = rng.gen_range(0..available_actions.len());
            Some(available_actions[index].clone())
        } else {
            // Exploit: Best action
            // Find index of max q_value
            let (best_idx, _) = q_values
                .iter()
                .enumerate()
                .max_by(|(_, a_val), (_, b_val)| {
                    a_val
                        .partial_cmp(b_val)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?;
            Some(available_actions[best_idx].clone())
        }
    }
}
