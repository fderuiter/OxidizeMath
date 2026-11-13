use super::q_function::TabularQFunction;
use super::strategies::{EpsilonGreedy, ExplorationStrategy};
use super::types::{Action, QFunction, State};
use math_commons::primitives::UnitInterval;

use std::hash::Hash;

/// Q-Learning Update Rule.
/// $Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
#[verified_engine::verified]
pub fn q_learning_update(
    current_q: f64,
    reward: f64,
    max_next_q: f64,
    alpha: UnitInterval,
    gamma: UnitInterval,
) -> f64 {
    let target = reward + gamma.value() * max_next_q;
    current_q + alpha.value() * (target - current_q)
}

/// A generic Q-Learning Agent using the Strategy Pattern for Q-Function storage.
///
/// This agent is decoupled from the storage mechanism, allowing for:
/// - Tabular storage (via `TabularQFunction`)
/// - Function Approximation (e.g., Linear, Neural Networks)
pub struct QLearningAgent<S, A, Q>
where
    S: State,
    A: Action,
    Q: QFunction<S, A>,
{
    #[allow(missing_docs)]
    pub q_func: Q,
    learning_rate: UnitInterval,
    discount_factor: UnitInterval,
    strategy: Box<dyn ExplorationStrategy<S, A> + Send>,
    #[allow(missing_docs)]
    pub rng: oxidize_core::rng::OxidizeRng,
}

/// Legacy alias for the Tabular Q-Learning Agent.
///
/// This preserves backward compatibility for existing code using `TabularQAgent`.
pub type TabularQAgent<S, A> = QLearningAgent<S, A, TabularQFunction<S, A>>;

impl<S, A, Q> QLearningAgent<S, A, Q>
where
    S: State,
    A: Action,
    Q: QFunction<S, A>,
{
    /// Creates a new generic Q-Learning agent. Optionally provide a seed.
    #[verified_engine::verified]
    pub fn new_generic(
        q_func: Q,
        learning_rate: UnitInterval,
        discount_factor: UnitInterval,
        strategy: Box<dyn ExplorationStrategy<S, A> + Send>,
        seed: Option<u64>,
    ) -> Self {
        let rng = if let Some(s) = seed {
            oxidize_core::rng::OxidizeRng::new(s)
        } else {
            oxidize_core::rng::OxidizeRng::default()
        };
        Self {
            q_func,
            learning_rate,
            discount_factor,
            strategy,
            rng,
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn get_q_value(&self, state: &S, action: &A) -> f64 {
        self.q_func.value(state, action)
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn update(
        &mut self,
        state: &S,
        action: &A,
        reward: f64,
        next_state: &S,
        possible_next_actions: &[A],
    ) {
        let current_q = self.get_q_value(state, action);

        let max_next_q = if possible_next_actions.is_empty() {
            0.0
        } else {
            possible_next_actions
                .iter()
                .map(|a| self.get_q_value(next_state, a))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        // Handle case where max_next_q is still NEG_INFINITY (e.g. no entries yet and fold default?)
        // actually fold with NEG_INFINITY on empty list is problematic if not checked, but we check is_empty.
        // If map returns 0.0 for defaults, then max is at least 0.0.

        let new_q = q_learning_update(
            current_q,
            reward,
            if max_next_q == f64::NEG_INFINITY {
                0.0
            } else {
                max_next_q
            },
            self.learning_rate,
            self.discount_factor,
        );

        self.q_func.update(state, action, new_q);
    }

    /// Selects an action using the injected strategy.
    /// This method uses the internal owned RNG.
    #[verified_engine::verified]
    pub fn select_action(&mut self, state: &S, available_actions: &[A]) -> Option<A> {
        let q_values: Vec<f64> = available_actions
            .iter()
            .map(|a| self.get_q_value(state, a))
            .collect();

        self.strategy
            .select_action(state, available_actions, &q_values, &mut self.rng)
    }

    /// Re-seeds the internal RNG of the agent.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = oxidize_core::rng::OxidizeRng::new(seed);
    }
}

// Backward compatibility implementation for TabularQAgent
impl<S, A> QLearningAgent<S, A, TabularQFunction<S, A>>
where
    S: State + Hash + Eq,
    A: Action + Hash + Eq,
{
    /// Creates a new Tabular Q-Agent. Optionally provide a seed.
    #[verified_engine::verified]
    pub fn new(
        learning_rate: UnitInterval,
        discount_factor: UnitInterval,
        epsilon: UnitInterval,
        seed: Option<u64>,
    ) -> Self {
        Self::new_generic(
            TabularQFunction::new(),
            learning_rate,
            discount_factor,
            Box::new(EpsilonGreedy::new(epsilon)),
            seed,
        )
    }

    /// Creates a new Tabular Q-Agent with a custom exploration strategy. Optionally provide a seed.
    #[verified_engine::verified]
    pub fn new_with_strategy(
        learning_rate: UnitInterval,
        discount_factor: UnitInterval,
        strategy: Box<dyn ExplorationStrategy<S, A>>,
        seed: Option<u64>,
    ) -> Self {
        Self::new_generic(
            TabularQFunction::new(),
            learning_rate,
            discount_factor,
            strategy,
            seed,
        )
    }
}

/// Policy Gradient Update (REINFORCE rule sketch).
/// $\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla \ln \pi(a_t | s_t, \theta)$
///
/// This function calculates the gradient component for a single step.
/// * `return_gt`: The cumulative return $G_t$.
/// * `grad_log_pi`: The gradient of the log probability of the action taken.
#[verified_engine::verified]
pub fn policy_gradient_step(
    return_gt: f64,
    grad_log_pi: &[f64],
    learning_rate: UnitInterval,
) -> Vec<f64> {
    grad_log_pi
        .iter()
        .map(|g| learning_rate.value() * return_gt * g)
        .collect()
}
