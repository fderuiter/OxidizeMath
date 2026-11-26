use super::types::{MarkovDecisionProcess, Policy};

/// Calculates the State-Value Function $V_\pi(s)$.
/// $V_\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_\pi(s')]$
#[verified_engine::verified]
pub fn state_value_bellman_equation<M>(
    mdp: &M,
    policy: &impl Policy<M::S, M::A>,
    state: &M::S,
    next_states: &[M::S],
    v_function: impl Fn(&M::S) -> f64,
) -> f64
where
    M: MarkovDecisionProcess,
{
    let mut value = 0.0;
    let gamma = mdp.discount_factor();

    for action in mdp.actions(state) {
        let prob_action = policy.probability(state, &action);
        let mut expected_return_for_action = 0.0;

        for next_state in next_states {
            let transition_prob = mdp.transition_probability(next_state, state, &action);
            if transition_prob > 0.0 {
                let reward = mdp.reward(state, &action, next_state);
                expected_return_for_action +=
                    transition_prob * (reward + gamma * v_function(next_state));
            }
        }

        value += prob_action * expected_return_for_action;
    }

    value
}

/// Calculates the Action-Value Function $Q_\pi(s, a)$.
/// $Q_\pi(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q_\pi(s', a')]$
/// OR simply: $Q_\pi(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_\pi(s')]$
#[verified_engine::verified]
pub fn action_value_bellman_equation<M>(
    mdp: &M,
    state: &M::S,
    action: &M::A,
    next_states: &[M::S],
    v_function: impl Fn(&M::S) -> f64,
) -> f64
where
    M: MarkovDecisionProcess,
{
    let mut q_value = 0.0;
    let gamma = mdp.discount_factor();

    for next_state in next_states {
        let transition_prob = mdp.transition_probability(next_state, state, action);
        if transition_prob > 0.0 {
            let reward = mdp.reward(state, action, next_state);
            q_value += transition_prob * (reward + gamma * v_function(next_state));
        }
    }

    q_value
}

/// The Bellman Optimality Equation for $V^*(s)$.
/// $V^*(s) = \max_{a} \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^*(s')]$
#[verified_engine::verified]
pub fn bellman_optimality_value<M>(
    mdp: &M,
    state: &M::S,
    next_states: &[M::S],
    v_star: impl Fn(&M::S) -> f64,
) -> f64
where
    M: MarkovDecisionProcess,
{
    let gamma = mdp.discount_factor();
    let actions = mdp.actions(state);

    if actions.is_empty() {
        return 0.0;
    }

    let mut max_value = f64::NEG_INFINITY;

    for action in actions {
        let mut expected_return = 0.0;
        for next_state in next_states {
            let transition_prob = mdp.transition_probability(next_state, state, &action);
            if transition_prob > 0.0 {
                let reward = mdp.reward(state, &action, next_state);
                expected_return += transition_prob * (reward + gamma * v_star(next_state));
            }
        }
        if expected_return > max_value {
            max_value = expected_return;
        }
    }

    // If all paths lead to probability 0 or negative infinity remains, handle gracefully?
    // Usually for valid MDPs max_value will be updated.
    if max_value == f64::NEG_INFINITY {
        0.0
    } else {
        max_value
    }
}
