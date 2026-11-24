use std::fmt::Debug;

/// Represents a state in the Markov Decision Process.
/// $S$: The set of all valid states the environment can be in.
pub trait State: Clone + Debug + PartialEq {}

/// Represents an action in the Markov Decision Process.
/// $A$: The set of all valid actions the agent can take.
pub trait Action: Clone + Debug + PartialEq {}

/// Represents the Markov Decision Process (MDP) tuple $(S, A, P, R, \gamma)$.
pub trait MarkovDecisionProcess {
    /// Concrete state type representing environment configurations.
    type S: State;
    /// Concrete action type representing agent decisions.
    type A: Action;

    /// The Transition Probability $P(s' | s, a)$.
    /// Defines the probability of moving to `next_state` given `current_state` and `action`.
    #[verified_engine::verified]
    fn transition_probability(
        &self,
        next_state: &Self::S,
        current_state: &Self::S,
        action: &Self::A,
    ) -> f64;

    /// The Reward Function $R(s, a, s')$.
    /// The immediate reward received for a transition.
    #[verified_engine::verified]
    fn reward(&self, current_state: &Self::S, action: &Self::A, next_state: &Self::S) -> f64;

    /// Returns the available actions for a given state.
    #[verified_engine::verified]
    fn actions(&self, state: &Self::S) -> Vec<Self::A>;

    /// The Discount Factor $\gamma$.
    /// Scales the importance of future rewards. $0 \le \gamma \le 1$.
    #[verified_engine::verified]
    fn discount_factor(&self) -> f64;

    /// Checks if a state is terminal.
    #[verified_engine::verified]
    fn is_terminal(&self, state: &Self::S) -> bool;
}

/// A Policy $\pi(a|s)$ maps a state to a probability distribution over actions.
pub trait Policy<S: State, A: Action> {
    /// Returns the probability $\pi(a|s)$ of taking `action` in `state`.
    #[verified_engine::verified]
    fn probability(&self, state: &S, action: &A) -> f64;

    /// Samples an action from the policy distribution for the given state.
    #[verified_engine::verified]
    fn sample(&self, state: &S) -> A;
}

/// A Q-Value Function $Q(s, a)$.
/// Represents the expected return of taking action $a$ in state $s$.
///
/// This trait abstracts the storage mechanism, allowing for:
/// - Tabular storage (HashMap)
/// - Linear Approximators
/// - Deep Neural Networks (DQN)
pub trait QFunction<S: State, A: Action> {
    /// Returns the Q-value for a given state-action pair.
    #[verified_engine::verified]
    fn value(&self, state: &S, action: &A) -> f64;

    /// Updates the Q-value for a given state-action pair.
    #[verified_engine::verified]
    fn update(&mut self, state: &S, action: &A, value: f64);
}
