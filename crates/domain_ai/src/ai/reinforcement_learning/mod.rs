//! # Reinforcement Learning (RL)
//!
//! A framework for training agents to make sequences of decisions in an environment to maximize cumulative reward.
//!
//! ## Core Concepts
//!
//! The RL problem is modeled as a Markov Decision Process (MDP) defined by the tuple $(S, A, P, R, \gamma)$:
//!
//! - **State ($S$)**: A representation of the environment's current situation.
//! - **Action ($A$)**: A decision made by the agent.
//! - **Transition ($P(s'|s,a)$)**: The probability of moving to state $s'$ given state $s$ and action $a$.
//! - **Reward ($R(s,a,s')$)**: The immediate feedback signal received after a transition.
//! - **Discount Factor ($\gamma$)**: Determines the present value of future rewards.
//!
//! ## The RL Loop
//!
//! The agent interacts with the environment in discrete time steps:
//!
//! ```mermaid
//! graph LR
//!     Agent[ Agent] -- Action $A_t$ --> Env[ Environment]
//!     Env -- State $S_{t+1}$ --> Agent
//!     Env -- Reward $R_{t+1}$ --> Agent
//!
//!     style Agent fill:#f9f,stroke:#333,stroke-width:2px
//!     style Env fill:#bbf,stroke:#333,stroke-width:2px
//! ```
//!
//! 1. The **Agent** observes the current state $S_t$.
//! 2. The **Agent** selects an action $A_t$ based on its policy $\pi$.
//! 3. The **Environment** transitions to a new state $S_{t+1}$ and emits a reward $R_{t+1}$.
//! 4. The **Agent** updates its internal knowledge (e.g., Q-values) based on the transition.
//!
//! ##  Quick Start: GridWorld Q-Learning
//!
//! Train an agent to navigate a simple 1D grid to reach a target.
//!
//! ```rust
//! # use domain_ai::__doctest_facade as math_explorer;
//! use math_explorer::ai::reinforcement_learning::{QLearningAgent, State, Action, TabularQFunction};
//! use math_explorer::primitives::UnitInterval;
//!
//! // 1. Define State and Action
//! #[derive(Clone, Debug, PartialEq, Eq, Hash)]
//! struct GridState(i32);
//! impl State for GridState {}
//!
//! #[derive(Clone, Debug, PartialEq, Eq, Hash)]
//! enum Move { Left, Right }
//! impl Action for Move {}
//!
//! // 2. Initialize Agent
//! // Learning Rate = 0.1, Discount Factor = 0.9, Epsilon = 0.1
//! let mut agent = QLearningAgent::<GridState, Move, TabularQFunction<GridState, Move>>::new(
//!     UnitInterval::new(0.1).unwrap(),
//!     UnitInterval::new(0.9).unwrap(),
//!     UnitInterval::new(0.1).unwrap(),
//!     None,
//! );
//!
//! // 3. Training Loop (Simplified)
//! let state = GridState(0);
//! let target = GridState(3);
//!
//! // Agent chooses an action (e.g., Right)
//! let action = Move::Right;
//!
//! // Environment responds
//! let next_state = GridState(1);
//! let reward = if next_state == target { 10.0 } else { -0.1 };
//!
//! // Agent learns from the experience
//! agent.update(
//!     &state,
//!     &action,
//!     reward,
//!     &next_state,
//!     &[Move::Left, Move::Right] // Possible next actions
//! );
//!
//! // Verify learning: Q-value for (State(0), Right) should increase
//! let q_value = agent.get_q_value(&state, &action);
//! assert!(q_value != 0.0);
//! ```

#[allow(missing_docs)]
pub mod algorithms;
#[allow(missing_docs)]
pub mod bellman;
#[allow(missing_docs)]
pub mod q_function;
#[allow(missing_docs)]
pub mod strategies;
#[allow(missing_docs)]
pub mod types;

// Re-exports for ease of use
pub use algorithms::QLearningAgent;
pub use q_function::TabularQFunction;
pub use strategies::{EpsilonGreedy, ExplorationStrategy};
pub use types::{Action, MarkovDecisionProcess, Policy, QFunction, State};

// [cite:algorithmic_information_rust]
#[allow(missing_docs)]
pub mod grid_world;
pub use grid_world::*;
