//! Zero-allocation `GRPO` re-exports from `domain_applied`, `QLearningAgent` wrapper, and receding-horizon `MpcController`.

pub mod mpc;

pub use mpc::MpcController;

use super::state::ClosedLoopState;
use math_explorer::ai::reinforcement_learning::{
    DiscreteAction, DiscreteState, EpsilonGreedy, FixedArrayQFunction, QLearningAgent,
};
pub use math_explorer::applied::grpo::{
    GrpoController, GrpoUpdateParams, GRPO_GROUP_SIZE, ROLLOUT_HORIZON,
};
use math_explorer::primitives::UnitInterval;
use math_explorer::rng::OxidizeRng;

/// Number of discretized state bins in `FixedArrayQFunction`.
pub const Q_STATE_BINS: usize = 64;
/// Number of discretized control actions in `FixedArrayQFunction`.
pub const Q_ACTION_BINS: usize = 7;

const DISCRETE_ACTION_LEVELS: [f64; Q_ACTION_BINS] = [-1.0, -0.64, -0.28, 0.0, 0.28, 0.64, 1.0];

/// Discretized zero-allocation controller backed by `QLearningAgent` and `FixedArrayQFunction<64, 7>`.
pub struct DiscreteQController {
    /// Core `domain_ai` Q-learning agent with stack-inline `64 x 7` Q-table.
    pub agent: QLearningAgent<
        DiscreteState,
        DiscreteAction,
        FixedArrayQFunction<Q_STATE_BINS, Q_ACTION_BINS>,
    >,
    actions: [DiscreteAction; Q_ACTION_BINS],
    q_scratch: [f64; Q_ACTION_BINS],
    /// Cumulative return from the latest Q-learning rollout episode.
    pub last_episode_return: f64,
}

impl DiscreteQController {
    /// Constructs a `DiscreteQController` seeded deterministically from `cfg.rng_seed`.
    pub fn new(cfg: &ClosedLoopState) -> Self {
        let alpha = UnitInterval::new(cfg.q_alpha.clamp(0.01, 0.99)).expect("clamped alpha");
        let gamma = UnitInterval::new(cfg.q_gamma.clamp(0.01, 0.999)).expect("clamped gamma");
        let eps = UnitInterval::new(cfg.q_epsilon.clamp(0.0, 0.99)).expect("clamped epsilon");
        let mut agent = QLearningAgent::new_generic(
            FixedArrayQFunction::new(),
            alpha,
            gamma,
            Box::new(EpsilonGreedy::new(eps)),
            Some(cfg.rng_seed),
        );
        agent.rng.set_stream(2);
        Self {
            agent,
            actions: [
                DiscreteAction(0),
                DiscreteAction(1),
                DiscreteAction(2),
                DiscreteAction(3),
                DiscreteAction(4),
                DiscreteAction(5),
                DiscreteAction(6),
            ],
            q_scratch: [0.0; Q_ACTION_BINS],
            last_episode_return: 0.0,
        }
    }

    /// Resets Q-values and RNG seed in place without heap allocations.
    pub fn reset_untrained(&mut self, cfg: &ClosedLoopState) {
        let alpha = UnitInterval::new(cfg.q_alpha.clamp(0.01, 0.99)).expect("clamped alpha");
        let gamma = UnitInterval::new(cfg.q_gamma.clamp(0.01, 0.999)).expect("clamped gamma");
        self.agent.q_func.reset();
        self.agent.set_learning_params(alpha, gamma);
        self.agent.rng = OxidizeRng::new(cfg.rng_seed).with_stream(2);
        self.last_episode_return = 0.0;
    }

    /// Selects an exploratory action via `QLearningAgent::select_action_into` with zero heap allocations.
    pub fn select_exploratory_action(&mut self, state_bin: usize) -> (DiscreteAction, f64) {
        let state = DiscreteState(state_bin.min(Q_STATE_BINS - 1));
        let picked = self
            .agent
            .select_action_into(&state, &self.actions, &mut self.q_scratch)
            .unwrap_or(DiscreteAction(3));
        (
            picked,
            DISCRETE_ACTION_LEVELS[picked.0.min(Q_ACTION_BINS - 1)],
        )
    }

    /// Selects the greedy argmax action $\arg\max_a Q(s, a)$ during live closed-loop evaluation.
    pub fn act_greedy(&self, state_bin: usize) -> f64 {
        let state = DiscreteState(state_bin.min(Q_STATE_BINS - 1));
        let mut best_idx = 3;
        let mut best_q = f64::NEG_INFINITY;
        for (idx, act) in self.actions.iter().enumerate() {
            let q = self.agent.get_q_value(&state, act);
            if q > best_q {
                best_q = q;
                best_idx = idx;
            }
        }
        DISCRETE_ACTION_LEVELS[best_idx]
    }

    /// Applies the Bellman Q-learning update in place without heap allocations.
    pub fn update_transition(
        &mut self,
        state_bin: usize,
        action: DiscreteAction,
        reward: f64,
        next_bin: usize,
    ) {
        let s = DiscreteState(state_bin.min(Q_STATE_BINS - 1));
        let ns = DiscreteState(next_bin.min(Q_STATE_BINS - 1));
        self.agent.update(&s, &action, reward, &ns, &self.actions);
    }

    /// Checks that every entry in the `64 x 7` Q-table is finite.
    pub fn is_finite(&self) -> bool {
        self.agent
            .q_func
            .as_table()
            .iter()
            .all(|row| row.iter().all(|v| v.is_finite()))
    }
}

// [cite:grpo]
