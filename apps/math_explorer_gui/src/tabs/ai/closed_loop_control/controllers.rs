//! Zero-allocation critic-free `GRPO` Gaussian policy and `QLearningAgent` controllers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::plants::sample_standard_normal;
use super::state::ClosedLoopState;
use math_explorer::ai::reinforcement_learning::{
    DiscreteAction, DiscreteState, EpsilonGreedy, FixedArrayQFunction, QLearningAgent,
};
use math_explorer::applied::grpo::formulas::{
    clipped_surrogate_objective, normalize_group_advantages,
};
use math_explorer::primitives::UnitInterval;
use math_explorer::rng::OxidizeRng;

/// Number of group trajectories sampled per critic-free GRPO episode.
pub const GRPO_GROUP_SIZE: usize = 6;
/// Fixed rollout horizon per trajectory in episodic GRPO and Q-learning updates.
pub const ROLLOUT_HORIZON: usize = 48;
/// Policy feature dimension extracted from the posterior EKF state.
pub const POLICY_DIM: usize = 4;
/// Number of discretized state bins in `FixedArrayQFunction`.
pub const Q_STATE_BINS: usize = 64;
/// Number of discretized control actions in `FixedArrayQFunction`.
pub const Q_ACTION_BINS: usize = 7;

const POLICY_SIGMA: f64 = 0.34;
const DISCRETE_ACTION_LEVELS: [f64; Q_ACTION_BINS] = [-1.0, -0.64, -0.28, 0.0, 0.28, 0.64, 1.0];

/// Critic-free Gaussian policy optimized via `normalize_group_advantages` and `clipped_surrogate_objective`.
#[derive(Debug, Clone)]
pub struct GrpoController {
    /// Linear policy weights $w_\theta \in \mathbb{R}^4$ mapping EKF features $\phi(\hat{x})$ to $\mu_\theta = \tanh(w_\theta^T \phi)$.
    pub weights: [f64; POLICY_DIM],
    /// Preallocated rollout feature histories `[group][step][dim]`.
    pub group_phis: [[[f64; POLICY_DIM]; ROLLOUT_HORIZON]; GRPO_GROUP_SIZE],
    /// Preallocated sampled normalized actions `[group][step]` in `[-1.0, 1.0]`.
    pub group_actions: [[f64; ROLLOUT_HORIZON]; GRPO_GROUP_SIZE],
    /// Cumulative scalar returns $R_g$ for each rollout in the current group.
    pub group_returns: [f64; GRPO_GROUP_SIZE],
    /// Group-normalized relative advantages $A_g$ computed by `normalize_group_advantages`.
    pub group_advantages: [f64; GRPO_GROUP_SIZE],
    pi_old: [f64; GRPO_GROUP_SIZE],
    pi_new: [f64; GRPO_GROUP_SIZE],
    /// Surrogate gain $-\mathcal{L}_{\text{CLIP+KL}}$ before the latest proximal update.
    pub last_surrogate_before: f64,
    /// Surrogate gain $-\mathcal{L}_{\text{CLIP+KL}}$ after the latest proximal update.
    pub last_surrogate_after: f64,
    /// Mean return across the most recent `GRPO_GROUP_SIZE` trajectories.
    pub last_mean_return: f64,
}

impl Default for GrpoController {
    fn default() -> Self {
        Self::new_untrained()
    }
}

impl GrpoController {
    /// Creates an untrained zero-weight GRPO Gaussian policy with preallocated rollout buffers.
    pub const fn new_untrained() -> Self {
        Self {
            weights: [0.0; POLICY_DIM],
            group_phis: [[[0.0; POLICY_DIM]; ROLLOUT_HORIZON]; GRPO_GROUP_SIZE],
            group_actions: [[0.0; ROLLOUT_HORIZON]; GRPO_GROUP_SIZE],
            group_returns: [0.0; GRPO_GROUP_SIZE],
            group_advantages: [0.0; GRPO_GROUP_SIZE],
            pi_old: [1.0; GRPO_GROUP_SIZE],
            pi_new: [1.0; GRPO_GROUP_SIZE],
            last_surrogate_before: 0.0,
            last_surrogate_after: 0.0,
            last_mean_return: 0.0,
        }
    }

    /// Resets policy weights and telemetry to zero in place without heap allocations.
    pub fn reset_untrained(&mut self) {
        self.weights = [0.0; POLICY_DIM];
        self.group_returns = [0.0; GRPO_GROUP_SIZE];
        self.group_advantages = [0.0; GRPO_GROUP_SIZE];
        self.last_surrogate_before = 0.0;
        self.last_surrogate_after = 0.0;
        self.last_mean_return = 0.0;
    }

    /// Evaluates the deterministic greedy policy mean $\mu_\theta(\hat{x}) = \tanh(w_\theta^T \phi(\hat{x})) \in [-1, 1]$.
    pub fn act_deterministic(&self, phi: &[f64; POLICY_DIM]) -> f64 {
        policy_mean(&self.weights, phi)
    }

    /// Samples an exploratory Gaussian action $a_{g,t} \in [-1, 1]$ and records `(phi, action)` in preallocated group buffers.
    pub fn sample_and_record(
        &mut self,
        group_idx: usize,
        step_idx: usize,
        phi: &[f64; POLICY_DIM],
        rng: &mut OxidizeRng,
    ) -> f64 {
        let mu = policy_mean(&self.weights, phi);
        let a = (mu + POLICY_SIGMA * sample_standard_normal(rng)).clamp(-1.0, 1.0);
        self.group_phis[group_idx][step_idx] = *phi;
        self.group_actions[group_idx][step_idx] = a;
        a
    }

    /// Updates `weights` in place from `group_returns` using `normalize_group_advantages` and `clipped_surrogate_objective`.
    pub fn finish_group_update(&mut self, cfg: &ClosedLoopState) {
        normalize_group_advantages(&self.group_returns, &mut self.group_advantages);
        self.last_mean_return = self.group_returns.iter().sum::<f64>() / (GRPO_GROUP_SIZE as f64);

        let old_weights = self.weights;
        for ((pi_slot, phis), acts) in self
            .pi_old
            .iter_mut()
            .zip(self.group_phis.iter())
            .zip(self.group_actions.iter())
        {
            *pi_slot = trajectory_likelihood(&old_weights, phis, acts);
        }
        self.pi_new = self.pi_old;
        let loss_before = clipped_surrogate_objective(
            &self.pi_new,
            &self.pi_old,
            &self.group_advantages,
            cfg.grpo_clip_eps,
            cfg.grpo_kl_beta,
            0.0,
        );
        self.last_surrogate_before = -loss_before;

        let mut candidate = old_weights;
        let mut best_weights = old_weights;
        let mut best_loss = loss_before;

        for _epoch in 0..4 {
            let grad = self.compute_clipped_gradient(&candidate, &old_weights, cfg);
            for (cand_w, &g_d) in candidate.iter_mut().zip(grad.iter()) {
                *cand_w = (*cand_w + cfg.grpo_lr * g_d).clamp(-6.0, 6.0);
            }
            for ((pi_slot, phis), acts) in self
                .pi_new
                .iter_mut()
                .zip(self.group_phis.iter())
                .zip(self.group_actions.iter())
            {
                *pi_slot = trajectory_likelihood(&candidate, phis, acts);
            }
            let kl = mean_gaussian_kl(&candidate, &old_weights, &self.group_phis);
            let loss = clipped_surrogate_objective(
                &self.pi_new,
                &self.pi_old,
                &self.group_advantages,
                cfg.grpo_clip_eps,
                cfg.grpo_kl_beta,
                kl,
            );
            if loss <= best_loss {
                best_loss = loss;
                best_weights = candidate;
            }
        }

        self.weights = best_weights;
        self.last_surrogate_after = -best_loss;
    }

    fn compute_clipped_gradient(
        &self,
        weights: &[f64; POLICY_DIM],
        old_weights: &[f64; POLICY_DIM],
        cfg: &ClosedLoopState,
    ) -> [f64; POLICY_DIM] {
        let inv_var = 1.0 / (POLICY_SIGMA * POLICY_SIGMA);
        let scale = 1.0 / ((GRPO_GROUP_SIZE * ROLLOUT_HORIZON) as f64);
        let mut grad = [0.0; POLICY_DIM];

        for (((phis, acts), &pi_old_g), &adv) in self
            .group_phis
            .iter()
            .zip(self.group_actions.iter())
            .zip(self.pi_old.iter())
            .zip(self.group_advantages.iter())
        {
            let pi_curr = trajectory_likelihood(weights, phis, acts);
            let ratio = (pi_curr / pi_old_g.max(1.0e-12)).clamp(0.05, 20.0);
            let unclipped = (adv > 0.0 && ratio < 1.0 + cfg.grpo_clip_eps)
                || (adv < 0.0 && ratio > 1.0 - cfg.grpo_clip_eps);

            for (phi, &act) in phis.iter().zip(acts.iter()) {
                let mu = policy_mean(weights, phi);
                let mu_old = policy_mean(old_weights, phi);
                let dmu_dz = (1.0 - mu * mu).max(1.0e-4);
                let kl_pull = cfg.grpo_kl_beta * (mu - mu_old) * inv_var * dmu_dz;
                let pg_push = if unclipped {
                    ratio * adv * (act - mu) * inv_var * dmu_dz
                } else {
                    0.0
                };
                let coeff = (pg_push - kl_pull) * scale;
                for (g_slot, &p_val) in grad.iter_mut().zip(phi.iter()) {
                    *g_slot += coeff * p_val;
                }
            }
        }
        grad
    }
}

fn policy_mean(weights: &[f64; POLICY_DIM], phi: &[f64; POLICY_DIM]) -> f64 {
    let mut dot = 0.0;
    for (&w, &p) in weights.iter().zip(phi.iter()) {
        dot += w * p;
    }
    dot.tanh()
}

fn trajectory_likelihood(
    weights: &[f64; POLICY_DIM],
    phis: &[[f64; POLICY_DIM]; ROLLOUT_HORIZON],
    actions: &[f64; ROLLOUT_HORIZON],
) -> f64 {
    let inv_two_var = 0.5 / (POLICY_SIGMA * POLICY_SIGMA);
    let mut avg_log_p = 0.0;
    for (phi, &act) in phis.iter().zip(actions.iter()) {
        let mu = policy_mean(weights, phi);
        let diff = act - mu;
        avg_log_p -= diff * diff * inv_two_var;
    }
    avg_log_p /= ROLLOUT_HORIZON as f64;
    avg_log_p.clamp(-20.0, 0.0).exp()
}

fn mean_gaussian_kl(
    weights: &[f64; POLICY_DIM],
    old_weights: &[f64; POLICY_DIM],
    group_phis: &[[[f64; POLICY_DIM]; ROLLOUT_HORIZON]; GRPO_GROUP_SIZE],
) -> f64 {
    let inv_two_var = 0.5 / (POLICY_SIGMA * POLICY_SIGMA);
    let mut sum = 0.0;
    for traj_phis in group_phis {
        for phi in traj_phis {
            let diff = policy_mean(weights, phi) - policy_mean(old_weights, phi);
            sum += diff * diff * inv_two_var;
        }
    }
    sum / ((GRPO_GROUP_SIZE * ROLLOUT_HORIZON) as f64)
}

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
