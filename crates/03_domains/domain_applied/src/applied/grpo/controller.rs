//! Zero-allocation critic-free Gaussian `GRPO` policy controller.
//!
//! Couples `normalize_group_advantages` and `clipped_surrogate_objective` over fixed-horizon
//! trajectory groups without allocating on the heap during rollouts or proximal updates.

use super::formulas::{clipped_surrogate_objective, normalize_group_advantages};
use oxidize_core::rng::{OxidizeRng, sample_standard_normal_f64};

/// Number of group trajectories sampled per critic-free GRPO episode.
pub const GRPO_GROUP_SIZE: usize = 6;
/// Fixed rollout horizon per trajectory in episodic GRPO updates.
pub const ROLLOUT_HORIZON: usize = 48;
/// Policy feature dimension extracted from posterior state estimates.
pub const POLICY_DIM: usize = 4;

const POLICY_SIGMA: f64 = 0.34;

/// Hyperparameters governing one proximal group-relative policy update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrpoUpdateParams {
    /// Proximal gradient ascent step size $\eta > 0$.
    pub lr: f64,
    /// PPO-style probability-ratio clipping radius $\epsilon > 0$.
    pub clip_eps: f64,
    /// Reference-policy Gaussian KL penalty weight $\beta_{\text{KL}} \ge 0$.
    pub kl_beta: f64,
}

impl Default for GrpoUpdateParams {
    fn default() -> Self {
        Self {
            lr: 0.14,
            clip_eps: 0.20,
            kl_beta: 0.02,
        }
    }
}

/// Critic-free Gaussian policy optimized via [`normalize_group_advantages`] and [`clipped_surrogate_objective`].
///
/// # Examples
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::grpo::{
///     GrpoController, GrpoUpdateParams, GRPO_GROUP_SIZE, ROLLOUT_HORIZON,
/// };
/// use oxidize_core::rng::OxidizeRng;
///
/// let mut ctrl = GrpoController::new_untrained();
/// let mut rng = OxidizeRng::new(7);
/// let phi = [0.5, -0.2, 0.1, 1.0];
/// for g in 0..GRPO_GROUP_SIZE {
///     let mut ret = 0.0;
///     for t in 0..ROLLOUT_HORIZON {
///         let a = ctrl.sample_and_record(g, t, &phi, &mut rng);
///         ret += -(a - 0.4).powi(2);
///     }
///     ctrl.group_returns[g] = ret;
/// }
/// ctrl.finish_group_update(GrpoUpdateParams::default());
/// assert!(ctrl.last_surrogate_after >= ctrl.last_surrogate_before - 1.0e-12);
/// ```
#[derive(Debug, Clone)]
pub struct GrpoController {
    /// Linear policy weights $w_\theta \in \mathbb{R}^4$ mapping features $\phi(\hat{x})$ to $\mu_\theta = \tanh(w_\theta^T \phi)$.
    pub weights: [f64; POLICY_DIM],
    /// Preallocated rollout feature histories `[group][step][dim]`.
    pub group_phis: [[[f64; POLICY_DIM]; ROLLOUT_HORIZON]; GRPO_GROUP_SIZE],
    /// Preallocated sampled normalized actions `[group][step]` in `[-1.0, 1.0]`.
    pub group_actions: [[f64; ROLLOUT_HORIZON]; GRPO_GROUP_SIZE],
    /// Cumulative scalar returns $R_g$ for each rollout in the current group.
    pub group_returns: [f64; GRPO_GROUP_SIZE],
    /// Group-normalized relative advantages $A_g$ computed by [`normalize_group_advantages`].
    pub group_advantages: [f64; GRPO_GROUP_SIZE],
    pi_old: [f64; GRPO_GROUP_SIZE],
    pi_new: [f64; GRPO_GROUP_SIZE],
    /// Surrogate gain $-\mathcal{L}_{\text{CLIP+KL}}$ before the latest proximal update.
    pub last_surrogate_before: f64,
    /// Surrogate gain $-\mathcal{L}_{\text{CLIP+KL}}$ after the latest proximal update.
    pub last_surrogate_after: f64,
    /// Mean return across the most recent [`GRPO_GROUP_SIZE`] trajectories.
    pub last_mean_return: f64,
}

impl Default for GrpoController {
    fn default() -> Self {
        Self::new_untrained()
    }
}

impl GrpoController {
    /// Creates an untrained zero-weight GRPO Gaussian policy with preallocated rollout buffers.
    #[must_use]
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
    #[verified_engine::verified]
    pub fn reset_untrained(&mut self) {
        self.weights = [0.0; POLICY_DIM];
        self.group_returns = [0.0; GRPO_GROUP_SIZE];
        self.group_advantages = [0.0; GRPO_GROUP_SIZE];
        self.last_surrogate_before = 0.0;
        self.last_surrogate_after = 0.0;
        self.last_mean_return = 0.0;
        debug_assert!(self.last_mean_return == 0.0);
    }

    /// Evaluates the deterministic greedy policy mean $\mu_\theta(\hat{x}) = \tanh(w_\theta^T \phi(\hat{x})) \in [-1, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn act_deterministic(&self, phi: &[f64; POLICY_DIM]) -> f64 {
        let act = policy_mean(&self.weights, phi);
        debug_assert!((-1.0..=1.0).contains(&act));
        act
    }

    /// Samples an exploratory Gaussian action $a_{g,t} \in [-1, 1]$ and records `(phi, action)` in preallocated group buffers.
    #[verified_engine::verified]
    pub fn sample_and_record(
        &mut self,
        group_idx: usize,
        step_idx: usize,
        phi: &[f64; POLICY_DIM],
        rng: &mut OxidizeRng,
    ) -> f64 {
        debug_assert!(group_idx < GRPO_GROUP_SIZE);
        debug_assert!(step_idx < ROLLOUT_HORIZON);
        let g = group_idx.min(GRPO_GROUP_SIZE - 1);
        let s = step_idx.min(ROLLOUT_HORIZON - 1);
        let mu = policy_mean(&self.weights, phi);
        let a = (mu + POLICY_SIGMA * sample_standard_normal_f64(rng)).clamp(-1.0, 1.0);
        self.group_phis[g][s] = *phi;
        self.group_actions[g][s] = a;
        debug_assert!((-1.0..=1.0).contains(&a));
        a
    }

    /// Updates `weights` in place from `group_returns` using [`normalize_group_advantages`] and [`clipped_surrogate_objective`].
    #[verified_engine::verified]
    pub fn finish_group_update(&mut self, params: GrpoUpdateParams) {
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
            params.clip_eps,
            params.kl_beta,
            0.0,
        );
        self.last_surrogate_before = -loss_before;

        let mut candidate = old_weights;
        let mut best_weights = old_weights;
        let mut best_loss = loss_before;

        for _epoch in 0..4 {
            let grad = self.compute_clipped_gradient(&candidate, &old_weights, params);
            for (cand_w, &g_d) in candidate.iter_mut().zip(grad.iter()) {
                *cand_w = (*cand_w + params.lr * g_d).clamp(-6.0, 6.0);
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
                params.clip_eps,
                params.kl_beta,
                kl,
            );
            if loss <= best_loss {
                best_loss = loss;
                best_weights = candidate;
            }
        }

        self.weights = best_weights;
        self.last_surrogate_after = -best_loss;
        debug_assert!(self.last_surrogate_after >= self.last_surrogate_before - 1.0e-12);
    }

    #[verified_engine::verified]
    fn compute_clipped_gradient(
        &self,
        weights: &[f64; POLICY_DIM],
        old_weights: &[f64; POLICY_DIM],
        params: GrpoUpdateParams,
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
            let unclipped = (adv > 0.0 && ratio < 1.0 + params.clip_eps)
                || (adv < 0.0 && ratio > 1.0 - params.clip_eps);

            for (phi, &act) in phis.iter().zip(acts.iter()) {
                let mu = policy_mean(weights, phi);
                let mu_old = policy_mean(old_weights, phi);
                let dmu_dz = (1.0 - mu * mu).max(1.0e-4);
                let kl_pull = params.kl_beta * (mu - mu_old) * inv_var * dmu_dz;
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
        debug_assert!(grad.iter().all(|g| g.is_finite()));
        grad
    }
}

#[verified_engine::verified]
fn policy_mean(weights: &[f64; POLICY_DIM], phi: &[f64; POLICY_DIM]) -> f64 {
    let mut dot = 0.0;
    for (&w, &p) in weights.iter().zip(phi.iter()) {
        dot += w * p;
    }
    let out = dot.tanh();
    debug_assert!((-1.0..=1.0).contains(&out));
    out
}

#[verified_engine::verified]
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
    let p = avg_log_p.clamp(-20.0, 0.0).exp();
    debug_assert!(p > 0.0 && p <= 1.0);
    p
}

#[verified_engine::verified]
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
    let kl = sum / ((GRPO_GROUP_SIZE * ROLLOUT_HORIZON) as f64);
    debug_assert!(kl >= 0.0 && kl.is_finite());
    kl
}

// [cite:grpo]

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_grpo_controller_zero_alloc_rollout_and_update() {
        let mut ctrl = GrpoController::new_untrained();
        let mut rng = OxidizeRng::new(42);
        let phi = [0.6, -0.3, 0.2, 1.0];
        let params = GrpoUpdateParams::default();

        {
            let _guard = crate::VERIFIED_ENGINE_TEST_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            VerifiedEngine::run_verified(|| {
                for g in 0..GRPO_GROUP_SIZE {
                    let mut ret = 0.0;
                    for t in 0..ROLLOUT_HORIZON {
                        let a = ctrl.sample_and_record(g, t, &phi, &mut rng);
                        ret += -(a - 0.5).powi(2);
                    }
                    ctrl.group_returns[g] = ret;
                }
                ctrl.finish_group_update(params);
                let _ = ctrl.act_deterministic(&phi);
            });
        }

        assert!(ctrl.weights.iter().all(|w| w.is_finite()));
        assert!(ctrl.last_surrogate_after >= ctrl.last_surrogate_before - 1.0e-12);
    }
}
