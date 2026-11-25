//! Zero-allocation tabular Q-learning and REINFORCE softmax policy-gradient rollout steps.

// [cite:algorithmic_information_rust]

use super::algorithms::policy_gradient_step;
use super::maze::{DenseRlWorkspace, GridCellRole, SculptedGridMdp, index_to_move, move_to_index};
use super::types::MarkovDecisionProcess;
use super::{GridState, Move};
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::primitives::UnitInterval;
use oxidize_core::rng::{OxidizeRng, sample_bounded_usize, sample_closed_open};

/// Telemetry emitted by a single zero-allocation RL step on [`SculptedGridMdp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RlStepOutcome {
    /// Agent state prior to taking [`Self::action`].
    pub prev_state: GridState,
    /// Action executed on this step.
    pub action: Move,
    /// Resulting environment state after wall bounce-back and wind slip.
    pub next_state: GridState,
    /// Immediate scalar reward $R_{t+1}$.
    pub reward: f64,
    /// Temporal-difference or advantage surprise signal $\delta_t$.
    pub td_error: f64,
    /// Estimated value $V(s_{t+1})$ of the resulting state.
    pub state_value: f64,
    /// Whether stochastic wind drift displaced the agent on this step.
    pub wind_slipped: bool,
    /// Stereo pan hint (`[-1.0, 1.0]`) when wind slip occurred.
    pub wind_pan: f32,
    /// Whether the episode terminated or truncated at the end of this step.
    pub episode_finished: bool,
    /// Completed episode cumulative return when `episode_finished` is `true`.
    pub completed_episode_return: Option<f64>,
}

/// Numerically stable 4-action Boltzmann softmax distribution $\pi_\theta(\cdot \mid s)$ on the stack.
#[inline]
#[must_use]
pub fn softmax_action_probs(logits: [f64; 4]) -> [f64; 4] {
    let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_l.is_finite() {
        return [0.25; 4];
    }
    let mut exps = [0.0_f64; 4];
    let mut sum = 0.0_f64;
    for i in 0..4 {
        let e = (logits[i] - max_l).clamp(-60.0, 60.0).exp();
        exps[i] = e;
        sum += e;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return [0.25; 4];
    }
    let inv = 1.0 / sum;
    [exps[0] * inv, exps[1] * inv, exps[2] * inv, exps[3] * inv]
}

/// Computes the 4-action REINFORCE parameter delta $\Delta \theta$ for a chosen action via [`policy_gradient_step`].
#[must_use]
#[verified_engine::verified]
pub fn compute_softmax_pg_delta(
    logits: [f64; 4],
    chosen_action: Move,
    advantage_gt: f64,
    learning_rate: UnitInterval,
) -> [f64; 4] {
    let probs = softmax_action_probs(logits);
    let chosen_idx = move_to_index(chosen_action);
    let mut grad_log_pi = [-probs[0], -probs[1], -probs[2], -probs[3]];
    grad_log_pi[chosen_idx] += 1.0;
    let delta_vec = policy_gradient_step(advantage_gt, &grad_log_pi, learning_rate);
    [delta_vec[0], delta_vec[1], delta_vec[2], delta_vec[3]]
}

#[inline]
fn next_uniform_f64(rng: &mut OxidizeRng) -> f64 {
    sample_closed_open(rng)
}

#[inline]
fn ensure_active_episode(mdp: &SculptedGridMdp, ws: &mut DenseRlWorkspace) {
    if !mdp.in_bounds(ws.current_state)
        || mdp.is_terminal(&ws.current_state)
        || mdp.cell_role(ws.current_state) == GridCellRole::SolidWall
        || ws.episode_steps >= ws.max_episode_steps
    {
        ws.current_state = mdp.start;
        ws.clear_episode_trajectory();
        ws.episode_steps = 0;
        ws.episode_return = 0.0;
    }
}

#[inline]
fn select_eps_greedy_action_index(q_row: [f64; 4], epsilon: f64, rng: &mut OxidizeRng) -> usize {
    if next_uniform_f64(rng) < epsilon {
        sample_bounded_usize(rng, 4)
    } else {
        let mut best = 0usize;
        for k in 1..4 {
            if q_row[k] > q_row[best] {
                best = k;
            }
        }
        best
    }
}

/// Advances one zero-allocation tabular Q-learning step using [`DenseRlWorkspace`].
pub fn step_tabular_q_into(
    mdp: &SculptedGridMdp,
    ws: &mut DenseRlWorkspace,
    alpha: UnitInterval,
    epsilon: UnitInterval,
) -> RlStepOutcome {
    ensure_active_episode(mdp, ws);
    let prev_state = ws.current_state;
    let s_idx = flatten_2d_index(prev_state.x as usize, prev_state.y as usize, ws.width);
    let q_row = ws.q_table()[s_idx];
    let a_idx = select_eps_greedy_action_index(q_row, epsilon.value(), &mut ws.rng);
    let action = index_to_move(a_idx);
    let (next_state, reward, wind_slipped, wind_pan) =
        mdp.sample_step(prev_state, action, &mut ws.rng);
    let ns_idx = flatten_2d_index(next_state.x as usize, next_state.y as usize, ws.width);
    let next_terminal = mdp.is_terminal(&next_state);
    let max_next_q = if next_terminal {
        0.0
    } else {
        ws.q_table()[ns_idx]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let current_q = q_row[a_idx];
    let gamma = mdp.gamma.clamp(0.0, 1.0);
    let td_error = reward + gamma * max_next_q - current_q;
    ws.q_table_mut()[s_idx][a_idx] = current_q + alpha.value() * td_error;

    ws.current_state = next_state;
    ws.episode_steps += 1;
    ws.episode_return += reward;
    let episode_finished = next_terminal || ws.episode_steps >= ws.max_episode_steps;
    let completed_episode_return = finalize_episode_if_done(mdp, ws, episode_finished);

    RlStepOutcome {
        prev_state,
        action,
        next_state,
        reward,
        td_error,
        state_value: if next_terminal {
            reward.clamp(-10.0, 10.0)
        } else {
            max_next_q
        },
        wind_slipped,
        wind_pan,
        episode_finished,
        completed_episode_return,
    }
}

#[inline]
fn finalize_episode_if_done(
    mdp: &SculptedGridMdp,
    ws: &mut DenseRlWorkspace,
    done: bool,
) -> Option<f64> {
    if done {
        let ret = ws.episode_return;
        ws.episodes_completed = ws.episodes_completed.saturating_add(1);
        ws.current_state = mdp.start;
        ws.episode_steps = 0;
        ws.episode_return = 0.0;
        Some(ret)
    } else {
        None
    }
}

fn flush_reinforce_episode_in_place(ws: &mut DenseRlWorkspace, gamma: f64, lr: f64) {
    let len = ws.episode_trajectory_len();
    if len == 0 {
        return;
    }
    let critic_lr = (lr * 0.5).clamp(0.01, 0.5);
    let (q_table, theta_logits, state_values, ep_states, ep_actions, ep_rewards) =
        ws.episode_update_slices();
    let mut g_t = 0.0_f64;
    for t in (0..len).rev() {
        let s_idx = ep_states[t];
        let a_idx = ep_actions[t] & 3;
        g_t = ep_rewards[t] + gamma * g_t;
        let baseline = state_values[s_idx];
        let advantage = (g_t - baseline).clamp(-25.0, 25.0);
        state_values[s_idx] = baseline + critic_lr * (g_t - baseline);
        let probs = softmax_action_probs(theta_logits[s_idx]);
        for (k, &prob) in probs.iter().enumerate() {
            let indicator = if k == a_idx { 1.0 } else { 0.0 };
            let grad = indicator - prob;
            theta_logits[s_idx][k] =
                (theta_logits[s_idx][k] + lr * advantage * grad).clamp(-20.0, 20.0);
            q_table[s_idx][k] = state_values[s_idx] + 0.25 * theta_logits[s_idx][k];
        }
    }
    ws.clear_episode_trajectory();
}

/// Advances one zero-allocation REINFORCE softmax policy-gradient step using [`DenseRlWorkspace`].
pub fn step_softmax_pg_into(
    mdp: &SculptedGridMdp,
    ws: &mut DenseRlWorkspace,
    learning_rate: UnitInterval,
) -> RlStepOutcome {
    ensure_active_episode(mdp, ws);
    let prev_state = ws.current_state;
    let s_idx = flatten_2d_index(prev_state.x as usize, prev_state.y as usize, ws.width);
    let probs = softmax_action_probs(ws.theta_logits()[s_idx]);
    let u = next_uniform_f64(&mut ws.rng);
    let mut cdf = 0.0_f64;
    let mut a_idx = 3usize;
    for (k, &p) in probs.iter().enumerate() {
        cdf += p;
        if u <= cdf {
            a_idx = k;
            break;
        }
    }
    let action = index_to_move(a_idx);
    let (next_state, reward, wind_slipped, wind_pan) =
        mdp.sample_step(prev_state, action, &mut ws.rng);
    let ns_idx = flatten_2d_index(next_state.x as usize, next_state.y as usize, ws.width);
    let next_terminal = mdp.is_terminal(&next_state);
    let gamma = mdp.gamma.clamp(0.0, 1.0);
    let next_val = if next_terminal {
        0.0
    } else {
        ws.state_values()[ns_idx]
    };
    let td_error = reward + gamma * next_val - ws.state_values()[s_idx];

    ws.push_episode_step(s_idx, a_idx, reward);

    ws.current_state = next_state;
    ws.episode_steps += 1;
    ws.episode_return += reward;
    let episode_finished = next_terminal || ws.episode_steps >= ws.max_episode_steps;
    if episode_finished {
        flush_reinforce_episode_in_place(ws, gamma, learning_rate.value());
    }
    let completed_episode_return = finalize_episode_if_done(mdp, ws, episode_finished);

    RlStepOutcome {
        prev_state,
        action,
        next_state,
        reward,
        td_error,
        state_value: if next_terminal { reward } else { next_val },
        wind_slipped,
        wind_pan,
        episode_finished,
        completed_episode_return,
    }
}
