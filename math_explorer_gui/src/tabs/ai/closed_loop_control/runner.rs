//! Cooperative background `SimulationRunner` and snapshot payload for `ClosedLoopControlTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::controllers::{Q_ACTION_BINS, Q_STATE_BINS};
use super::pipeline::{ClosedLoopPipeline, MAX_RETURN_HISTORY};
use super::state::ClosedLoopState;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use std::sync::{Arc, RwLock};

/// Snapshot payload transferred from the background `ClosedLoopRunner` to the UI thread.
#[derive(Clone)]
pub struct ClosedLoopTrainingSnapshot {
    /// Total RL episodes completed.
    pub episodes_trained: u32,
    /// Learned GRPO policy weights.
    pub grpo_weights: [f64; 4],
    /// Latest GRPO group mean return.
    pub grpo_mean_return: f64,
    /// Latest GRPO surrogate objective before proximal update.
    pub grpo_surrogate_before: f64,
    /// Latest GRPO surrogate objective after proximal update.
    pub grpo_surrogate_after: f64,
    /// Learned `64 x 7` Q-table snapshot.
    pub q_table: [[f64; Q_ACTION_BINS]; Q_STATE_BINS],
    /// Latest Q-learning episodic return.
    pub q_last_return: f64,
    /// Episodic learning curve `[episode, return]`.
    pub return_history: Vec<[f64; 2]>,
}

/// Cooperative background `SimulationRunner` capping per-frame rollout batches within `WASM_FRAME_SLICE_BUDGET`.
pub struct ClosedLoopRunner {
    cfg: ClosedLoopState,
    pipeline: ClosedLoopPipeline,
    target_episodes: u32,
    is_training: bool,
}

impl ClosedLoopRunner {
    /// Creates a background training runner seeded from the current tool configuration and learned weights.
    pub fn from_state_and_pipeline(cfg: ClosedLoopState, src: &ClosedLoopPipeline) -> Self {
        let mut runner_cfg = cfg;
        runner_cfg.warmup_episodes = 0;
        let mut pipeline = ClosedLoopPipeline::new(&runner_cfg);
        pipeline.episodes_trained = src.episodes_trained;
        pipeline.grpo.weights = src.grpo.weights;
        pipeline.grpo.last_mean_return = src.grpo.last_mean_return;
        pipeline
            .q_ctrl
            .agent
            .q_func
            .copy_from_table(src.q_ctrl.agent.q_func.as_table());
        for &pt in src.return_history.iter().take(MAX_RETURN_HISTORY) {
            pipeline.return_history.push(pt);
        }
        Self {
            cfg: runner_cfg,
            pipeline,
            target_episodes: src.episodes_trained,
            is_training: false,
        }
    }
}

impl SimulationRunner for ClosedLoopRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::TrainEpisodes(count) => {
                self.target_episodes = self.pipeline.episodes_trained.saturating_add(count as u32);
                self.is_training = true;
            }
            SimCommand::Start => {
                if self.pipeline.episodes_trained < self.target_episodes {
                    self.is_training = true;
                }
            }
            SimCommand::Pause => {
                self.is_training = false;
            }
            SimCommand::Reset => {
                self.pipeline.reset_untrained(&self.cfg);
                self.target_episodes = 0;
                self.is_training = false;
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if self.is_training && self.pipeline.episodes_trained < self.target_episodes {
            let _ = self.pipeline.train_episode_zero_alloc(&self.cfg);
            if self.pipeline.episodes_trained >= self.target_episodes {
                self.is_training = false;
            }
        }
    }

    fn step_once(&mut self) {
        let _ = self.pipeline.train_episode_zero_alloc(&self.cfg);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let snap = ClosedLoopTrainingSnapshot {
            episodes_trained: self.pipeline.episodes_trained,
            grpo_weights: self.pipeline.grpo.weights,
            grpo_mean_return: self.pipeline.grpo.last_mean_return,
            grpo_surrogate_before: self.pipeline.grpo.last_surrogate_before,
            grpo_surrogate_after: self.pipeline.grpo.last_surrogate_after,
            q_table: *self.pipeline.q_ctrl.agent.q_func.as_table(),
            q_last_return: self.pipeline.q_ctrl.last_episode_return,
            return_history: self.pipeline.return_history.clone(),
        };
        StateSnapshot {
            width: Q_STATE_BINS,
            height: Q_ACTION_BINS,
            pixels: Arc::new(RwLock::new(Vec::new())),
            custom_data: vec![f64::from(self.pipeline.episodes_trained)],
            structured_data: Some(Box::new(snap)),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        2
    }

    fn is_complete(&self) -> bool {
        !self.is_training || self.pipeline.episodes_trained >= self.target_episodes
    }

    fn is_finite_state(&self) -> bool {
        self.pipeline.grpo.weights.iter().all(|w| w.is_finite()) && self.pipeline.q_ctrl.is_finite()
    }
}

// [cite:grpo]
