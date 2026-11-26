//! Background episodic RL simulation runner for [`super::RewardPlotsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::audio::{mappings::rl_td_surprise_frame, SonificationFrame};
use crate::tabs::ai::rl_sculpt::{
    apply_rl_brush_stroke, apply_rl_sculpt_payload, mean_bellman_residual, RlSolverMode,
};
use math_commons::primitives::UnitInterval;
use math_explorer::ai::reinforcement_learning::{
    step_softmax_pg_into, step_tabular_q_into, BellmanSolution, DenseRlWorkspace, SculptedGridMdp,
};
use std::sync::Arc;

/// Structured episodic trajectory telemetry emitted by [`RewardPlotsRunner`].
pub struct RlSnapshotData {
    /// Cumulative reward per completed episode.
    pub rewards_per_episode: Vec<f64>,
    /// Mean Bellman optimality residual $|\hat{V}(s) - V^*(s)|$ per completed episode.
    pub bellman_residuals: Vec<f64>,
    /// Total episodes trained so far.
    pub episodes_trained: usize,
    /// Target episode count for the active training run.
    pub target_episodes: usize,
}

/// Zero-allocation sculpted MDP episodic training runner for [`super::RewardPlotsTool`].
pub struct RewardPlotsRunner {
    mdp: SculptedGridMdp,
    workspace: DenseRlWorkspace,
    bellman: BellmanSolution,
    solver_mode: RlSolverMode,
    rewards_per_episode: Vec<f64>,
    bellman_residuals: Vec<f64>,
    episodes_trained: usize,
    target_episodes: usize,
    is_training: bool,
    latest_audio: SonificationFrame,
}

impl Default for RewardPlotsRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardPlotsRunner {
    /// Creates an initialized episodic RL runner on the default 5x5 layout.
    pub fn new() -> Self {
        Self::from_mdp(
            SculptedGridMdp::default_with_traps(5, 5, 0.9),
            RlSolverMode::TabularQ,
        )
    }

    pub(super) fn from_mdp(mdp: SculptedGridMdp, solver_mode: RlSolverMode) -> Self {
        let mut workspace =
            DenseRlWorkspace::new(mdp.width as usize, mdp.height as usize, 100, None);
        workspace.reset_episode(&mdp);
        let bellman = BellmanSolution::solve(&mdp, 1e-7, 160);
        Self {
            mdp,
            workspace,
            bellman,
            solver_mode,
            rewards_per_episode: Vec::new(),
            bellman_residuals: Vec::new(),
            episodes_trained: 0,
            target_episodes: 0,
            is_training: false,
            latest_audio: SonificationFrame::silent(),
        }
    }

    fn train_one_episode(&mut self) {
        let alpha = UnitInterval::new(0.15).unwrap();
        let eps = UnitInterval::new(0.12).unwrap();
        let pg_lr = UnitInterval::new(0.06).unwrap();

        self.workspace.reset_episode(&self.mdp);
        let mut episode_reward = 0.0;
        for _ in 0..100 {
            let outcome = match self.solver_mode {
                RlSolverMode::TabularQ => {
                    step_tabular_q_into(&self.mdp, &mut self.workspace, alpha, eps)
                }
                RlSolverMode::PolicyGradient => {
                    step_softmax_pg_into(&self.mdp, &mut self.workspace, pg_lr)
                }
            };
            episode_reward += outcome.reward;
            self.latest_audio = rl_td_surprise_frame(
                outcome.state_value,
                outcome.td_error,
                outcome.reward,
                outcome.wind_pan,
            );
            if outcome.episode_finished {
                break;
            }
        }

        let residual =
            mean_bellman_residual(&self.mdp, &self.workspace, self.solver_mode, &self.bellman);
        self.rewards_per_episode.push(episode_reward);
        self.bellman_residuals.push(residual);
        self.episodes_trained += 1;
    }
}

impl SimulationRunner for RewardPlotsRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::TrainEpisodes(count) => {
                self.target_episodes = self.episodes_trained + count;
                self.is_training = true;
            }
            SimCommand::Start => {
                if self.episodes_trained < self.target_episodes {
                    self.is_training = true;
                }
            }
            SimCommand::Pause => self.is_training = false,
            SimCommand::Reset => {
                self.workspace
                    .resize_and_clear(self.mdp.width as usize, self.mdp.height as usize);
                self.workspace.reset_episode(&self.mdp);
                self.rewards_per_episode.clear();
                self.bellman_residuals.clear();
                self.episodes_trained = 0;
                self.target_episodes = 0;
                self.is_training = false;
                self.latest_audio = SonificationFrame::silent();
            }
            SimCommand::ApplyBrushStroke(stroke) => {
                if apply_rl_brush_stroke(&mut self.mdp, &stroke) {
                    self.bellman = BellmanSolution::solve(&self.mdp, 1e-7, 160);
                }
            }
            SimCommand::SyncSculptState(payload) => {
                if apply_rl_sculpt_payload(&mut self.mdp, &mut self.solver_mode, &payload).is_ok() {
                    if self.workspace.width != self.mdp.width as usize
                        || self.workspace.height != self.mdp.height as usize
                    {
                        self.workspace
                            .resize_and_clear(self.mdp.width as usize, self.mdp.height as usize);
                    }
                    self.workspace.reset_episode(&self.mdp);
                    self.bellman = BellmanSolution::solve(&self.mdp, 1e-7, 160);
                }
            }
            SimCommand::ClearObstacles => {
                self.mdp = SculptedGridMdp::default_with_traps(
                    self.mdp.width,
                    self.mdp.height,
                    self.mdp.gamma,
                );
                self.bellman = BellmanSolution::solve(&self.mdp, 1e-7, 160);
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if self.is_training && self.episodes_trained < self.target_episodes {
            self.train_one_episode();
            if self.episodes_trained >= self.target_episodes {
                self.is_training = false;
            }
        }
    }

    fn step_once(&mut self) {
        self.train_one_episode();
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let snapshot_data = Box::new(RlSnapshotData {
            rewards_per_episode: self.rewards_per_episode.clone(),
            bellman_residuals: self.bellman_residuals.clone(),
            episodes_trained: self.episodes_trained,
            target_episodes: self.target_episodes,
        });

        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![self.episodes_trained as f64],
            structured_data: Some(snapshot_data),
            sonification: self.latest_audio,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        20
    }

    fn is_complete(&self) -> bool {
        !self.is_training || self.episodes_trained >= self.target_episodes
    }

    fn is_finite_state(&self) -> bool {
        self.rewards_per_episode.iter().all(|r| r.is_finite())
            && self.bellman_residuals.iter().all(|r| r.is_finite())
            && self.workspace.is_finite_state()
    }
}
