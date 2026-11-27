#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::audio::{mappings::rl_td_surprise_frame, SonificationFrame};
use crate::tabs::ai::rl_sculpt::{
    apply_rl_brush_stroke, apply_rl_sculpt_payload, solve_bellman_with_policy, RlSolverMode,
};
use math_commons::primitives::UnitInterval;
use math_explorer::ai::reinforcement_learning::grid_world::Move;
#[cfg(test)]
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv};
#[cfg(test)]
use math_explorer::ai::reinforcement_learning::{algorithms::TabularQAgent, MarkovDecisionProcess};
use math_explorer::ai::reinforcement_learning::{
    step_softmax_pg_into, step_tabular_q_into, BellmanSolution, DenseRlWorkspace, SculptedGridMdp,
};
#[cfg(test)]
use std::collections::HashMap;
use std::sync::Arc;

/// Structured worker snapshot carrying learned tables, sculpted MDP, and exact Bellman values.
#[derive(Clone)]
#[allow(missing_docs)]
pub struct QTableSnapshot {
    pub episodes_trained: u32,
    pub mdp: SculptedGridMdp,
    pub solver_mode: RlSolverMode,
    pub q_table: Vec<[f64; 4]>,
    pub state_values: Vec<f64>,
    pub greedy_actions: Vec<Option<Move>>,
    pub bellman: BellmanSolution,
}

#[cfg(test)]
pub(crate) fn default_grid_env() -> GridWorldEnv {
    GridWorldEnv {
        width: 5,
        height: 5,
        start: GridState { x: 0, y: 0 },
        goal: GridState { x: 4, y: 4 },
        traps: vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }],
        walls: Vec::new(),
        gamma: 0.9,
    }
}

#[cfg(test)]
pub(crate) fn default_q_agent() -> TabularQAgent<GridState, Move> {
    TabularQAgent::new(
        UnitInterval::new(0.1).unwrap(),
        UnitInterval::new(0.9).unwrap(),
        UnitInterval::new(0.1).unwrap(),
        None,
    )
}

/// Legacy baseline runner preserved with exact public fields for existing integration tests.
#[cfg(test)]
#[allow(missing_docs)]
pub struct QTableRunner {
    pub env: GridWorldEnv,
    pub agent: TabularQAgent<GridState, Move>,
    pub episodes_trained: u32,
    pub target_episodes: u32,
    pub is_training: bool,
}

#[cfg(test)]
impl Default for QTableRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl QTableRunner {
    /// Creates an initialized baseline Q-learning background runner.
    pub fn new() -> Self {
        Self {
            env: default_grid_env(),
            agent: default_q_agent(),
            episodes_trained: 0,
            target_episodes: 0,
            is_training: false,
        }
    }

    fn run_one_episode(&mut self) {
        let mut current_state = self.env.start;
        let mut steps = 0;
        while !self.env.is_terminal(&current_state) && steps < 100 {
            let actions = self.env.actions(&current_state);
            if actions.is_empty() {
                break;
            }
            if let Some(action) = self.agent.select_action(&current_state, &actions) {
                let next_state = self.env.step(&current_state, &action);
                let reward = self.env.reward(&current_state, &action, &next_state);
                let next_actions = self.env.actions(&next_state);
                self.agent
                    .update(&current_state, &action, reward, &next_state, &next_actions);
                current_state = next_state;
            }
            steps += 1;
        }
        self.episodes_trained += 1;
    }
}

#[cfg(test)]
impl SimulationRunner for QTableRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::TrainEpisodes(count) => {
                self.target_episodes = self.episodes_trained + count as u32;
                self.is_training = true;
            }
            SimCommand::Start => {
                if self.episodes_trained < self.target_episodes {
                    self.is_training = true;
                }
            }
            SimCommand::Pause => self.is_training = false,
            SimCommand::Reset => {
                self.agent = default_q_agent();
                self.episodes_trained = 0;
                self.target_episodes = 0;
                self.is_training = false;
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if self.is_training && self.episodes_trained < self.target_episodes {
            let batch = (self.target_episodes - self.episodes_trained).min(50);
            for _ in 0..batch {
                self.run_one_episode();
            }
            if self.episodes_trained >= self.target_episodes {
                self.is_training = false;
            }
        }
    }

    fn step_once(&mut self) {
        self.run_one_episode();
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let mut q_values = HashMap::new();
        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let state = GridState { x, y };
                for action in [Move::Up, Move::Down, Move::Left, Move::Right] {
                    q_values.insert((state, action), self.agent.get_q_value(&state, &action));
                }
            }
        }
        StateSnapshot {
            width: self.env.width as usize,
            height: self.env.height as usize,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![f64::from(self.episodes_trained)],
            structured_data: Some(Box::new(q_values)),
            sonification: SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        20
    }

    fn is_complete(&self) -> bool {
        !self.is_training || self.episodes_trained >= self.target_episodes
    }

    fn is_finite_state(&self) -> bool {
        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let s = GridState { x, y };
                for a in [Move::Up, Move::Down, Move::Left, Move::Right] {
                    if !self.agent.get_q_value(&s, &a).is_finite() {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Zero-allocation sculpted MDP runner powering [`super::QTableInspectorTool`].
pub struct SculptedQTableRunner {
    /// Active sculpted GridWorld MDP.
    pub mdp: SculptedGridMdp,
    /// Pre-allocated dense RL buffers.
    pub workspace: DenseRlWorkspace,
    /// Active tabular Q-learning or REINFORCE solver mode.
    pub solver_mode: RlSolverMode,
    /// Completed training episodes count.
    pub episodes_trained: u32,
    /// Target episode count for background batch runs.
    pub target_episodes: u32,
    /// Whether background batch training is currently running.
    pub is_training: bool,
    /// Latest step sonification frame.
    pub latest_audio: SonificationFrame,
}

impl Default for SculptedQTableRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl SculptedQTableRunner {
    /// Creates a sculpted Q-table runner on the default 5x5 layout.
    pub fn new() -> Self {
        Self::from_mdp(
            SculptedGridMdp::default_with_traps(5, 5, 0.9),
            RlSolverMode::TabularQ,
        )
    }

    /// Creates a sculpted Q-table runner from an explicit `mdp` and `solver_mode`.
    pub fn from_mdp(mdp: SculptedGridMdp, solver_mode: RlSolverMode) -> Self {
        let mut workspace =
            DenseRlWorkspace::new(mdp.width as usize, mdp.height as usize, 100, None);
        workspace.reset_episode(&mdp);
        Self {
            mdp,
            workspace,
            solver_mode,
            episodes_trained: 0,
            target_episodes: 0,
            is_training: false,
            latest_audio: SonificationFrame::silent(),
        }
    }

    pub(super) fn run_one_episode(&mut self) {
        let alpha = UnitInterval::new(0.15).unwrap();
        let eps = UnitInterval::new(0.15).unwrap();
        let pg_lr = UnitInterval::new(0.06).unwrap();
        self.workspace.reset_episode(&self.mdp);
        for _ in 0..100 {
            let outcome = match self.solver_mode {
                RlSolverMode::TabularQ => {
                    step_tabular_q_into(&self.mdp, &mut self.workspace, alpha, eps)
                }
                RlSolverMode::PolicyGradient => {
                    step_softmax_pg_into(&self.mdp, &mut self.workspace, pg_lr)
                }
            };
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
        self.episodes_trained += 1;
    }

    pub(super) fn build_q_snapshot(&self) -> QTableSnapshot {
        let bellman = solve_bellman_with_policy(&self.mdp, &self.workspace, self.solver_mode);
        let mut greedy_actions = Vec::with_capacity(self.workspace.states.len());
        for &st in &self.workspace.states {
            let act = match self.solver_mode {
                RlSolverMode::TabularQ => self.workspace.greedy_q_action(st),
                RlSolverMode::PolicyGradient => self.workspace.greedy_policy_action(st),
            };
            greedy_actions.push(act);
        }
        QTableSnapshot {
            episodes_trained: self.episodes_trained,
            mdp: self.mdp.clone(),
            solver_mode: self.solver_mode,
            q_table: self.workspace.q_table.clone(),
            state_values: self.workspace.state_values.clone(),
            greedy_actions,
            bellman,
        }
    }
}

impl SimulationRunner for SculptedQTableRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::TrainEpisodes(count) => {
                self.target_episodes = self.episodes_trained + count as u32;
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
                self.episodes_trained = 0;
                self.target_episodes = 0;
                self.is_training = false;
                self.latest_audio = SonificationFrame::silent();
            }
            SimCommand::ApplyBrushStroke(stroke) => {
                let _ = apply_rl_brush_stroke(&mut self.mdp, &stroke);
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
                }
            }
            SimCommand::ClearObstacles => {
                self.mdp = SculptedGridMdp::default_with_traps(
                    self.mdp.width,
                    self.mdp.height,
                    self.mdp.gamma,
                );
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if self.is_training && self.episodes_trained < self.target_episodes {
            let batch = (self.target_episodes - self.episodes_trained).min(50);
            for _ in 0..batch {
                self.run_one_episode();
            }
            if self.episodes_trained >= self.target_episodes {
                self.is_training = false;
            }
        }
    }

    fn step_once(&mut self) {
        self.run_one_episode();
    }

    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: self.mdp.width as usize,
            height: self.mdp.height as usize,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![f64::from(self.episodes_trained)],
            structured_data: Some(Box::new(self.build_q_snapshot())),
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
        self.workspace.is_finite_state()
    }
}
