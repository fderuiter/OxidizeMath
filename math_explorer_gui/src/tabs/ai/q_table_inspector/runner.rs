#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv, Move};
use math_explorer::ai::reinforcement_learning::{algorithms::TabularQAgent, MarkovDecisionProcess};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
#[allow(missing_docs)]
pub struct QTableSnapshot {
    pub episodes_trained: u32,
    pub q_values: HashMap<(GridState, Move), f64>,
}

pub(crate) fn default_grid_env() -> GridWorldEnv {
    GridWorldEnv {
        width: 5,
        height: 5,
        start: GridState { x: 0, y: 0 },
        goal: GridState { x: 4, y: 4 },
        traps: vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }],
        gamma: 0.9,
    }
}

pub(crate) fn default_q_agent() -> TabularQAgent<GridState, Move> {
    TabularQAgent::new(
        math_commons::primitives::UnitInterval::new(0.1).unwrap(),
        math_commons::primitives::UnitInterval::new(0.9).unwrap(),
        math_commons::primitives::UnitInterval::new(0.1).unwrap(),
        None,
    )
}

#[allow(missing_docs)]
pub struct QTableRunner {
    pub env: GridWorldEnv,
    pub agent: TabularQAgent<GridState, Move>,
    pub episodes_trained: u32,
    pub target_episodes: u32,
    pub is_training: bool,
}

impl Default for QTableRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl QTableRunner {
    /// Creates an initialized Q-learning background runner.
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
            SimCommand::Pause => {
                self.is_training = false;
            }
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
                    let q = self.agent.get_q_value(&state, &action);
                    q_values.insert((state, action), q);
                }
            }
        }
        let snapshot = QTableSnapshot {
            episodes_trained: self.episodes_trained,
            q_values,
        };
        StateSnapshot {
            width: self.env.width as usize,
            height: self.env.height as usize,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![f64::from(self.episodes_trained)],
            structured_data: Some(Box::new(snapshot)),
            sonification: crate::audio::SonificationFrame::silent(),
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
