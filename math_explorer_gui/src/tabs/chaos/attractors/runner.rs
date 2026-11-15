#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverExecutionState {
    Ready,
    Running,
    Paused,
    Diverged {
        invalid_vec: Vector3<f64>,
        diverged_at_step: usize,
    },
}

#[allow(missing_docs)]
pub struct AttractorSharedParams {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
    pub dt: f64,
    pub simulation_speed: usize,
    pub max_points: usize,
}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct AttractorSnapshot {
    pub history: VecDeque<Vector3<f64>>,
    pub current_vec: Vector3<f64>,
    pub step_count: usize,
    pub execution_state: SolverExecutionState,
}

#[allow(missing_docs)]
pub struct AttractorRunner {
    pub system: LorenzSystem,
    pub params: Arc<std::sync::RwLock<AttractorSharedParams>>,
    pub history: VecDeque<Vector3<f64>>,
    pub step_count: usize,
    pub execution_state: SolverExecutionState,
}

impl SimulationRunner for AttractorRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                let (sigma, rho, beta) = {
                    let p = self.params.read().unwrap();
                    (p.sigma, p.rho, p.beta)
                };
                self.system = LorenzBuilder::new()
                    .sigma(sigma)
                    .rho(rho)
                    .beta(beta)
                    .build(LorenzState::new(1.0, 1.0, 1.0));
                self.history.clear();
                self.step_count = 0;
                self.execution_state = SolverExecutionState::Ready;
            }
            SimCommand::SetSpeed(speed) => {
                if let Ok(mut p) = self.params.write() {
                    p.simulation_speed = speed;
                }
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (sigma, rho, beta, dt, simulation_speed, max_points) = {
            let p = self.params.read().unwrap();
            (
                p.sigma,
                p.rho,
                p.beta,
                p.dt,
                p.simulation_speed,
                p.max_points,
            )
        };
        self.system.sigma = sigma;
        self.system.rho = rho;
        self.system.beta = beta;

        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }

        for _ in 0..simulation_speed {
            let current_vec = self.system.state.vec;
            if !current_vec.x.is_finite()
                || !current_vec.y.is_finite()
                || !current_vec.z.is_finite()
            {
                self.execution_state = SolverExecutionState::Diverged {
                    invalid_vec: current_vec,
                    diverged_at_step: self.step_count,
                };
                break;
            }

            self.system.step(dt);
            self.step_count += 1;

            let vec = self.system.state.vec;
            if !vec.x.is_finite() || !vec.y.is_finite() || !vec.z.is_finite() {
                self.execution_state = SolverExecutionState::Diverged {
                    invalid_vec: vec,
                    diverged_at_step: self.step_count,
                };
                break;
            }

            if self.history.len() >= max_points {
                self.history.pop_front();
            }
            self.history.push_back(vec);
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let snapshot = AttractorSnapshot {
            history: self.history.clone(),
            current_vec: self.system.state.vec,
            step_count: self.step_count,
            execution_state: self.execution_state,
        };
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![
                self.system.state.vec.x,
                self.system.state.vec.y,
                self.system.state.vec.z,
                self.step_count as f64,
            ],
            structured_data: Some(Box::new(snapshot)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_complete(&self) -> bool {
        matches!(self.execution_state, SolverExecutionState::Diverged { .. })
    }
}

impl scientific_metadata::theory::TheoryDescribable for super::AttractorPlotter {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
