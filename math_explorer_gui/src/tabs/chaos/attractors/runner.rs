#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::flow::{AttractorMode, CompiledFlowJob};
use super::serde_vec3;
use crate::async_sim::{
    CheckpointParams, CheckpointPayload, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use crate::framework::recovery::DiagnosticReport;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use nalgebra::Vector3;
use oxidize_core::rng::OxidizeRng;
use std::collections::VecDeque;
use std::sync::Arc;

/// Maximum allowed simulation sub-steps per frame for [`super::AttractorPlotter`].
pub const MAX_SIMULATION_SPEED: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SolverExecutionState {
    Ready,
    Running,
    Paused,
    Diverged {
        #[serde(with = "serde_vec3")]
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
    pub mode: AttractorMode,
    pub custom_flow: Option<Arc<CompiledFlowJob>>,
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
    pub rng: OxidizeRng,
}

impl AttractorRunner {
    /// Builds a fresh runner initialized from shared parameters.
    pub fn from_params(params: Arc<std::sync::RwLock<AttractorSharedParams>>) -> Self {
        let (sigma, rho, beta, max_points) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.sigma, p.rho, p.beta, p.max_points)
        };
        Self {
            system: LorenzBuilder::new()
                .sigma(sigma)
                .rho(rho)
                .beta(beta)
                .build(LorenzState::new(1.0, 1.0, 1.0)),
            params,
            history: VecDeque::with_capacity(max_points),
            step_count: 0,
            execution_state: SolverExecutionState::Ready,
            rng: OxidizeRng::new(0x4C4F_5245_4E5A_3139),
        }
    }

    fn sync_params(
        &mut self,
    ) -> (
        f64,
        usize,
        usize,
        AttractorMode,
        Option<Arc<CompiledFlowJob>>,
    ) {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        self.system.sigma = p.sigma;
        self.system.rho = p.rho;
        self.system.beta = p.beta;
        (
            p.dt,
            p.simulation_speed.clamp(1, MAX_SIMULATION_SPEED),
            p.max_points,
            p.mode,
            p.custom_flow.clone(),
        )
    }

    fn advance_tick(
        &mut self,
        dt: f64,
        max_points: usize,
        mode: AttractorMode,
        custom_flow: Option<&CompiledFlowJob>,
    ) -> bool {
        let current_vec = self.system.state.vec;
        let lorenz_ok = mode != AttractorMode::Lorenz
            || (self.system.sigma.is_finite()
                && self.system.rho.is_finite()
                && self.system.beta.is_finite());
        if !current_vec.x.is_finite()
            || !current_vec.y.is_finite()
            || !current_vec.z.is_finite()
            || !lorenz_ok
            || !dt.is_finite()
        {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: current_vec,
                diverged_at_step: self.step_count,
            };
            return false;
        }

        match mode {
            AttractorMode::Lorenz => self.system.step(dt),
            AttractorMode::Custom => {
                if let Some(job) = custom_flow {
                    self.system.state.vec = job.rk4_step(current_vec, dt);
                }
            }
        }
        self.step_count += 1;

        let vec = self.system.state.vec;
        if !vec.x.is_finite() || !vec.y.is_finite() || !vec.z.is_finite() {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: vec,
                diverged_at_step: self.step_count,
            };
            return false;
        }

        if self.history.len() >= max_points {
            self.history.pop_front();
        }
        self.history.push_back(vec);
        true
    }
}

impl SimulationRunner for AttractorRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Start => {
                if !matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
                    self.execution_state = SolverExecutionState::Running;
                }
            }
            SimCommand::Pause => {
                if !matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
                    self.execution_state = SolverExecutionState::Paused;
                }
            }
            SimCommand::Reset => {
                let (sigma, rho, beta) = {
                    let p = self.params.read().unwrap_or_else(|e| e.into_inner());
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
                self.rng = OxidizeRng::new(0x4C4F_5245_4E5A_3139);
            }
            SimCommand::SetSpeed(speed) => {
                self.params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .simulation_speed = speed.clamp(1, MAX_SIMULATION_SPEED);
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, simulation_speed, max_points, mode, custom_flow) = self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        for _ in 0..simulation_speed {
            if !self.advance_tick(dt, max_points, mode, custom_flow.as_deref()) {
                break;
            }
        }
    }

    fn step_once(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, _, max_points, mode, custom_flow) = self.sync_params();
        if self.advance_tick(dt, max_points, mode, custom_flow.as_deref()) {
            self.execution_state = SolverExecutionState::Paused;
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

    fn is_finite_state(&self) -> bool {
        let v = self.system.state.vec;
        !matches!(self.execution_state, SolverExecutionState::Diverged { .. })
            && v.x.is_finite()
            && v.y.is_finite()
            && v.z.is_finite()
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        if let SolverExecutionState::Diverged {
            invalid_vec,
            diverged_at_step,
        } = self.execution_state
        {
            Some(DiagnosticReport::new(
                diverged_at_step as f64,
                diverged_at_step,
                "lorenz_state",
                format!(
                    "Non-finite attractor state [{:.2}, {:.2}, {:.2}] at step {}",
                    invalid_vec.x, invalid_vec.y, invalid_vec.z, diverged_at_step
                ),
            ))
        } else if !self.is_finite_state() {
            let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
            Some(DiagnosticReport::new(
                self.step_count as f64 * if dt.is_finite() { dt } else { 0.01 },
                self.step_count,
                "lorenz_state",
                "Non-finite attractor trajectory coordinates detected",
            ))
        } else {
            None
        }
    }

    fn checkpoint_stride(&self) -> Option<usize> {
        Some(1)
    }

    fn current_step(&self) -> usize {
        self.step_count
    }

    fn capture_active_params(&self) -> Option<CheckpointParams> {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let mode_flag = match p.mode {
            AttractorMode::Lorenz => 0.0,
            AttractorMode::Custom => 1.0,
        };
        Some(CheckpointParams::scalars(&[
            p.sigma,
            p.rho,
            p.beta,
            p.dt,
            p.max_points as f64,
            mode_flag,
        ]))
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let params = self.capture_active_params()?;
        let v = self.system.state.vec;
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params,
            state: CheckpointPayload::scalars(&[v.x, v.y, v.z]),
        })
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let Some(s) = checkpoint.state.as_scalars() {
            if s.len() >= 3 {
                self.system.state.vec = Vector3::new(s[0], s[1], s[2]);
            }
        }
        if let Some(p) = checkpoint.params.as_scalars() {
            if p.len() >= 3 {
                self.system.sigma = p[0];
                self.system.rho = p[1];
                self.system.beta = p[2];
            }
        }
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
        self.execution_state = if checkpoint.step == 0 {
            SolverExecutionState::Ready
        } else {
            SolverExecutionState::Paused
        };
    }

    fn rebuild_history(&mut self, front: &[TimeTravelCheckpoint], back: &[TimeTravelCheckpoint]) {
        let max_points = back
            .last()
            .or_else(|| front.last())
            .and_then(|cp| cp.params.as_scalars()?.get(4).copied())
            .map_or(2000usize, |v| (v as usize).max(1));
        self.history.clear();
        let total_post_zero = front.iter().chain(back).filter(|cp| cp.step >= 1).count();
        let skip = total_post_zero.saturating_sub(max_points);
        for cp in front
            .iter()
            .chain(back)
            .filter(|cp| cp.step >= 1)
            .skip(skip)
        {
            if let Some(s) = cp.state.as_scalars().filter(|s| s.len() >= 3) {
                self.history.push_back(Vector3::new(s[0], s[1], s[2]));
            }
        }
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        let Some(p) = params.as_scalars() else {
            return;
        };
        if p.len() < 5 {
            return;
        }
        self.system.sigma = p[0];
        self.system.rho = p[1];
        self.system.beta = p[2];
        let dt = p[3];
        let max_points = (p[4] as usize).max(1);
        let mode = if p.get(5).copied().unwrap_or(0.0) > 0.5 {
            AttractorMode::Custom
        } else {
            AttractorMode::Lorenz
        };
        let custom_flow = (mode == AttractorMode::Custom)
            .then(|| {
                self.params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .custom_flow
                    .clone()
            })
            .flatten();
        if self.advance_tick(dt, max_points, mode, custom_flow.as_deref()) {
            self.execution_state = SolverExecutionState::Paused;
        }
    }

    fn running_substeps_per_tick(&self) -> usize {
        self.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .simulation_speed
            .clamp(1, MAX_SIMULATION_SPEED)
    }

    fn step_running_substep(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, _, max_points, mode, custom_flow) = self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        let _ = self.advance_tick(dt, max_points, mode, custom_flow.as_deref());
    }
}
