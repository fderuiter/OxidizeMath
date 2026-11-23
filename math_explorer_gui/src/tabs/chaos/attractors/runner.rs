#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::flow::{AttractorMode, CompiledFlowJob};
use super::serde_vec3;
use super::twin::{is_vec_finite, step_flow_vec, TwinConfig, TwinRunnerState};
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
    pub twin: TwinConfig,
}

impl Default for AttractorSharedParams {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            dt: 0.01,
            simulation_speed: 5,
            max_points: 2000,
            mode: AttractorMode::Lorenz,
            custom_flow: None,
            twin: TwinConfig::default(),
        }
    }
}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct AttractorSnapshot {
    pub history: VecDeque<Vector3<f64>>,
    pub current_vec: Vector3<f64>,
    pub twin_history: VecDeque<Vector3<f64>>,
    pub shadow_vec: Vector3<f64>,
    pub divergence_history: VecDeque<[f64; 2]>,
    pub twin_enabled: bool,
    pub step_count: usize,
    pub execution_state: SolverExecutionState,
}

impl AttractorSnapshot {
    /// Estimates heap bytes owned by this snapshot's `VecDeque` buffers for `SnapshotHistory`.
    #[must_use]
    pub fn estimated_heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + (self.history.capacity() + self.twin_history.capacity())
                * std::mem::size_of::<Vector3<f64>>()
            + self.divergence_history.capacity() * std::mem::size_of::<[f64; 2]>()
    }
}

#[allow(missing_docs)]
pub struct AttractorRunner {
    pub system: LorenzSystem,
    pub params: Arc<std::sync::RwLock<AttractorSharedParams>>,
    pub history: VecDeque<Vector3<f64>>,
    pub twin: TwinRunnerState,
    pub step_count: usize,
    pub execution_state: SolverExecutionState,
    pub rng: OxidizeRng,
}

impl AttractorRunner {
    /// Builds a fresh runner initialized from shared parameters.
    pub fn from_params(params: Arc<std::sync::RwLock<AttractorSharedParams>>) -> Self {
        let (sigma, rho, beta, max_points, twin_cfg) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.sigma, p.rho, p.beta, p.max_points, p.twin.sanitized())
        };
        let init = twin_cfg.initial_state;
        Self {
            system: LorenzBuilder::new()
                .sigma(sigma)
                .rho(rho)
                .beta(beta)
                .build(LorenzState::new(init.x, init.y, init.z)),
            params,
            history: VecDeque::with_capacity(max_points),
            twin: TwinRunnerState::from_config(twin_cfg, max_points),
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
        bool,
        Option<Arc<CompiledFlowJob>>,
    ) {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        self.system.sigma = p.sigma;
        self.system.rho = p.rho;
        self.system.beta = p.beta;
        self.twin.enabled = p.twin.enabled;
        (
            p.dt,
            p.simulation_speed.clamp(1, MAX_SIMULATION_SPEED),
            p.max_points,
            p.mode,
            p.twin.enabled,
            p.custom_flow.clone(),
        )
    }

    fn check_pre_tick(&mut self, dt: f64, mode: AttractorMode, twin_enabled: bool) -> bool {
        let current_vec = self.system.state.vec;
        let lorenz_ok = mode != AttractorMode::Lorenz
            || (self.system.sigma.is_finite()
                && self.system.rho.is_finite()
                && self.system.beta.is_finite());
        let twin_ok = !twin_enabled || is_vec_finite(self.twin.shadow_vec);
        if is_vec_finite(current_vec) && lorenz_ok && twin_ok && dt.is_finite() {
            return true;
        }
        self.execution_state = SolverExecutionState::Diverged {
            invalid_vec: if twin_ok {
                current_vec
            } else {
                self.twin.shadow_vec
            },
            diverged_at_step: self.step_count,
        };
        false
    }

    fn advance_tick(
        &mut self,
        dt: f64,
        max_points: usize,
        mode: AttractorMode,
        twin_enabled: bool,
        custom_flow: Option<&CompiledFlowJob>,
    ) -> bool {
        if !self.check_pre_tick(dt, mode, twin_enabled) {
            return false;
        }
        let current_vec = self.system.state.vec;
        match mode {
            AttractorMode::Lorenz => self.system.step(dt),
            AttractorMode::Custom => {
                if let Some(job) = custom_flow {
                    self.system.state.vec = job.rk4_step(current_vec, dt);
                }
            }
        }
        let next_shadow = if twin_enabled {
            step_flow_vec(
                mode,
                self.system.sigma,
                self.system.rho,
                self.system.beta,
                dt,
                custom_flow,
                self.twin.shadow_vec,
            )
        } else {
            self.twin.shadow_vec
        };
        self.step_count += 1;

        let vec = self.system.state.vec;
        let shadow_finite = !twin_enabled || is_vec_finite(next_shadow);
        if !is_vec_finite(vec) || !shadow_finite {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: if shadow_finite { vec } else { next_shadow },
                diverged_at_step: self.step_count,
            };
            return false;
        }

        if self.history.len() >= max_points {
            self.history.pop_front();
        }
        self.history.push_back(vec);
        self.twin.enabled = twin_enabled;
        self.twin
            .record_step(self.step_count, dt, max_points, vec, next_shadow);
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
                let (sigma, rho, beta, max_points, twin_cfg) = {
                    let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                    (p.sigma, p.rho, p.beta, p.max_points, p.twin.sanitized())
                };
                let init = twin_cfg.initial_state;
                self.system = LorenzBuilder::new()
                    .sigma(sigma)
                    .rho(rho)
                    .beta(beta)
                    .build(LorenzState::new(init.x, init.y, init.z));
                self.history.clear();
                self.twin.reset(twin_cfg, max_points);
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
        let (dt, simulation_speed, max_points, mode, twin_enabled, custom_flow) =
            self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        for _ in 0..simulation_speed {
            if !self.advance_tick(dt, max_points, mode, twin_enabled, custom_flow.as_deref()) {
                break;
            }
        }
        if self.history.len() >= 16 && self.step_count % 15 < simulation_speed {
            self.publish_live_landmarks(dt);
        }
    }

    fn step_once(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, _, max_points, mode, twin_enabled, custom_flow) = self.sync_params();
        if self.advance_tick(dt, max_points, mode, twin_enabled, custom_flow.as_deref()) {
            self.execution_state = SolverExecutionState::Paused;
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let v = self.system.state.vec;
        let rhs = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            super::twin::eval_flow_rhs(
                p.mode,
                self.system.sigma,
                self.system.rho,
                self.system.beta,
                p.custom_flow.as_deref(),
                v,
            )
        };
        let diverged = matches!(self.execution_state, SolverExecutionState::Diverged { .. });
        let sonification = crate::audio::mappings::attractor_frame([v.x, v.y, v.z], rhs, diverged);
        let snapshot = AttractorSnapshot {
            history: self.history.clone(),
            current_vec: v,
            twin_history: self.twin.twin_history.clone(),
            shadow_vec: self.twin.shadow_vec,
            divergence_history: self.twin.divergence_history.clone(),
            twin_enabled: self.twin.enabled,
            step_count: self.step_count,
            execution_state: self.execution_state,
        };
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![v.x, v.y, v.z, self.step_count as f64],
            structured_data: Some(Box::new(snapshot)),
            sonification,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        let v = self.system.state.vec;
        let sv = self.twin.shadow_vec;
        let shadow_ok =
            !self.twin.enabled || (sv.x.is_finite() && sv.y.is_finite() && sv.z.is_finite());
        !matches!(self.execution_state, SolverExecutionState::Diverged { .. })
            && v.x.is_finite()
            && v.y.is_finite()
            && v.z.is_finite()
            && shadow_ok
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        if let SolverExecutionState::Diverged {
            invalid_vec: v,
            diverged_at_step: s,
        } = self.execution_state
        {
            let msg = format!(
                "Non-finite attractor state [{:.2}, {:.2}, {:.2}] at step {s}",
                v.x, v.y, v.z
            );
            return Some(DiagnosticReport::new(s as f64, s, "lorenz_state", msg));
        }
        if !self.is_finite_state() {
            let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
            let t = self.step_count as f64 * if dt.is_finite() { dt } else { 0.01 };
            return Some(DiagnosticReport::new(
                t,
                self.step_count,
                "lorenz_state",
                "Non-finite attractor trajectory coordinates detected",
            ));
        }
        None
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
        let twin_flag = if p.twin.enabled { 1.0 } else { 0.0 };
        Some(CheckpointParams::scalars(&[
            p.sigma,
            p.rho,
            p.beta,
            p.dt,
            p.max_points as f64,
            mode_flag,
            twin_flag,
            p.twin.epsilon,
        ]))
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let params = self.capture_active_params()?;
        let v = self.system.state.vec;
        let sv = self.twin.shadow_vec;
        let twin_flag = if self.twin.enabled { 1.0 } else { 0.0 };
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params,
            state: CheckpointPayload::scalars(&[v.x, v.y, v.z, sv.x, sv.y, sv.z, twin_flag]),
        })
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let Some(s) = checkpoint.state.as_scalars() {
            if s.len() >= 3 {
                self.system.state.vec = Vector3::new(s[0], s[1], s[2]);
            }
            if s.len() >= 7 {
                self.twin.shadow_vec = Vector3::new(s[3], s[4], s[5]);
                self.twin.enabled = s[6] > 0.5;
            }
        }
        if let Some(p) = checkpoint.params.as_scalars().filter(|p| p.len() >= 3) {
            (self.system.sigma, self.system.rho, self.system.beta) = (p[0], p[1], p[2]);
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
        self.twin.rebuild_histories(&mut self.history, front, back);
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        let Some(p) = params.as_scalars().filter(|p| p.len() >= 5) else {
            return;
        };
        (self.system.sigma, self.system.rho, self.system.beta) = (p[0], p[1], p[2]);
        let (dt, max_points) = (p[3], (p[4] as usize).max(1));
        let mode = if p.get(5).copied().unwrap_or(0.0) > 0.5 {
            AttractorMode::Custom
        } else {
            AttractorMode::Lorenz
        };
        let twin_enabled = p.get(6).copied().unwrap_or(0.0) > 0.5;
        let custom_flow = (mode == AttractorMode::Custom)
            .then(|| {
                self.params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .custom_flow
                    .clone()
            })
            .flatten();
        if self.advance_tick(dt, max_points, mode, twin_enabled, custom_flow.as_deref()) {
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
        let (dt, _, max_points, mode, twin_enabled, custom_flow) = self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        if self.advance_tick(dt, max_points, mode, twin_enabled, custom_flow.as_deref())
            && self.history.len() >= 16
            && self.step_count.is_multiple_of(15)
        {
            self.publish_live_landmarks(dt);
        }
    }
}
