#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::branch::{register_runner_branch, with_existing_runner_branch, AttractorBranchSnapshot};
use super::flow::{AttractorMode, CompiledFlowJob};
use super::serde_vec3;
use crate::async_sim::{
    CheckpointParams, SimCommand, SimulationRunner, StateSnapshot, TimeTravelCheckpoint,
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
    pub branch: AttractorBranchSnapshot,
    pub twin: Option<super::AttractorTwinState>,
}

#[allow(missing_docs)]
pub struct AttractorRunner {
    pub system: LorenzSystem,
    pub params: Arc<std::sync::RwLock<AttractorSharedParams>>,
    pub history: VecDeque<Vector3<f64>>,
    pub step_count: usize,
    pub execution_state: SolverExecutionState,
    pub twin: Option<super::AttractorTwinState>,
    pub rng: OxidizeRng,
}

impl AttractorRunner {
    /// Builds a fresh runner initialized from shared parameters.
    pub fn from_params(params: Arc<std::sync::RwLock<AttractorSharedParams>>) -> Self {
        register_runner_branch(&params);
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
            history: VecDeque::with_capacity(max_points.max(super::ATTRACTOR_RING_CAPACITY)),
            step_count: 0,
            execution_state: SolverExecutionState::Ready,
            twin: None,
            rng: OxidizeRng::new(0x4C4F_5245_4E5A_3139),
        }
    }

    pub(crate) fn sync_params_pub(
        &mut self,
    ) -> (
        f64,
        usize,
        usize,
        AttractorMode,
        Option<Arc<CompiledFlowJob>>,
    ) {
        self.sync_params()
    }

    pub(crate) fn advance_tick_pub(
        &mut self,
        dt: f64,
        max_points: usize,
        mode: AttractorMode,
        custom_flow: Option<&CompiledFlowJob>,
    ) -> bool {
        self.advance_tick(dt, max_points, mode, custom_flow, true)
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

    fn step_twin_if_enabled(&mut self, dt: f64, max_points: usize, enabled: bool) -> bool {
        if enabled {
            if let Some(ref mut twin) = self.twin {
                if !twin.step_branch_a(dt, max_points) {
                    self.execution_state = SolverExecutionState::Diverged {
                        invalid_vec: twin.system_a.state.vec,
                        diverged_at_step: self.step_count,
                    };
                    return false;
                }
            }
        }
        true
    }

    pub(super) fn advance_tick(
        &mut self,
        dt: f64,
        max_points: usize,
        mode: AttractorMode,
        custom_flow: Option<&CompiledFlowJob>,
        step_twin_too: bool,
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
        if !self.step_twin_if_enabled(dt, max_points, step_twin_too) {
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

        let cap = max_points.clamp(1, super::ATTRACTOR_RING_CAPACITY);
        if self.history.len() >= cap {
            self.history.pop_front();
        }
        self.history.push_back(vec);
        let step = self.step_count;
        let _ = with_existing_runner_branch(&self.params, |b| {
            b.on_step_in_place(step, dt, max_points, vec);
        });
        if step_twin_too {
            if let Some(ref mut twin) = self.twin {
                twin.observe_lockstep(self.step_count, dt, vec);
            }
        }
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
                self.twin = None;
                self.execution_state = SolverExecutionState::Ready;
                self.reset_runner_branch();
                self.rng = OxidizeRng::new(0x4C4F_5245_4E5A_3139);
            }
            SimCommand::SetSpeed(speed) => {
                self.params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .simulation_speed = speed.clamp(1, MAX_SIMULATION_SPEED);
            }
            ref other => {
                if !self.handle_branch_command(other) {
                    if let SimCommand::Custom(ref cmd) = *other {
                        self.handle_custom_command(cmd);
                    }
                }
            }
        }
    }

    fn has_counterfactual_twin(&self) -> bool {
        self.twin.is_some()
    }

    fn step_branch_a(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, simulation_speed, max_points, _, _) = self.sync_params();
        if let Some(ref mut twin) = self.twin {
            for _ in 0..simulation_speed {
                if !twin.step_branch_a(dt, max_points) {
                    self.execution_state = SolverExecutionState::Diverged {
                        invalid_vec: twin.system_a.state.vec,
                        diverged_at_step: self.step_count,
                    };
                    break;
                }
            }
        }
    }

    fn step_branch_b(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, simulation_speed, max_points, mode, custom_flow) = self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        for _ in 0..simulation_speed {
            if !self.advance_tick(dt, max_points, mode, custom_flow.as_deref(), false) {
                break;
            }
        }
        if let Some(ref mut twin) = self.twin {
            twin.observe_lockstep(self.step_count, dt, self.system.state.vec);
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
            if !self.advance_tick(dt, max_points, mode, custom_flow.as_deref(), true) {
                break;
            }
        }
    }

    fn step_once(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, _, max_points, mode, custom_flow) = self.sync_params();
        if self.advance_tick(dt, max_points, mode, custom_flow.as_deref(), true) {
            self.execution_state = SolverExecutionState::Paused;
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let v = self.system.state.vec;
        let (dx, dy, dz) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            match (p.mode, p.custom_flow.as_deref()) {
                (AttractorMode::Custom, Some(job)) => {
                    let rhs = job.eval_rhs(v);
                    (rhs.x, rhs.y, rhs.z)
                }
                _ => (
                    self.system.sigma * (v.y - v.x),
                    v.x * (self.system.rho - v.z) - v.y,
                    v.x * v.y - self.system.beta * v.z,
                ),
            }
        };
        let diverged = matches!(self.execution_state, SolverExecutionState::Diverged { .. });
        let sonification =
            crate::audio::mappings::attractor_frame([v.x, v.y, v.z], [dx, dy, dz], diverged);
        let snapshot = AttractorSnapshot {
            history: self.history.clone(),
            current_vec: v,
            step_count: self.step_count,
            execution_state: self.execution_state,
            branch: self.snapshot_branch(),
            twin: self.twin.clone(),
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
        self.capture_active_checkpoint_params()
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        self.capture_time_travel_checkpoint()
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        self.restore_time_travel_checkpoint(checkpoint);
    }

    fn rebuild_history(&mut self, front: &[TimeTravelCheckpoint], back: &[TimeTravelCheckpoint]) {
        self.rebuild_time_travel_history(front, back);
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        self.replay_time_travel_step(params);
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
        let _ = self.advance_tick(dt, max_points, mode, custom_flow.as_deref(), true);
    }
}
