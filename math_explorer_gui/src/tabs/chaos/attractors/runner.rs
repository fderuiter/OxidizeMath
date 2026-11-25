#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::flow::{AttractorMode, CompiledFlowJob};
use super::serde_vec3;
use super::twin::{is_vec_finite, step_flow_vec, TwinConfig, TwinRunnerState};
use crate::app::SharedSimulationBus;
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
    pub twin: TwinConfig,
    pub sim_bus: Option<SharedSimulationBus>,
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
            sim_bus: None,
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
    pub twin: Option<super::AttractorTwinState>,
}

impl AttractorSnapshot {
    /// Estimates heap bytes owned by this snapshot's `VecDeque` buffers for `SnapshotHistory`.
    #[must_use]
    pub fn estimated_heap_bytes(&self) -> usize {
        let branch_a_cap = self.twin.as_ref().map_or(0, |t| t.history_a.capacity());
        std::mem::size_of::<Self>()
            + (self.history.capacity() + self.twin_history.capacity() + branch_a_cap)
                * std::mem::size_of::<Vector3<f64>>()
            + self.divergence_history.capacity() * std::mem::size_of::<[f64; 2]>()
    }
}

#[allow(missing_docs)]
pub struct AttractorRunner {
    pub system: LorenzSystem,
    pub params: Arc<std::sync::RwLock<AttractorSharedParams>>,
    pub history: VecDeque<Vector3<f64>>,
    pub twin_state: TwinRunnerState,
    pub step_count: usize,
    pub execution_state: SolverExecutionState,
    pub twin: Option<super::AttractorTwinState>,
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
            history: VecDeque::with_capacity(max_points.max(super::ATTRACTOR_RING_CAPACITY)),
            twin_state: TwinRunnerState::from_config(twin_cfg, max_points),
            step_count: 0,
            execution_state: SolverExecutionState::Ready,
            twin: None,
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
        self.twin_state.enabled = p.twin.enabled;
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
        let twin_ok = !twin_enabled || is_vec_finite(self.twin_state.shadow_vec);
        if is_vec_finite(current_vec) && lorenz_ok && twin_ok && dt.is_finite() {
            return true;
        }
        self.execution_state = SolverExecutionState::Diverged {
            invalid_vec: if twin_ok {
                current_vec
            } else {
                self.twin_state.shadow_vec
            },
            diverged_at_step: self.step_count,
        };
        false
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
        twin_enabled: bool,
        custom_flow: Option<&CompiledFlowJob>,
        step_twin_too: bool,
    ) -> bool {
        if !self.check_pre_tick(dt, mode, twin_enabled)
            || !self.step_twin_if_enabled(dt, max_points, step_twin_too)
        {
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
                self.twin_state.shadow_vec,
            )
        } else {
            self.twin_state.shadow_vec
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
        self.record_tick_history(
            dt,
            max_points,
            twin_enabled,
            step_twin_too,
            vec,
            next_shadow,
        );
        true
    }

    fn record_tick_history(
        &mut self,
        dt: f64,
        max_points: usize,
        twin_enabled: bool,
        step_twin_too: bool,
        vec: Vector3<f64>,
        next_shadow: Vector3<f64>,
    ) {
        let cap = max_points.clamp(1, super::ATTRACTOR_RING_CAPACITY);
        if self.history.len() >= cap {
            self.history.pop_front();
        }
        self.history.push_back(vec);
        self.twin_state.enabled = twin_enabled;
        self.twin_state
            .record_step(self.step_count, dt, cap, vec, next_shadow);
        if step_twin_too {
            if let Some(ref mut twin) = self.twin {
                twin.observe_lockstep(self.step_count, dt, vec);
            }
        }
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
            SimCommand::Reset => self.reset_runner(),
            SimCommand::SetSpeed(speed) => {
                self.params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .simulation_speed = speed.clamp(1, MAX_SIMULATION_SPEED);
            }
            SimCommand::Custom(ref cmd) => self.handle_custom_command(cmd),
            _ => {}
        }
    }

    fn has_counterfactual_twin(&self) -> bool {
        self.twin.is_some()
    }

    fn step_branch_a(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, simulation_speed, max_points, _, _, _) = self.sync_params();
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
        let (dt, simulation_speed, max_points, mode, twin_enabled, custom_flow) =
            self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        for _ in 0..simulation_speed {
            if !self.advance_tick(
                dt,
                max_points,
                mode,
                twin_enabled,
                custom_flow.as_deref(),
                false,
            ) {
                break;
            }
        }
        if let Some(ref mut twin) = self.twin {
            twin.observe_lockstep(self.step_count, dt, self.system.state.vec);
        }
    }

    fn step(&mut self) {
        self.step_sliced(None);
    }

    fn step_sliced(&mut self, max_slice: Option<std::time::Duration>) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, simulation_speed, max_points, mode, twin_enabled, custom_flow) =
            self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        let slice_start = max_slice.map(|_| crate::web_io::PlatformInstant::now());
        let mut advanced = 0_usize;
        for tick_idx in 0..simulation_speed {
            if tick_idx > 0 {
                if let (Some(budget), Some(start)) = (max_slice, slice_start) {
                    if start.elapsed() >= budget {
                        break;
                    }
                }
            }
            if !self.advance_tick(
                dt,
                max_points,
                mode,
                twin_enabled,
                custom_flow.as_deref(),
                true,
            ) {
                break;
            }
            advanced += 1;
        }
        if advanced > 0 && self.history.len() >= 16 && self.step_count % 15 < advanced {
            self.publish_live_landmarks(dt);
        }
    }

    fn step_once(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, _, max_points, mode, twin_enabled, custom_flow) = self.sync_params();
        if self.advance_tick(
            dt,
            max_points,
            mode,
            twin_enabled,
            custom_flow.as_deref(),
            true,
        ) {
            self.execution_state = SolverExecutionState::Paused;
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        self.build_snapshot()
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        let v = self.system.state.vec;
        let sv = self.twin_state.shadow_vec;
        let shadow_ok =
            !self.twin_state.enabled || (sv.x.is_finite() && sv.y.is_finite() && sv.z.is_finite());
        !matches!(self.execution_state, SolverExecutionState::Diverged { .. })
            && v.x.is_finite()
            && v.y.is_finite()
            && v.z.is_finite()
            && shadow_ok
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        self.build_diagnostic_report()
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
        let (dt, _, max_points, mode, twin_enabled, custom_flow) = self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        if self.advance_tick(
            dt,
            max_points,
            mode,
            twin_enabled,
            custom_flow.as_deref(),
            true,
        ) && self.history.len() >= 16
            && self.step_count.is_multiple_of(15)
        {
            self.publish_live_landmarks(dt);
        }
    }
}
