#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Pre-allocated twin chaotic ODE background runner with `CheckpointRing` and `TimelineRingBuffer` scrubbing.

use super::flow::AttractorMode;
use super::runner_types::{
    build_runner_diagnostic, build_runner_sonification, AttractorSharedParams, AttractorSnapshot,
    SolverExecutionState, SyncedStepParams, MAX_SIMULATION_SPEED,
};
use super::twin_state::{AttractorSystemKind, TwinFrame};
use crate::async_sim::{
    CheckpointParams, SimCommand, SimulationRunner, StateSnapshot, TimeTravelCheckpoint,
};
use crate::framework::recovery::DiagnosticReport;
use crate::framework::{TimelineRingBuffer, DEFAULT_TIMELINE_RING_CAPACITY};
use math_explorer::physics::chaos::double_pendulum::{
    DoublePendulumBuilder, DoublePendulumState, DoublePendulumSystem,
};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use math_explorer::physics::chaos::rossler::{RosslerBuilder, RosslerState, RosslerSystem};
use math_explorer::pure_math::analysis::ode::{RungeKutta4, Solver};
use nalgebra::Vector3;
use oxidize_core::rng::OxidizeRng;
use std::collections::VecDeque;
use std::sync::Arc;

const ATTRACTOR_RNG_SEED: u64 = 0x4C4F_5245_4E5A_3139;

/// Zero-allocation twin chaotic ODE runner backed by pre-allocated `RungeKutta4`
/// buffers, `CompiledFlowJob` bytecode, and a `4,096`-frame [`TimelineRingBuffer`].
pub struct AttractorRunner {
    /// Primary Lorenz / custom-flow state solver.
    pub system: LorenzSystem,
    /// Shadow Lorenz / custom-flow state solver.
    pub shadow_lorenz: LorenzSystem,
    /// Primary Rössler solver.
    pub rossler_primary: RosslerSystem,
    /// Shadow Rössler solver.
    pub rossler_shadow: RosslerSystem,
    /// Primary double-pendulum solver.
    pub pendulum_primary: DoublePendulumSystem,
    /// Shadow double-pendulum solver.
    pub pendulum_shadow: DoublePendulumSystem,
    rk4_vec3: RungeKutta4<Vector3<f64>>,
    /// Shared configuration synchronized from UI controls.
    pub params: Arc<std::sync::RwLock<AttractorSharedParams>>,
    /// Pre-allocated primary trail buffer (`4,096` slots).
    pub history: VecDeque<Vector3<f64>>,
    /// Pre-allocated shadow trail buffer (`4,096` slots).
    pub shadow_history: VecDeque<Vector3<f64>>,
    /// Pre-allocated `4,096`-frame timeline ring buffer.
    pub timeline: TimelineRingBuffer<TwinFrame>,
    /// Active simulation step count.
    pub step_count: usize,
    /// Current execution state.
    pub execution_state: SolverExecutionState,
    /// Optional mid-flight counterfactual shadow twin (`Branch A`).
    pub twin: Option<super::AttractorTwinState>,
    /// Deterministic PRNG captured across time-travel checkpoints.
    pub rng: OxidizeRng,
}

impl AttractorRunner {
    /// Constructs a pre-allocated twin runner from shared parameters.
    #[must_use]
    pub fn from_params(params: Arc<std::sync::RwLock<AttractorSharedParams>>) -> Self {
        let p = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let [x0, y0, z0] = p.initial_state;
        let init_lorenz = LorenzState::new(x0, y0, z0);
        let mut runner = Self {
            system: LorenzBuilder::new()
                .sigma(p.sigma)
                .rho(p.rho)
                .beta(p.beta)
                .build(init_lorenz),
            shadow_lorenz: LorenzBuilder::new().build(init_lorenz),
            rossler_primary: RosslerBuilder::new().build(RosslerState::new(0.1, 0.0, 0.0)),
            rossler_shadow: RosslerBuilder::new().build(RosslerState::new(0.1, 0.0, 0.0)),
            pendulum_primary: DoublePendulumBuilder::new()
                .build(DoublePendulumState::new(2.4, 2.4, 0.0, 0.0)),
            pendulum_shadow: DoublePendulumBuilder::new()
                .build(DoublePendulumState::new(2.4, 2.4, 0.0, 0.0)),
            rk4_vec3: RungeKutta4::new(&init_lorenz.vec),
            params,
            history: VecDeque::with_capacity(super::ATTRACTOR_RING_CAPACITY),
            shadow_history: VecDeque::with_capacity(DEFAULT_TIMELINE_RING_CAPACITY),
            timeline: TimelineRingBuffer::new(TwinFrame::default()),
            step_count: 0,
            execution_state: SolverExecutionState::Ready,
            twin: None,
            rng: OxidizeRng::new(ATTRACTOR_RNG_SEED),
        };
        runner.reset_from_shared(&p);
        runner
    }

    fn reset_from_shared(&mut self, p: &AttractorSharedParams) {
        self.apply_coefficients(p);
        let init_p = [
            p.initial_state[0],
            p.initial_state[1],
            p.initial_state[2],
            0.0,
        ];
        let off = if p.shadow_enabled {
            p.perturbation_offset
        } else {
            [0.0; 3]
        };
        let init_s = [
            init_p[0] + off[0],
            init_p[1] + off[1],
            init_p[2] + off[2],
            0.0,
        ];
        self.restore_solver_coordinates(init_p, init_s);
        self.history.clear();
        self.shadow_history.clear();
        self.timeline.clear();
        self.timeline.push(TwinFrame::from_states(
            0,
            0.0,
            init_p,
            init_s,
            p.shadow_enabled,
        ));
        self.step_count = 0;
        self.twin = None;
        self.execution_state = SolverExecutionState::Ready;
        self.rng = OxidizeRng::new(ATTRACTOR_RNG_SEED);
    }

    fn step_active_solvers(&mut self, sp: &SyncedStepParams) -> ([f64; 4], [f64; 4]) {
        if sp.mode == AttractorMode::Custom {
            return self.step_custom_flow(sp.custom_flow.as_deref(), sp.dt, sp.shadow_enabled);
        }
        match sp.kind {
            AttractorSystemKind::Lorenz => {
                let mut pv = self.system.state.vec;
                self.rk4_vec3.step(&self.system, 0.0, &mut pv, sp.dt);
                self.system.state.vec = pv;
                let mut sv = self.shadow_lorenz.state.vec;
                if sp.shadow_enabled {
                    self.rk4_vec3.step(&self.shadow_lorenz, 0.0, &mut sv, sp.dt);
                    self.shadow_lorenz.state.vec = sv;
                } else {
                    sv = pv;
                }
                ([pv.x, pv.y, pv.z, 0.0], [sv.x, sv.y, sv.z, 0.0])
            }
            AttractorSystemKind::Rossler => {
                self.rossler_primary.step(sp.dt);
                let pv = self.rossler_primary.state.vec;
                let sv = if sp.shadow_enabled {
                    self.rossler_shadow.step(sp.dt);
                    self.rossler_shadow.state.vec
                } else {
                    pv
                };
                ([pv.x, pv.y, pv.z, 0.0], [sv.x, sv.y, sv.z, 0.0])
            }
            AttractorSystemKind::DoublePendulum => {
                self.pendulum_primary.step(sp.dt);
                let p4 = self.pendulum_primary.state.state.0;
                let s4 = if sp.shadow_enabled {
                    self.pendulum_shadow.step(sp.dt);
                    self.pendulum_shadow.state.state.0
                } else {
                    p4
                };
                (p4, s4)
            }
        }
    }

    pub(super) fn advance_tick(&mut self, sp: &SyncedStepParams, step_twin_too: bool) -> bool {
        let pre_vec = self.active_primary_vec(sp.mode, sp.kind);
        if !pre_vec.x.is_finite()
            || !pre_vec.y.is_finite()
            || !pre_vec.z.is_finite()
            || !self.parameters_finite(sp.mode, sp.kind, sp.dt)
        {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: pre_vec,
                diverged_at_step: self.step_count,
            };
            return false;
        }
        if !self.step_twin_if_enabled(sp.dt, sp.max_points, step_twin_too) {
            return false;
        }

        self.truncate_trails_to_cursor();
        let (p4, s4) = self.step_active_solvers(sp);
        self.step_count += 1;

        if p4.iter().any(|v| !v.is_finite()) || s4.iter().any(|v| !v.is_finite()) {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: Vector3::new(p4[0], p4[1], p4[2]),
                diverged_at_step: self.step_count,
            };
            return false;
        }

        let frame = TwinFrame::from_states(
            self.step_count,
            self.step_count as f64 * sp.dt,
            p4,
            s4,
            sp.shadow_enabled,
        );
        self.timeline.push(frame);
        self.push_frame_to_trails(frame, sp.max_points);
        if step_twin_too {
            if let Some(ref mut twin) = self.twin {
                twin.observe_lockstep(self.step_count, sp.dt, frame.primary_vec3());
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
            SimCommand::StepBack(count) => {
                if let Some(frame) = self.timeline.step_back(count) {
                    self.restore_frame(frame);
                }
            }
            SimCommand::SeekStep(target_step) => {
                let pos = self.timeline.iter_all().position(|f| f.step == target_step);
                if let Some(pos) = pos {
                    if let Some(frame) = self.timeline.seek(pos) {
                        self.restore_frame(frame);
                    }
                }
            }
            SimCommand::Reset => {
                let p = self
                    .params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                self.reset_from_shared(&p);
            }
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
        let sp = self.sync_params();
        if let Some(ref mut twin) = self.twin {
            for _ in 0..sp.speed {
                if !twin.step_branch_a(sp.dt, sp.max_points) {
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
        let sp = self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        for _ in 0..sp.speed {
            if !self.advance_tick(&sp, false) {
                break;
            }
        }
        let vb = self.active_primary_vec(sp.mode, sp.kind);
        if let Some(ref mut twin) = self.twin {
            twin.observe_lockstep(self.step_count, sp.dt, vb);
        }
    }

    fn step(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let sp = self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        for _ in 0..sp.speed {
            if !self.advance_tick(&sp, true) {
                break;
            }
        }
    }

    fn step_once(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let sp = self.sync_params();
        if self.advance_tick(&sp, true) {
            self.execution_state = SolverExecutionState::Paused;
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let (v, dt, speed, sonification) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            let v = self.active_primary_vec(p.mode, p.system_kind);
            let sonification = build_runner_sonification(self, &p, v);
            (
                v,
                p.dt,
                p.simulation_speed.clamp(1, MAX_SIMULATION_SPEED),
                sonification,
            )
        };
        super::invariants::publish_runner_landmarks(self, dt, speed);
        AttractorSnapshot {
            history: self.history.clone(),
            shadow_history: self.shadow_history.clone(),
            timeline: self.timeline.clone(),
            current_vec: v,
            step_count: self.step_count,
            execution_state: self.execution_state,
            twin: self.twin.clone(),
        }
        .into_state_snapshot(sonification)
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        let (mode, kind) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.mode, p.system_kind)
        };
        let v = self.active_primary_vec(mode, kind);
        let s = self.active_shadow_vec(mode, kind);
        !matches!(self.execution_state, SolverExecutionState::Diverged { .. })
            && v.x.is_finite()
            && v.y.is_finite()
            && v.z.is_finite()
            && s.x.is_finite()
            && s.y.is_finite()
            && s.z.is_finite()
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
        build_runner_diagnostic(
            self.execution_state,
            self.step_count,
            dt,
            self.is_finite_state(),
        )
    }
    fn checkpoint_stride(&self) -> Option<usize> {
        Some(1)
    }

    fn current_step(&self) -> usize {
        self.step_count
    }

    fn capture_active_params(&self) -> Option<CheckpointParams> {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        Some(Self::pack_checkpoint_params(&p))
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        self.capture_twin_checkpoint()
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        self.restore_twin_checkpoint(checkpoint);
    }

    fn rebuild_history(&mut self, front: &[TimeTravelCheckpoint], back: &[TimeTravelCheckpoint]) {
        self.rebuild_twin_history(front, back);
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        let Some(ps) = params.as_scalars() else {
            return;
        };
        let Some(sp) = self.unpack_checkpoint_params(ps) else {
            return;
        };
        if self.advance_tick(&sp, false) {
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
        let sp = self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        if self.advance_tick(&sp, true) {
            super::invariants::publish_runner_landmarks(self, sp.dt, 1);
        }
    }
}
