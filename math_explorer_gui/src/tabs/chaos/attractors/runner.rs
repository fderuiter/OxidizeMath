#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Pre-allocated twin chaotic ODE background runner with `TimelineRingBuffer` scrubbing.

use super::flow::{AttractorMode, CompiledFlowJob};
use super::runner_types::{
    build_runner_diagnostic, build_runner_sonification, AttractorSharedParams, AttractorSnapshot,
    SolverExecutionState, SyncedStepParams, MAX_SIMULATION_SPEED,
};
use super::twin_state::{AttractorSystemKind, TwinFrame};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use crate::framework::{TimelineRingBuffer, DEFAULT_TIMELINE_RING_CAPACITY};
use math_explorer::physics::chaos::double_pendulum::{
    DoublePendulum, DoublePendulumBuilder, DoublePendulumState,
};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use math_explorer::physics::chaos::rossler::{RosslerBuilder, RosslerState, RosslerSystem};
use math_explorer::pure_math::analysis::ode::{RungeKutta4, Solver};
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

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
    pub pendulum_primary: DoublePendulum,
    /// Shadow double-pendulum solver.
    pub pendulum_shadow: DoublePendulum,
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
            history: VecDeque::with_capacity(DEFAULT_TIMELINE_RING_CAPACITY),
            shadow_history: VecDeque::with_capacity(DEFAULT_TIMELINE_RING_CAPACITY),
            timeline: TimelineRingBuffer::new(TwinFrame::default()),
            step_count: 0,
            execution_state: SolverExecutionState::Ready,
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
        self.execution_state = SolverExecutionState::Ready;
    }

    fn apply_coefficients(&mut self, p: &AttractorSharedParams) {
        self.system.sigma = p.sigma;
        self.system.rho = p.rho;
        self.system.beta = p.beta;
        self.shadow_lorenz.sigma = p.sigma;
        self.shadow_lorenz.rho = p.rho;
        self.shadow_lorenz.beta = p.beta;

        self.rossler_primary.a = p.extra.rossler_a;
        self.rossler_primary.b = p.extra.rossler_b;
        self.rossler_primary.c = p.extra.rossler_c;
        self.rossler_shadow.a = p.extra.rossler_a;
        self.rossler_shadow.b = p.extra.rossler_b;
        self.rossler_shadow.c = p.extra.rossler_c;

        for pend in [&mut self.pendulum_primary, &mut self.pendulum_shadow] {
            pend.m1 = p.extra.pendulum_m1;
            pend.m2 = p.extra.pendulum_m2;
            pend.l1 = p.extra.pendulum_l1;
            pend.l2 = p.extra.pendulum_l2;
            pend.g = p.extra.pendulum_g;
        }
    }

    fn restore_solver_coordinates(&mut self, primary: [f64; 4], shadow: [f64; 4]) {
        self.system.state.vec = Vector3::new(primary[0], primary[1], primary[2]);
        self.shadow_lorenz.state.vec = Vector3::new(shadow[0], shadow[1], shadow[2]);
        self.rossler_primary.state.vec = Vector3::new(primary[0], primary[1], primary[2]);
        self.rossler_shadow.state.vec = Vector3::new(shadow[0], shadow[1], shadow[2]);
        self.pendulum_primary.state.state.0 = primary;
        self.pendulum_shadow.state.state.0 = shadow;
    }

    fn sync_params(&mut self) -> SyncedStepParams {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        self.apply_coefficients(&p);
        let custom_flow = (p.mode == AttractorMode::Custom)
            .then_some(p.custom_flow)
            .flatten();
        SyncedStepParams {
            mode: p.mode,
            custom_flow,
            kind: p.system_kind,
            dt: p.dt,
            speed: p.simulation_speed.clamp(1, MAX_SIMULATION_SPEED),
            max_points: p.max_points.clamp(1, DEFAULT_TIMELINE_RING_CAPACITY),
            shadow_enabled: p.shadow_enabled,
        }
    }

    fn active_primary_vec(&self, mode: AttractorMode, kind: AttractorSystemKind) -> Vector3<f64> {
        match (mode, kind) {
            (AttractorMode::Custom, _) | (_, AttractorSystemKind::Lorenz) => self.system.state.vec,
            (_, AttractorSystemKind::Rossler) => self.rossler_primary.state.vec,
            (_, AttractorSystemKind::DoublePendulum) => {
                self.pendulum_primary.state.phase_point_3d()
            }
        }
    }

    fn active_shadow_vec(&self, mode: AttractorMode, kind: AttractorSystemKind) -> Vector3<f64> {
        match (mode, kind) {
            (AttractorMode::Custom, _) | (_, AttractorSystemKind::Lorenz) => {
                self.shadow_lorenz.state.vec
            }
            (_, AttractorSystemKind::Rossler) => self.rossler_shadow.state.vec,
            (_, AttractorSystemKind::DoublePendulum) => self.pendulum_shadow.state.phase_point_3d(),
        }
    }

    fn parameters_finite(&self, mode: AttractorMode, kind: AttractorSystemKind, dt: f64) -> bool {
        dt.is_finite()
            && dt > 0.0
            && (mode == AttractorMode::Custom
                || match kind {
                    AttractorSystemKind::Lorenz => {
                        self.system.sigma.is_finite()
                            && self.system.rho.is_finite()
                            && self.system.beta.is_finite()
                    }
                    AttractorSystemKind::Rossler => {
                        self.rossler_primary.a.is_finite()
                            && self.rossler_primary.b.is_finite()
                            && self.rossler_primary.c.is_finite()
                    }
                    AttractorSystemKind::DoublePendulum => {
                        self.pendulum_primary.m1.is_finite()
                            && self.pendulum_primary.m2.is_finite()
                            && self.pendulum_primary.l1.is_finite()
                            && self.pendulum_primary.l2.is_finite()
                            && self.pendulum_primary.g.is_finite()
                    }
                })
    }

    fn step_custom_flow(
        &mut self,
        custom_flow: Option<&CompiledFlowJob>,
        dt: f64,
        shadow_enabled: bool,
    ) -> ([f64; 4], [f64; 4]) {
        let mut pv = self.system.state.vec;
        let mut sv = self.shadow_lorenz.state.vec;
        if let Some(job) = custom_flow {
            pv = job.rk4_step(pv, dt);
            self.system.state.vec = pv;
            if shadow_enabled {
                sv = job.rk4_step(sv, dt);
                self.shadow_lorenz.state.vec = sv;
            } else {
                sv = pv;
            }
        }
        ([pv.x, pv.y, pv.z, 0.0], [sv.x, sv.y, sv.z, 0.0])
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

    fn truncate_trails_to_cursor(&mut self) {
        let cursor_trail_len = self.timeline.cursor();
        while self.history.len() > cursor_trail_len {
            self.history.pop_back();
        }
        while self.shadow_history.len() > cursor_trail_len {
            self.shadow_history.pop_back();
        }
    }

    fn push_frame_to_trails(&mut self, frame: TwinFrame, max_points: usize) {
        let cap = max_points.min(DEFAULT_TIMELINE_RING_CAPACITY);
        if self.history.len() >= cap {
            self.history.pop_front();
        }
        self.history.push_back(frame.primary_vec3());
        if self.shadow_history.len() >= cap {
            self.shadow_history.pop_front();
        }
        self.shadow_history.push_back(frame.shadow_vec3());
    }

    fn advance_tick(&mut self, sp: &SyncedStepParams) -> bool {
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
        true
    }

    fn restore_frame(&mut self, frame: TwinFrame) {
        self.restore_solver_coordinates(frame.primary, frame.shadow);
        self.step_count = frame.step;
        self.history.clear();
        self.shadow_history.clear();
        let cap = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .max_points
            .clamp(1, DEFAULT_TIMELINE_RING_CAPACITY);
        for item in self.timeline.iter_up_to_cursor().skip(1) {
            if self.history.len() >= cap {
                self.history.pop_front();
            }
            self.history.push_back(item.primary_vec3());
            if self.shadow_history.len() >= cap {
                self.shadow_history.pop_front();
            }
            self.shadow_history.push_back(item.shadow_vec3());
        }
        if !matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            self.execution_state = SolverExecutionState::Paused;
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
            SimCommand::StepBack(count) => {
                if let Some(frame) = self.timeline.step_back(count) {
                    self.restore_frame(frame);
                }
            }
            SimCommand::SeekFrame(index) => {
                if let Some(frame) = self.timeline.seek(index) {
                    self.restore_frame(frame);
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
            _ => {}
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
            if !self.advance_tick(&sp) {
                break;
            }
        }
    }

    fn step_once(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let sp = self.sync_params();
        if self.advance_tick(&sp) {
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
}
