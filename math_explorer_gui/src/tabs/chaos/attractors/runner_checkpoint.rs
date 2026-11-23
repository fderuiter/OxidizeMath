#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Zero-allocation inline scalar checkpoint capture, restore, and sub-step replay
//! for [`super::AttractorRunner`].

use super::flow::{AttractorMode, CompiledFlowJob};
use super::runner_types::{
    AttractorSharedParams, SolverExecutionState, SyncedStepParams, MAX_SIMULATION_SPEED,
};
use super::twin_state::{AttractorSystemKind, TwinFrame};
use super::AttractorRunner;
use crate::async_sim::{CheckpointParams, CheckpointPayload, TimeTravelCheckpoint};
use crate::framework::DEFAULT_TIMELINE_RING_CAPACITY;
use nalgebra::Vector3;

impl AttractorRunner {
    pub(super) fn active_state_pair(
        &self,
        mode: AttractorMode,
        kind: AttractorSystemKind,
    ) -> ([f64; 4], [f64; 4]) {
        match (mode, kind) {
            (AttractorMode::Custom, _) | (_, AttractorSystemKind::Lorenz) => {
                let p = self.system.state.vec;
                let s = self.shadow_lorenz.state.vec;
                ([p.x, p.y, p.z, 0.0], [s.x, s.y, s.z, 0.0])
            }
            (_, AttractorSystemKind::Rossler) => {
                let p = self.rossler_primary.state.vec;
                let s = self.rossler_shadow.state.vec;
                ([p.x, p.y, p.z, 0.0], [s.x, s.y, s.z, 0.0])
            }
            (_, AttractorSystemKind::DoublePendulum) => (
                self.pendulum_primary.state.state.0,
                self.pendulum_shadow.state.state.0,
            ),
        }
    }

    pub(super) fn pack_checkpoint_params(p: &AttractorSharedParams) -> CheckpointParams {
        let (c0, c1, c2, c3, c4) = match p.system_kind {
            AttractorSystemKind::Lorenz => (p.sigma, p.rho, p.beta, 0.0, 0.0),
            AttractorSystemKind::Rossler => (
                p.extra.rossler_a,
                p.extra.rossler_b,
                p.extra.rossler_c,
                0.0,
                0.0,
            ),
            AttractorSystemKind::DoublePendulum => (
                p.extra.pendulum_m1,
                p.extra.pendulum_m2,
                p.extra.pendulum_l1,
                p.extra.pendulum_l2,
                p.extra.pendulum_g,
            ),
        };
        let mode_flag = match p.mode {
            AttractorMode::Lorenz => 0.0,
            AttractorMode::Custom => 1.0,
        };
        let kind_flag = match p.system_kind {
            AttractorSystemKind::Lorenz => 0.0,
            AttractorSystemKind::Rossler => 1.0,
            AttractorSystemKind::DoublePendulum => 2.0,
        };
        let shadow_flag = if p.shadow_enabled { 1.0 } else { 0.0 };
        CheckpointParams::scalars(&[
            c0,
            c1,
            c2,
            c3,
            c4,
            p.dt,
            p.max_points as f64,
            mode_flag,
            kind_flag,
            shadow_flag,
        ])
    }

    pub(super) fn unpack_checkpoint_params(&mut self, scalars: &[f64]) -> Option<SyncedStepParams> {
        if scalars.len() < 10 {
            return None;
        }
        let mode = if scalars[7] > 0.5 {
            AttractorMode::Custom
        } else {
            AttractorMode::Lorenz
        };
        let kind = if scalars[8] > 1.5 {
            AttractorSystemKind::DoublePendulum
        } else if scalars[8] > 0.5 {
            AttractorSystemKind::Rossler
        } else {
            AttractorSystemKind::Lorenz
        };
        let shadow_enabled = scalars[9] > 0.5;
        self.apply_packed_coefficients(kind, scalars);

        let dt = scalars[5];
        let max_points = (scalars[6] as usize).clamp(1, DEFAULT_TIMELINE_RING_CAPACITY);
        let custom_flow = (mode == AttractorMode::Custom)
            .then(|| {
                self.params
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .custom_flow
                    .clone()
            })
            .flatten();
        Some(SyncedStepParams {
            mode,
            custom_flow,
            kind,
            dt,
            speed: 1,
            max_points,
            shadow_enabled,
        })
    }

    fn apply_packed_coefficients(&mut self, kind: AttractorSystemKind, scalars: &[f64]) {
        match kind {
            AttractorSystemKind::Lorenz => {
                self.system.sigma = scalars[0];
                self.system.rho = scalars[1];
                self.system.beta = scalars[2];
                self.shadow_lorenz.sigma = scalars[0];
                self.shadow_lorenz.rho = scalars[1];
                self.shadow_lorenz.beta = scalars[2];
            }
            AttractorSystemKind::Rossler => {
                self.rossler_primary.a = scalars[0];
                self.rossler_primary.b = scalars[1];
                self.rossler_primary.c = scalars[2];
                self.rossler_shadow.a = scalars[0];
                self.rossler_shadow.b = scalars[1];
                self.rossler_shadow.c = scalars[2];
            }
            AttractorSystemKind::DoublePendulum => {
                for pend in [&mut self.pendulum_primary, &mut self.pendulum_shadow] {
                    pend.m1 = scalars[0];
                    pend.m2 = scalars[1];
                    pend.l1 = scalars[2];
                    pend.l2 = scalars[3];
                    pend.g = scalars[4];
                }
            }
        }
    }

    pub(super) fn capture_twin_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let (params, mode, kind) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (Self::pack_checkpoint_params(&p), p.mode, p.system_kind)
        };
        let (p4, s4) = self.active_state_pair(mode, kind);
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params,
            state: CheckpointPayload::scalars(&[
                p4[0], p4[1], p4[2], p4[3], s4[0], s4[1], s4[2], s4[3],
            ]),
        })
    }

    pub(super) fn restore_twin_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let Some(ps) = checkpoint.params.as_scalars() {
            let _ = self.unpack_checkpoint_params(ps);
        }
        if let Some(s) = checkpoint.state.as_scalars() {
            if s.len() >= 8 {
                self.restore_solver_coordinates([s[0], s[1], s[2], s[3]], [s[4], s[5], s[6], s[7]]);
            }
        }
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
        self.twin = None;
        let pos = self
            .timeline
            .iter_all()
            .position(|f| f.step == checkpoint.step);
        if let Some(pos) = pos {
            let _ = self.timeline.seek(pos);
        }
        self.execution_state = if checkpoint.step == 0 {
            SolverExecutionState::Ready
        } else {
            SolverExecutionState::Paused
        };
    }

    pub(super) fn rebuild_twin_history(
        &mut self,
        front: &[TimeTravelCheckpoint],
        back: &[TimeTravelCheckpoint],
    ) {
        let tail = back.last().or_else(|| front.last());
        let tail_scalars = tail.and_then(|cp| cp.params.as_scalars());
        let dt = tail_scalars.and_then(|s| s.get(5).copied()).unwrap_or(0.01);
        let max_points = tail_scalars
            .and_then(|s| s.get(6).copied())
            .map_or(2000usize, |v| {
                (v as usize).clamp(1, DEFAULT_TIMELINE_RING_CAPACITY)
            });
        let shadow_enabled = tail_scalars.and_then(|s| s.get(9).copied()).unwrap_or(1.0) > 0.5;

        self.history.clear();
        self.shadow_history.clear();
        let total_post_zero = front.iter().chain(back).filter(|cp| cp.step >= 1).count();
        let skip = total_post_zero.saturating_sub(max_points);
        for cp in front
            .iter()
            .chain(back)
            .filter(|cp| cp.step >= 1)
            .skip(skip)
        {
            if let Some(s) = cp.state.as_scalars().filter(|s| s.len() >= 8) {
                self.history.push_back(Vector3::new(s[0], s[1], s[2]));
                self.shadow_history
                    .push_back(Vector3::new(s[4], s[5], s[6]));
            }
        }

        if self.timeline.current_copied().map(|f| f.step) != Some(self.step_count) {
            self.rebuild_timeline_from_checkpoints(front, back, dt, shadow_enabled);
        }
    }

    fn rebuild_timeline_from_checkpoints(
        &mut self,
        front: &[TimeTravelCheckpoint],
        back: &[TimeTravelCheckpoint],
        dt: f64,
        shadow_enabled: bool,
    ) {
        let total = front.len() + back.len();
        let skip = total.saturating_sub(DEFAULT_TIMELINE_RING_CAPACITY);
        self.timeline.clear();
        for cp in front.iter().chain(back).skip(skip) {
            if let Some(s) = cp.state.as_scalars().filter(|s| s.len() >= 8) {
                self.timeline.push(TwinFrame::from_states(
                    cp.step,
                    cp.step as f64 * dt,
                    [s[0], s[1], s[2], s[3]],
                    [s[4], s[5], s[6], s[7]],
                    shadow_enabled,
                ));
            }
        }
    }

    pub(super) fn active_primary_vec(
        &self,
        mode: AttractorMode,
        kind: AttractorSystemKind,
    ) -> Vector3<f64> {
        match (mode, kind) {
            (AttractorMode::Custom, _) | (_, AttractorSystemKind::Lorenz) => self.system.state.vec,
            (_, AttractorSystemKind::Rossler) => self.rossler_primary.state.vec,
            (_, AttractorSystemKind::DoublePendulum) => {
                self.pendulum_primary.state.phase_point_3d()
            }
        }
    }

    pub(super) fn active_shadow_vec(
        &self,
        mode: AttractorMode,
        kind: AttractorSystemKind,
    ) -> Vector3<f64> {
        match (mode, kind) {
            (AttractorMode::Custom, _) | (_, AttractorSystemKind::Lorenz) => {
                self.shadow_lorenz.state.vec
            }
            (_, AttractorSystemKind::Rossler) => self.rossler_shadow.state.vec,
            (_, AttractorSystemKind::DoublePendulum) => self.pendulum_shadow.state.phase_point_3d(),
        }
    }

    pub(super) fn parameters_finite(
        &self,
        mode: AttractorMode,
        kind: AttractorSystemKind,
        dt: f64,
    ) -> bool {
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

    pub(super) fn truncate_trails_to_cursor(&mut self) {
        let cursor_trail_len = self.timeline.cursor();
        while self.history.len() > cursor_trail_len {
            self.history.pop_back();
        }
        while self.shadow_history.len() > cursor_trail_len {
            self.shadow_history.pop_back();
        }
    }

    pub(super) fn push_frame_to_trails(&mut self, frame: TwinFrame, max_points: usize) {
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

    pub(super) fn restore_frame(&mut self, frame: TwinFrame) {
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

    pub(super) fn apply_coefficients(&mut self, p: &AttractorSharedParams) {
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

    pub(super) fn restore_solver_coordinates(&mut self, primary: [f64; 4], shadow: [f64; 4]) {
        self.system.state.vec = Vector3::new(primary[0], primary[1], primary[2]);
        self.shadow_lorenz.state.vec = Vector3::new(shadow[0], shadow[1], shadow[2]);
        self.rossler_primary.state.vec = Vector3::new(primary[0], primary[1], primary[2]);
        self.rossler_shadow.state.vec = Vector3::new(shadow[0], shadow[1], shadow[2]);
        self.pendulum_primary.state.state.0 = primary;
        self.pendulum_shadow.state.state.0 = shadow;
    }

    pub(super) fn sync_params(&mut self) -> SyncedStepParams {
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

    pub(super) fn step_twin_if_enabled(
        &mut self,
        dt: f64,
        max_points: usize,
        enabled: bool,
    ) -> bool {
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

    pub(super) fn step_custom_flow(
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
}
