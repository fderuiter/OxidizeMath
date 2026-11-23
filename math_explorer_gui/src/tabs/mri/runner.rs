#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::step_bloch_rotating_frame;
use crate::async_sim::{
    CheckpointParams, CheckpointPayload, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use crate::framework::recovery::DiagnosticReport;
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use nalgebra::Vector3;
use oxidize_core::rng::OxidizeRng;
use std::collections::VecDeque;
use std::sync::Arc;

/// Minimum valid simulation speed multiplier for MRI Bloch evolution.
pub const MIN_MRI_TIME_SCALE: f64 = 1.0;
/// Maximum valid simulation speed multiplier for MRI Bloch evolution.
pub const MAX_MRI_TIME_SCALE: f64 = 100.0;
/// Hard upper bound on Bloch integration sub-steps executed per worker frame.
pub const MAX_MRI_STEPS_PER_FRAME: usize = 100;

/// Sanitizes an MRI `time_scale` value into `1.0..=100.0`, defaulting non-finite inputs to `1.0`.
#[must_use]
pub fn sanitize_mri_time_scale(time_scale: f64) -> f64 {
    if time_scale.is_finite() {
        time_scale.clamp(MIN_MRI_TIME_SCALE, MAX_MRI_TIME_SCALE)
    } else {
        MIN_MRI_TIME_SCALE
    }
}

#[allow(missing_docs)]
pub struct MriSharedParams {
    pub t1: f64,
    pub t2: f64,
    pub b_field: Vector3<f64>,
    pub time_scale: f64,
}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct MriSnapshot {
    pub current_time: f64,
    pub magnetization: Vector3<f64>,
    pub history: VecDeque<(f64, Vector3<f64>)>,
    pub twin: Option<super::MriTwinState>,
}

#[allow(missing_docs)]
pub struct MriRunner {
    pub simulator: BlochSimulator,
    pub solver: RungeKutta4<Vector3<f64>>,
    pub params: Arc<std::sync::RwLock<MriSharedParams>>,
    pub current_time: f64,
    pub history: VecDeque<(f64, Vector3<f64>)>,
    pub max_history: usize,
    pub twin: Option<super::MriTwinState>,
    pub step_count: usize,
    pub rng: OxidizeRng,
}

impl MriRunner {
    /// Creates an initialized Bloch runner with pre-allocated `history` and seeded `t = 0` point.
    pub fn new(
        params: Arc<std::sync::RwLock<MriSharedParams>>,
        initial_m: Vector3<f64>,
        m0: f64,
    ) -> Self {
        let (t1, t2, b_field) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.t1, p.t2, p.b_field)
        };
        let mut simulator = BlochSimulator::new(initial_m, m0);
        simulator.set_relaxation(t1, t2);
        simulator.set_b_field(b_field);
        let max_history = math_commons::registry::MAX_ITERATIONS;
        let mut history = VecDeque::with_capacity(max_history);
        history.push_back((0.0, initial_m));
        Self {
            simulator,
            solver: RungeKutta4::new(&initial_m),
            params,
            current_time: 0.0,
            history,
            max_history,
            twin: None,
            step_count: 0,
            rng: OxidizeRng::new(0x4D52_495F_424C_4F43),
        }
    }

    pub(super) fn sync_params_and_get_scale(&mut self) -> f64 {
        let (t1, t2, b_field, time_scale) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.t1, p.t2, p.b_field, p.time_scale)
        };
        self.simulator.t1 = t1;
        self.simulator.t2 = t2;
        self.simulator.b_field = b_field;
        time_scale
    }

    fn advance_bloch_ticks(&mut self, ticks: usize, step_twin_too: bool) {
        let dt = 0.001;
        for _ in 0..ticks {
            let next_time = self.current_time + dt;
            if step_twin_too && let Some(ref mut twin) = self.twin {
                twin.step_branch_a(next_time, dt, self.max_history);
            }
            let used_spin_echo = self.twin.as_mut().is_some_and(|twin| {
                twin.step_branch_b_spin_echo(&mut self.simulator, &mut self.solver, dt)
            });
            if !used_spin_echo {
                step_bloch_rotating_frame(&mut self.simulator, &mut self.solver, dt, 0.0);
            }
            self.current_time = next_time;
            self.step_count += 1;
            let m = self.simulator.magnetization;
            if !m.x.is_finite() || !m.y.is_finite() || !m.z.is_finite() {
                break;
            }
            if self.history.len() >= self.max_history {
                self.history.pop_front();
            }
            self.history.push_back((self.current_time, m));
            if step_twin_too && let Some(ref mut twin) = self.twin {
                twin.observe_lockstep(self.current_time, m);
            }
        }
    }
}

impl SimulationRunner for MriRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                let _ = self.sync_params_and_get_scale();
                let initial_m = Vector3::new(self.simulator.m0, 0.0, 0.0);
                self.simulator.magnetization = initial_m;
                self.current_time = 0.0;
                self.step_count = 0;
                self.rng = OxidizeRng::new(0x4D52_495F_424C_4F43);
                self.history.clear();
                self.history.push_back((0.0, initial_m));
                self.twin = None;
            }
            SimCommand::Custom(ref custom) => self.handle_custom_command(custom),
            _ => {}
        }
    }

    fn has_counterfactual_twin(&self) -> bool {
        self.twin.is_some()
    }

    fn step_branch_a(&mut self) {
        let time_scale = self.sync_params_and_get_scale();
        let steps = if time_scale.is_finite() && time_scale > 0.0 {
            (time_scale.ceil().min(MAX_MRI_STEPS_PER_FRAME as f64) as usize).max(1)
        } else {
            1
        };
        if let Some(ref mut twin) = self.twin {
            for idx in 1..=steps {
                let next_time = self.current_time + idx as f64 * 0.001;
                twin.step_branch_a(next_time, 0.001, self.max_history);
            }
        }
    }

    fn step_branch_b(&mut self) {
        let time_scale = self.sync_params_and_get_scale();
        let steps = if time_scale.is_finite() && time_scale > 0.0 {
            (time_scale.ceil().min(MAX_MRI_STEPS_PER_FRAME as f64) as usize).max(1)
        } else {
            1
        };
        self.advance_bloch_ticks(steps, false);
        if let Some(ref mut twin) = self.twin {
            twin.observe_lockstep(self.current_time, self.simulator.magnetization);
        }
    }

    fn step(&mut self) {
        let time_scale = self.sync_params_and_get_scale();
        let steps = if time_scale.is_finite() && time_scale > 0.0 {
            (time_scale.ceil().min(MAX_MRI_STEPS_PER_FRAME as f64) as usize).max(1)
        } else {
            1
        };
        self.advance_bloch_ticks(steps, true);
    }

    fn step_once(&mut self) {
        let _ = self.sync_params_and_get_scale();
        self.advance_bloch_ticks(1, true);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let m = self.simulator.magnetization;
        let b_field = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .b_field;
        let sonification = if self.is_finite_state() {
            crate::audio::mappings::mri_bloch_frame(m, self.simulator.m0, b_field, 0.0)
        } else {
            crate::audio::SonificationFrame::silent()
        };
        let snapshot = MriSnapshot {
            current_time: self.current_time,
            magnetization: m,
            history: self.history.clone(),
            twin: self.twin.clone(),
        };
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![self.current_time, m.x, m.y, m.z],
            structured_data: Some(Box::new(snapshot)),
            sonification,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        let m = self.simulator.magnetization;
        self.current_time.is_finite() && m.x.is_finite() && m.y.is_finite() && m.z.is_finite()
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        if !self.is_finite_state() {
            Some(DiagnosticReport::new(
                self.current_time,
                self.history.len(),
                "magnetization",
                "Bloch magnetization vector became non-finite.",
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
        Some(CheckpointParams::scalars(&[
            self.simulator.m0,
            p.t1,
            p.t2,
            p.b_field.x,
            p.b_field.y,
            p.b_field.z,
        ]))
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let params = self.capture_active_params()?;
        let m = self.simulator.magnetization;
        let seeded_t0 = if self.step_count == 0 && !self.history.is_empty() {
            1.0
        } else {
            0.0
        };
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params,
            state: CheckpointPayload::scalars(&[m.x, m.y, m.z, self.current_time, seeded_t0]),
        })
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let Some(s) = checkpoint.state.as_scalars()
            && s.len() >= 4
        {
            self.simulator.magnetization = Vector3::new(s[0], s[1], s[2]);
            self.current_time = s[3];
        }
        if let Some(p) = checkpoint.params.as_scalars()
            && p.len() >= 6
        {
            self.simulator.m0 = p[0];
            self.simulator.t1 = p[1];
            self.simulator.t2 = p[2];
            self.simulator.b_field = Vector3::new(p[3], p[4], p[5]);
        }
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
        self.twin = None;
    }

    fn rebuild_history(&mut self, front: &[TimeTravelCheckpoint], back: &[TimeTravelCheckpoint]) {
        let include_zero = front
            .first()
            .and_then(|cp| (cp.step == 0).then(|| cp.state.as_scalars()).flatten())
            .and_then(|s| s.get(4))
            .is_some_and(|&flag| flag > 0.5);
        self.history.clear();
        let eligible = front
            .iter()
            .chain(back.iter())
            .filter(|cp| include_zero || cp.step >= 1)
            .count();
        let skip = eligible.saturating_sub(self.max_history);
        for cp in front
            .iter()
            .chain(back.iter())
            .filter(|cp| include_zero || cp.step >= 1)
            .skip(skip)
        {
            if let Some(s) = cp.state.as_scalars()
                && s.len() >= 4
            {
                self.history
                    .push_back((s[3], Vector3::new(s[0], s[1], s[2])));
            }
        }
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        if let Some(p) = params.as_scalars()
            && p.len() >= 6
        {
            self.simulator.m0 = p[0];
            self.simulator.t1 = p[1];
            self.simulator.t2 = p[2];
            self.simulator.b_field = Vector3::new(p[3], p[4], p[5]);
        }
        self.advance_bloch_ticks(1, false);
    }

    fn running_substeps_per_tick(&self) -> usize {
        let time_scale = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .time_scale;
        if time_scale.is_finite() && time_scale > 0.0 {
            (time_scale.ceil().min(MAX_MRI_STEPS_PER_FRAME as f64) as usize).max(1)
        } else {
            1
        }
    }

    fn step_running_substep(&mut self) {
        let _ = self.sync_params_and_get_scale();
        self.advance_bloch_ticks(1, true);
    }
}
