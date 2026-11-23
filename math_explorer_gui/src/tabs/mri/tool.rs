#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::pulse::{sync_mri_callouts, MriPulseMode, MriPulseWorkspace, RfPulseConfig};
use super::runner::{MriRunner, MriSharedParams, MAX_MRI_TIME_SCALE, MIN_MRI_TIME_SCALE};
use super::theory::MriBlochTheory;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::recovery::DiagnosticReport;
use crate::framework::{CalloutManager, ChallengeEvaluation, ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::physics::mri::bloch::{rotating_frame_effective_field, BlochSimulator};
use math_explorer::physics::mri::proton;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
use nalgebra::Vector3;
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Default longitudinal relaxation time constant `T1` in seconds.
pub const DEFAULT_MRI_T1: f64 = 1.0;
/// Default transverse relaxation time constant `T2` in seconds.
pub const DEFAULT_MRI_T2: f64 = 0.1;
/// Default static longitudinal field `Bz` in Tesla.
pub const DEFAULT_MRI_BZ: f64 = 1.0;
/// Default simulation speed multiplier.
pub const DEFAULT_MRI_TIME_SCALE: f64 = 1.0;

/// Validated state schema for [`MriTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MriToolState {
    /// Longitudinal relaxation time `T1` (`0.1..=5.0`).
    pub t1: f64,
    /// Transverse relaxation time `T2` (`0.01..=2.0`).
    pub t2: f64,
    /// Transverse magnetic field `Bx` (`-50.0..=50.0`).
    pub bx: f64,
    /// Transverse magnetic field `By` (`-50.0..=50.0`).
    pub by: f64,
    /// Main longitudinal magnetic field `Bz` (`-50.0..=50.0`).
    pub bz: f64,
    /// Simulation speed multiplier (`1.0..=100.0`).
    pub time_scale: f64,
    /// Optional worker execution state (`Ready` when omitted unless `paused == Some(false)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<SimWorkerState>,
    /// Legacy boolean pause flag accepted on deserialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Optional rotating-frame RF pulse sequence parameters (omitted in default forward mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pulse: Option<RfPulseConfig>,
}

impl MriToolState {
    /// Validates finiteness and physical domain bounds for all 6 MRI parameters.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("t1", self.t1, 0.1, 5.0),
            ("t2", self.t2, 0.01, 2.0),
            ("bx", self.bx, -50.0, 50.0),
            ("by", self.by, -50.0, 50.0),
            ("bz", self.bz, -50.0, 50.0),
            (
                "time_scale",
                self.time_scale,
                MIN_MRI_TIME_SCALE,
                MAX_MRI_TIME_SCALE,
            ),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "MRI parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if let Some(ref pulse) = self.pulse {
            pulse.validate()?;
        }
        Ok(())
    }

    pub(super) fn resolve_target_state(&self) -> SimWorkerState {
        self.state.unwrap_or_else(|| {
            if self.paused.unwrap_or(true) {
                SimWorkerState::Ready
            } else {
                SimWorkerState::Running
            }
        })
    }
}

/// Interactive tool for MRI Bloch precession and T1/T2 relaxation dynamics.
pub struct MriTool {
    pub(crate) simulator: BlochSimulator,
    pub(crate) solver: RungeKutta4<Vector3<f64>>,
    /// Simulation speed multiplier (`1.0..=100.0`).
    pub time_scale: f64,
    /// Current simulated time in seconds.
    pub current_time: f64,
    pub(crate) history: VecDeque<(f64, Vector3<f64>)>,
    pub(crate) max_history: usize,
    /// Cached plot series for `Mx(t)`.
    pub mx_points: Vec<PlotPoint>,
    /// Cached plot series for `My(t)`.
    pub my_points: Vec<PlotPoint>,
    /// Cached plot series for `Mz(t)`.
    pub mz_points: Vec<PlotPoint>,
    /// Shared worker parameter handle.
    pub params: Arc<std::sync::RwLock<MriSharedParams>>,
    /// Background Bloch worker controller.
    pub worker: BackgroundWorker<MriRunner>,
    /// Automated target & extremum plot callouts.
    pub callout_manager: CalloutManager,
    /// Rotating-frame RF pulse sequence settings.
    pub pulse_config: RfPulseConfig,
    /// Preallocated zero-allocation workspace for pulse sequences.
    pub pulse_workspace: MriPulseWorkspace,
    /// Active inverse challenge identifier, if any.
    pub active_challenge: Option<&'static str>,
    theory: MriBlochTheory,
}

impl Default for MriTool {
    fn default() -> Self {
        let initial_m = Vector3::new(1.0, 0.0, 0.0);
        let m0 = 1.0;
        let mut simulator = BlochSimulator::new(initial_m, m0);
        simulator.set_relaxation(DEFAULT_MRI_T1, DEFAULT_MRI_T2);
        simulator.set_b_field(Vector3::new(0.0, 0.0, DEFAULT_MRI_BZ));
        let solver = RungeKutta4::new(&initial_m);

        let params = Arc::new(std::sync::RwLock::new(MriSharedParams {
            t1: DEFAULT_MRI_T1,
            t2: DEFAULT_MRI_T2,
            b_field: Vector3::new(0.0, 0.0, DEFAULT_MRI_BZ),
            time_scale: DEFAULT_MRI_TIME_SCALE,
        }));

        let runner = MriRunner::new(Arc::clone(&params), initial_m, m0);
        let worker = BackgroundWorker::new(runner);
        let mut callout_manager = CalloutManager::new();
        let pulse_config = RfPulseConfig::default();
        sync_mri_callouts(
            &mut callout_manager,
            MriPulseMode::ContinuousField,
            m0,
            DEFAULT_MRI_T2,
            &pulse_config,
            0.0,
            initial_m,
        );

        Self {
            simulator,
            solver,
            time_scale: DEFAULT_MRI_TIME_SCALE,
            current_time: 0.0,
            history: VecDeque::new(),
            max_history: math_commons::registry::MAX_ITERATIONS,
            mx_points: Vec::new(),
            my_points: Vec::new(),
            mz_points: Vec::new(),
            params,
            worker,
            callout_manager,
            pulse_config,
            pulse_workspace: MriPulseWorkspace::default(),
            active_challenge: None,
            theory: MriBlochTheory,
        }
    }
}

impl MriTool {
    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let (t1, t2, b_field) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.t1, p.t2, p.b_field)
        };
        self.simulator.t1 = t1;
        self.simulator.t2 = t2;
        self.simulator.b_field = b_field;
        let mut runner_simulator =
            BlochSimulator::new(self.simulator.magnetization, self.simulator.m0);
        runner_simulator.set_relaxation(t1, t2);
        runner_simulator.set_b_field(b_field);
        let mut history = self.history.clone();
        if history.is_empty() {
            history.push_back((self.current_time, self.simulator.magnetization));
        }
        let runner = MriRunner {
            simulator: runner_simulator,
            solver: RungeKutta4::new(&self.simulator.magnetization),
            params: Arc::clone(&self.params),
            current_time: self.current_time,
            history,
            max_history: self.max_history,
        };
        self.worker = BackgroundWorker::new(runner);
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    /// Repopulates cached plot point buffers from `history` while preserving allocated capacity.
    pub fn update_plot_buffers(&mut self) {
        self.mx_points.clear();
        self.my_points.clear();
        self.mz_points.clear();
        self.mx_points.reserve(self.history.len());
        self.my_points.reserve(self.history.len());
        self.mz_points.reserve(self.history.len());
        for (t, m) in &self.history {
            self.mx_points.push(PlotPoint::new(*t, m.x));
            self.my_points.push(PlotPoint::new(*t, m.y));
            self.mz_points.push(PlotPoint::new(*t, m.z));
        }
        sync_mri_callouts(
            &mut self.callout_manager,
            self.pulse_config.mode,
            self.simulator.m0,
            self.simulator.t2,
            &self.pulse_config,
            self.current_time,
            self.simulator.magnetization,
        );
    }

    /// Restarts dynamic magnetization to `(M0, 0, 0)` and clears history while preserving finite user parameters.
    pub fn reset(&mut self) {
        if !self.simulator.t1.is_finite() || self.simulator.t1 <= 0.0 {
            self.simulator.t1 = DEFAULT_MRI_T1;
        }
        if !self.simulator.t2.is_finite() || self.simulator.t2 <= 0.0 {
            self.simulator.t2 = DEFAULT_MRI_T2;
        }
        if !self.simulator.b_field.x.is_finite() {
            self.simulator.b_field.x = 0.0;
        }
        if !self.simulator.b_field.y.is_finite() {
            self.simulator.b_field.y = 0.0;
        }
        if !self.simulator.b_field.z.is_finite() {
            self.simulator.b_field.z = DEFAULT_MRI_BZ;
        }
        self.time_scale = super::runner::sanitize_mri_time_scale(self.time_scale);
        if self.pulse_config.validate().is_err() {
            self.pulse_config = RfPulseConfig::default();
        }
        self.enforce_challenge_locks();
        self.sync_shared_params();
        self.reset_dynamic_trajectory(false);
    }

    /// Restores factory default `t1`, `t2`, `b_field`, and `time_scale` and resets magnetization to `(M0, 0, 0)`.
    #[cfg(test)]
    pub fn reset_parameters(&mut self) {
        self.simulator.t1 = DEFAULT_MRI_T1;
        self.simulator.t2 = DEFAULT_MRI_T2;
        self.simulator.b_field = Vector3::new(0.0, 0.0, DEFAULT_MRI_BZ);
        self.time_scale = DEFAULT_MRI_TIME_SCALE;
        if self.active_challenge.is_none() {
            self.pulse_config = RfPulseConfig::default();
        }
        self.enforce_challenge_locks();
        self.sync_shared_params();
        self.reset_dynamic_trajectory(false);
    }

    fn sync_shared_params(&self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.t1 = self.simulator.t1;
        p.t2 = self.simulator.t2;
        p.b_field = self.simulator.b_field;
        p.time_scale = self.time_scale;
    }

    pub(super) fn reset_dynamic_trajectory(&mut self, replace_worker: bool) {
        let initial_m = Vector3::new(self.simulator.m0, 0.0, 0.0);
        self.simulator.magnetization = initial_m;
        self.solver = RungeKutta4::new(&initial_m);
        self.current_time = 0.0;
        self.history.clear();
        self.mx_points.clear();
        self.my_points.clear();
        self.mz_points.clear();
        let params = Arc::clone(&self.params);
        let m0 = self.simulator.m0;
        if replace_worker {
            self.worker = BackgroundWorker::new(MriRunner::new(params, initial_m, m0));
        } else {
            self.worker
                .reset_or_respawn(|| MriRunner::new(params, initial_m, m0));
        }
        if !self.pulse_config.mode.is_continuous() {
            self.recompute_pulse_sequence();
        } else {
            sync_mri_callouts(
                &mut self.callout_manager,
                MriPulseMode::ContinuousField,
                m0,
                self.simulator.t2,
                &self.pulse_config,
                0.0,
                initial_m,
            );
        }
    }

    pub(super) fn apply_worker_state(&mut self, target_state: SimWorkerState) {
        if target_state == SimWorkerState::Running {
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else if self.worker.is_running() {
            let _ = self.worker.send_command(SimCommand::Pause);
        }
    }
}

impl InteractiveTool for MriTool {
    fn name(&self) -> &'static str {
        "MRI Bloch Relaxation"
    }

    fn domain(&self) -> &'static str {
        "mri"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.export_mri_datasets()
    }

    fn step_simulation(&mut self) {
        if !self.pulse_config.mode.is_continuous() {
            self.recompute_pulse_sequence();
            return;
        }
        if self.history.is_empty() {
            self.history
                .push_back((self.current_time, self.simulator.magnetization));
        }
        let dt = 0.01;
        let b_lab = self.simulator.b_field;
        let omega_rf = proton::larmor_frequency(b_lab.z);
        let b1_amp = b_lab.x.hypot(b_lab.y);
        let phase_rad = b_lab.y.atan2(b_lab.x);
        self.simulator.b_field =
            rotating_frame_effective_field(b_lab.z, omega_rf, b1_amp, phase_rad);
        TimeStepper::step_with(&mut self.simulator, &mut self.solver, dt);
        self.simulator.b_field = b_lab;
        self.current_time += dt;
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history
            .push_back((self.current_time, self.simulator.magnetization));
        self.update_plot_buffers();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.show_ui_inside(ui);
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.simulator.t1.to_bits().hash(&mut h);
        self.simulator.t2.to_bits().hash(&mut h);
        self.simulator.b_field.x.to_bits().hash(&mut h);
        self.simulator.b_field.y.to_bits().hash(&mut h);
        self.simulator.b_field.z.to_bits().hash(&mut h);
        self.time_scale.to_bits().hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.pulse_config.mode.hash(&mut h);
        self.pulse_config.b1_amp.to_bits().hash(&mut h);
        self.pulse_config.duration_tau.to_bits().hash(&mut h);
        self.pulse_config.phase_deg.to_bits().hash(&mut h);
        self.pulse_config.detuning_hz.to_bits().hash(&mut h);
        self.pulse_config.echo_time_te.to_bits().hash(&mut h);
        self.active_challenge.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_mri_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_mri_state(state)
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        self.set_mri_challenge(challenge_id)
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.eval_mri_challenge()
    }

    fn has_finite_parameters(&self) -> bool {
        let m = self.simulator.magnetization;
        self.simulator.t1.is_finite()
            && self.simulator.t2.is_finite()
            && self.simulator.b_field.x.is_finite()
            && self.simulator.b_field.y.is_finite()
            && self.simulator.b_field.z.is_finite()
            && self.time_scale.is_finite()
            && self.current_time.is_finite()
            && m.x.is_finite()
            && m.y.is_finite()
            && m.z.is_finite()
            && self.pulse_config.validate().is_ok()
            && self.worker.state() != SimWorkerState::Faulted
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        self.build_diagnostic_report()
    }

    fn auto_scale_step_size(&mut self) -> bool {
        self.time_scale =
            (super::runner::sanitize_mri_time_scale(self.time_scale) * 0.5).max(MIN_MRI_TIME_SCALE);
        self.reset();
        true
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_mri_parameters();
        true
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "MriTool",
        display_name: "MRI Bloch Relaxation",
        domain: "mri",
        tags: &[
            "mri",
            "bloch",
            "t1",
            "t2",
            "larmor",
            "magnetization",
            "nuclear induction",
        ],
        build: || Box::new(MriTool::default()),
    }
}
