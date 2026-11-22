#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::{
    MriRunner, MriSharedParams, MriSnapshot, MAX_MRI_TIME_SCALE, MIN_MRI_TIME_SCALE,
};
use super::theory::MriBlochTheory;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::physics::mri::bloch::BlochSimulator;
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
    /// Persisted non-mute acoustic synthesis settings.
    #[serde(
        default,
        skip_serializing_if = "crate::audio::AcousticSettings::is_default"
    )]
    pub audio: crate::audio::AcousticSettings,
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
        self.audio.validate()
    }

    fn resolve_target_state(&self) -> SimWorkerState {
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
    /// Persisted non-mute acoustic synthesis settings.
    pub audio_settings: crate::audio::AcousticSettings,
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
            audio_settings: crate::audio::AcousticSettings::default(),
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
    }

    /// Restores factory default `t1`, `t2`, `b_field`, and `time_scale` and resets magnetization to `(M0, 0, 0)`.
    pub fn reset(&mut self) {
        let default_b = Vector3::new(0.0, 0.0, DEFAULT_MRI_BZ);
        self.simulator.t1 = DEFAULT_MRI_T1;
        self.simulator.t2 = DEFAULT_MRI_T2;
        self.simulator.b_field = default_b;
        self.time_scale = DEFAULT_MRI_TIME_SCALE;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.t1 = DEFAULT_MRI_T1;
            p.t2 = DEFAULT_MRI_T2;
            p.b_field = default_b;
            p.time_scale = DEFAULT_MRI_TIME_SCALE;
        }
        let initial_m = Vector3::new(self.simulator.m0, 0.0, 0.0);
        self.simulator.magnetization = initial_m;
        self.current_time = 0.0;
        self.history.clear();
        self.mx_points.clear();
        self.my_points.clear();
        self.mz_points.clear();
        let params = Arc::clone(&self.params);
        let m0 = self.simulator.m0;
        self.worker
            .reset_or_respawn(|| MriRunner::new(params, initial_m, m0));
    }

    fn apply_worker_state(&mut self, target_state: SimWorkerState) {
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
        ExportCapabilities::plot().with_audio()
    }

    fn sonification_frame(&self) -> Option<crate::audio::SonificationFrame> {
        Some(super::sonification::bloch_fid_sonification_frame(
            &self.simulator,
            self.worker.state() == SimWorkerState::Faulted,
        ))
    }

    fn acoustic_settings(&self) -> crate::audio::AcousticSettings {
        self.audio_settings
    }

    fn set_acoustic_settings(&mut self, settings: crate::audio::AcousticSettings) {
        self.audio_settings = settings.sanitized();
    }

    fn sample_sonification_frames(
        &mut self,
        frame_count: usize,
    ) -> Vec<crate::audio::SonificationFrame> {
        self.render_wav_frames(frame_count)
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if !self.history.is_empty() {
            let mx: Vec<[f64; 2]> = self.history.iter().map(|(t, m)| [*t, m.x]).collect();
            let my: Vec<[f64; 2]> = self.history.iter().map(|(t, m)| [*t, m.y]).collect();
            let mz: Vec<[f64; 2]> = self.history.iter().map(|(t, m)| [*t, m.z]).collect();
            return vec![
                ("Mx (Transverse)".to_string(), mx),
                ("My (Transverse)".to_string(), my),
                ("Mz (Longitudinal)".to_string(), mz),
            ];
        }
        if let Some(snap) = self.worker.latest_snapshot() {
            if let Some(data) = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MriSnapshot>())
            {
                if !data.history.is_empty() {
                    let mx = data.history.iter().map(|(t, m)| [*t, m.x]).collect();
                    let my = data.history.iter().map(|(t, m)| [*t, m.y]).collect();
                    let mz = data.history.iter().map(|(t, m)| [*t, m.z]).collect();
                    return vec![
                        ("Mx (Transverse)".to_string(), mx),
                        ("My (Transverse)".to_string(), my),
                        ("Mz (Longitudinal)".to_string(), mz),
                    ];
                }
            }
        }
        let m = self.simulator.magnetization;
        vec![
            (
                "Mx (Transverse)".to_string(),
                vec![[self.current_time, m.x]],
            ),
            (
                "My (Transverse)".to_string(),
                vec![[self.current_time, m.y]],
            ),
            (
                "Mz (Longitudinal)".to_string(),
                vec![[self.current_time, m.z]],
            ),
        ]
    }

    fn step_simulation(&mut self) {
        if self.history.is_empty() {
            self.history
                .push_back((self.current_time, self.simulator.magnetization));
        }
        let dt = 0.01;
        TimeStepper::step_with(&mut self.simulator, &mut self.solver, dt);
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
        self.audio_settings.gain.to_bits().hash(&mut h);
        self.audio_settings.pitch_scale.to_bits().hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&MriToolState {
            t1: self.simulator.t1,
            t2: self.simulator.t2,
            bx: self.simulator.b_field.x,
            by: self.simulator.b_field.y,
            bz: self.simulator.b_field.z,
            time_scale: self.time_scale,
            state: Some(self.worker.state()),
            paused: None,
            audio: self.audio_settings,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: MriToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid MRI state: {e}"))?;
        parsed.validate()?;
        let target_state = parsed.resolve_target_state();
        let b_field = Vector3::new(parsed.bx, parsed.by, parsed.bz);
        self.simulator.t1 = parsed.t1;
        self.simulator.t2 = parsed.t2;
        self.simulator.b_field = b_field;
        self.time_scale = parsed.time_scale;
        self.audio_settings = parsed.audio;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.t1 = parsed.t1;
            p.t2 = parsed.t2;
            p.b_field = b_field;
            p.time_scale = parsed.time_scale;
        }
        self.apply_worker_state(target_state);
        Ok(())
    }

    fn has_finite_parameters(&self) -> bool {
        self.simulator.t1.is_finite()
            && self.simulator.t2.is_finite()
            && self.simulator.b_field.x.is_finite()
            && self.simulator.b_field.y.is_finite()
            && self.simulator.b_field.z.is_finite()
            && self.time_scale.is_finite()
            && self.worker.state() != SimWorkerState::Faulted
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
