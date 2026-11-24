//! State persistence and challenge lock helpers for [`super::tool::MriTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::pulse::{
    score_hahn_spin_echo, score_pi_inversion, sync_mri_callouts, MriPulseMode,
    CHALLENGE_BLOCH_HAHN_ECHO, CHALLENGE_BLOCH_PI_INVERSION, CHALLENGE_MRI_B0, CHALLENGE_MRI_T1,
    CHALLENGE_MRI_T2, CHALLENGE_MRI_TE, MRI_PULSE_SAMPLES,
};
use super::runner::{MriRunner, MriSnapshot, MAX_MRI_TIME_SCALE, MIN_MRI_TIME_SCALE};
use super::tool::{MriTool, MriToolState};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::ChallengeEvaluation;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

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
        self.audio.validate()?;
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

impl MriTool {
    /// Evaluates the active rotating-frame RF pulse sequence and populates plot buffers & callouts.
    pub fn recompute_pulse_sequence(&mut self) {
        if self.pulse_config.mode.is_continuous() {
            return;
        }
        self.enforce_challenge_locks();
        let b0 = self.simulator.b_field.z;
        let t1 = self.simulator.t1;
        let t2 = self.simulator.t2;
        let m0 = self.simulator.m0;
        let final_m = match self.pulse_config.mode {
            MriPulseMode::InversionPulse => {
                self.pulse_workspace
                    .evaluate_inversion(b0, t1, t2, m0, &self.pulse_config)
            }
            MriPulseMode::HahnSpinEcho => {
                self.pulse_workspace
                    .evaluate_hahn_echo(b0, t1, t2, m0, &self.pulse_config)
            }
            MriPulseMode::ContinuousField => return,
        };
        self.history.clear();
        for idx in 0..MRI_PULSE_SAMPLES {
            self.history.push_back((
                self.pulse_workspace.times[idx],
                self.pulse_workspace.trajectory[idx],
            ));
        }
        self.current_time = self.pulse_workspace.times[MRI_PULSE_SAMPLES - 1];
        self.simulator.magnetization = final_m;
        self.pulse_workspace.playhead_idx = MRI_PULSE_SAMPLES - 1;
        self.update_plot_buffers();
        self.latest_sonification = self.current_sonification_frame();
    }

    pub(crate) fn enforce_challenge_locks(&mut self) {
        if let Some(cid) = self.active_challenge {
            self.simulator.t1 = CHALLENGE_MRI_T1;
            self.simulator.t2 = CHALLENGE_MRI_T2;
            self.simulator.b_field = Vector3::new(0.0, 0.0, CHALLENGE_MRI_B0);
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.t1 = CHALLENGE_MRI_T1;
            p.t2 = CHALLENGE_MRI_T2;
            p.b_field = self.simulator.b_field;
            if cid == CHALLENGE_BLOCH_PI_INVERSION {
                self.pulse_config.mode = MriPulseMode::InversionPulse;
            } else if cid == CHALLENGE_BLOCH_HAHN_ECHO {
                self.pulse_config.mode = MriPulseMode::HahnSpinEcho;
                self.pulse_config.echo_time_te = CHALLENGE_MRI_TE;
            }
        }
    }

    pub(super) fn serialize_mri_state(&self) -> Option<String> {
        let pulse = (!self.pulse_config.mode.is_continuous()).then_some(self.pulse_config);
        let fork = self
            .twin
            .as_ref()
            .map(|t| t.to_descriptor(self.current_step_index()));
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
            fork,
            pulse,
        })
        .ok()
    }

    pub(super) fn deserialize_mri_state(&mut self, state: &str) -> Result<(), String> {
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
        self.pulse_config = parsed.pulse.unwrap_or_default();
        self.enforce_challenge_locks();
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.t1 = self.simulator.t1;
            p.t2 = self.simulator.t2;
            p.b_field = self.simulator.b_field;
            p.time_scale = self.time_scale;
        }
        self.reset_dynamic_trajectory(true);
        if let Some(desc) = parsed.fork {
            self.apply_mri_fork_descriptor(desc)?;
        }
        if self.pulse_config.mode.is_continuous() {
            self.apply_worker_state(target_state);
            self.latest_sonification = self.current_sonification_frame();
        }
        Ok(())
    }

    /// Restores factory default `t1`, `t2`, `b_field`, and `time_scale` and resets magnetization to `(M0, 0, 0)`.
    #[cfg(test)]
    pub fn reset_parameters(&mut self) {
        use super::pulse::RfPulseConfig;
        use super::tool::{DEFAULT_MRI_BZ, DEFAULT_MRI_T1, DEFAULT_MRI_T2, DEFAULT_MRI_TIME_SCALE};
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

    pub(super) fn reset_dynamic_trajectory(&mut self, replace_worker: bool) {
        let initial_m = Vector3::new(self.simulator.m0, 0.0, 0.0);
        self.simulator.magnetization = initial_m;
        self.solver = RungeKutta4::new(&initial_m);
        self.current_time = 0.0;
        self.history.clear();
        self.mx_points.clear();
        self.my_points.clear();
        self.mz_points.clear();
        self.twin = None;
        self.branch_engine.clear();
        self.pulse_workspace.playhead_playing = false;
        self.pulse_workspace.playhead_idx = MRI_PULSE_SAMPLES - 1;
        self.fault_notified = false;
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
        self.latest_sonification = self.current_sonification_frame();
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

    pub(super) fn set_mri_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                Ok(())
            }
            Some(cid) if cid.eq_ignore_ascii_case(CHALLENGE_BLOCH_PI_INVERSION) => {
                self.active_challenge = Some(CHALLENGE_BLOCH_PI_INVERSION);
                self.pulse_config.mode = MriPulseMode::InversionPulse;
                self.recompute_pulse_sequence();
                Ok(())
            }
            Some(cid) if cid.eq_ignore_ascii_case(CHALLENGE_BLOCH_HAHN_ECHO) => {
                self.active_challenge = Some(CHALLENGE_BLOCH_HAHN_ECHO);
                self.pulse_config.mode = MriPulseMode::HahnSpinEcho;
                self.recompute_pulse_sequence();
                Ok(())
            }
            Some(other) => Err(format!("Unsupported MRI challenge '{other}'")),
        }
    }

    pub(super) fn eval_mri_challenge(&self) -> Option<ChallengeEvaluation> {
        match self.active_challenge? {
            CHALLENGE_BLOCH_PI_INVERSION => Some(score_pi_inversion(
                self.simulator.magnetization,
                self.simulator.m0,
            )),
            CHALLENGE_BLOCH_HAHN_ECHO => Some(score_hahn_spin_echo(
                self.simulator.magnetization,
                self.simulator.m0,
                self.pulse_config.echo_time_te,
                self.simulator.t2,
            )),
            _ => None,
        }
    }

    pub(super) fn pulse_sonification_frame_at(
        &self,
        sample_idx: usize,
    ) -> crate::audio::SonificationFrame {
        let idx = sample_idx.min(super::pulse::MRI_PULSE_SAMPLES - 1);
        let t_s = self.pulse_workspace.times[idx];
        match self.pulse_config.mode {
            MriPulseMode::ContinuousField => crate::audio::mappings::mri_bloch_frame(
                self.simulator.magnetization,
                self.simulator.m0,
                self.simulator.b_field,
                self.pulse_config.detuning_hz,
            ),
            MriPulseMode::InversionPulse => crate::audio::mappings::mri_pulse_playhead_frame(
                MriPulseMode::InversionPulse,
                t_s,
                self.pulse_workspace.trajectory[idx],
                self.pulse_workspace.dispersion[idx],
                self.simulator.m0,
                self.simulator.b_field.z,
                &self.pulse_config,
            ),
            MriPulseMode::HahnSpinEcho => {
                let mut frame = self.pulse_workspace.sample_sonification_frame(
                    MriPulseMode::HahnSpinEcho,
                    idx,
                    self.simulator.b_field,
                    self.simulator.m0,
                );
                let detune_shift =
                    0.25 * (self.pulse_config.detuning_hz.clamp(-5000.0, 5000.0) as f32);
                if detune_shift.abs() > 1e-3 {
                    for v in &mut frame.voices[..frame.active_voices as usize] {
                        v.freq_hz = crate::audio::frame::clamp_freq_hz(v.freq_hz + detune_shift);
                    }
                }
                if crate::audio::mappings::is_rf_pulse_window(
                    MriPulseMode::HahnSpinEcho,
                    t_s,
                    &self.pulse_config,
                ) {
                    frame.voices[0].transient_impulse = 0.85;
                }
                frame.sanitize()
            }
        }
    }

    /// Computes the stack-allocated [`crate::audio::SonificationFrame`] for the active mode/playhead.
    #[must_use]
    pub fn current_sonification_frame(&self) -> crate::audio::SonificationFrame {
        if self.worker.state() == SimWorkerState::Faulted {
            return crate::audio::SonificationFrame::silent();
        }
        self.pulse_sonification_frame_at(self.pulse_workspace.playhead_idx)
    }

    pub(super) fn collect_mri_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let map_xyz = |hist: &VecDeque<(f64, Vector3<f64>)>, labels: [&str; 3]| {
            vec![
                (
                    labels[0].to_string(),
                    hist.iter().map(|(t, m)| [*t, m.x]).collect(),
                ),
                (
                    labels[1].to_string(),
                    hist.iter().map(|(t, m)| [*t, m.y]).collect(),
                ),
                (
                    labels[2].to_string(),
                    hist.iter().map(|(t, m)| [*t, m.z]).collect(),
                ),
            ]
        };
        let std_labels = ["Mx (Transverse)", "My (Transverse)", "Mz (Longitudinal)"];
        if let Some(ref twin) = self.twin {
            if !self.history.is_empty() {
                let mut out = map_xyz(
                    &self.history,
                    ["Mx (Branch B)", "My (Branch B)", "Mz (Branch B)"],
                );
                out.extend(map_xyz(
                    &twin.history_a,
                    ["Mx (Branch A)", "My (Branch A)", "Mz (Branch A)"],
                ));
                return out;
            }
        }
        if !self.history.is_empty() {
            return map_xyz(&self.history, std_labels);
        }
        if let Some(data) = self
            .worker
            .latest_snapshot()
            .and_then(|s| s.structured_data.as_ref())
            .and_then(|d| d.downcast_ref::<MriSnapshot>())
            .filter(|d| !d.history.is_empty())
        {
            return map_xyz(&data.history, std_labels);
        }
        let m = self.simulator.magnetization;
        vec![
            (std_labels[0].to_string(), vec![[self.current_time, m.x]]),
            (std_labels[1].to_string(), vec![[self.current_time, m.y]]),
            (std_labels[2].to_string(), vec![[self.current_time, m.z]]),
        ]
    }
}

// [cite:graph_parameters_rust]
