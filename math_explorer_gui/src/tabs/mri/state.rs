//! State persistence and challenge lock helpers for [`super::tool::MriTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::pulse::{
    score_hahn_spin_echo, score_pi_inversion, MriPulseMode, CHALLENGE_BLOCH_HAHN_ECHO,
    CHALLENGE_BLOCH_PI_INVERSION, CHALLENGE_MRI_B0, CHALLENGE_MRI_T1, CHALLENGE_MRI_T2,
    CHALLENGE_MRI_TE, MRI_PULSE_SAMPLES,
};
use super::runner::MriSnapshot;
use super::tool::{MriTool, MriToolState};
use crate::async_sim::SimWorkerState;
use crate::framework::ChallengeEvaluation;
use nalgebra::Vector3;

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
        serde_json::to_string(&MriToolState {
            t1: self.simulator.t1,
            t2: self.simulator.t2,
            bx: self.simulator.b_field.x,
            by: self.simulator.b_field.y,
            bz: self.simulator.b_field.z,
            time_scale: self.time_scale,
            state: Some(self.worker.state()),
            paused: None,
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
        if self.pulse_config.mode.is_continuous() {
            self.apply_worker_state(target_state);
            self.latest_sonification = self.current_sonification_frame();
        }
        Ok(())
    }

    pub(super) fn export_mri_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
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

    /// Computes the stack-allocated [`crate::audio::SonificationFrame`] for the active mode/playhead.
    #[must_use]
    pub fn current_sonification_frame(&self) -> crate::audio::SonificationFrame {
        if self.worker.state() == SimWorkerState::Faulted {
            return crate::audio::SonificationFrame::silent();
        }
        match self.pulse_config.mode {
            MriPulseMode::ContinuousField => crate::audio::mappings::mri_bloch_frame(
                self.simulator.magnetization,
                self.simulator.m0,
                self.simulator.b_field,
                self.pulse_config.detuning_hz,
            ),
            MriPulseMode::InversionPulse => {
                let (t_s, m_sample, disp) = self.pulse_workspace.playhead_sample();
                crate::audio::mappings::mri_pulse_playhead_frame(
                    MriPulseMode::InversionPulse,
                    t_s,
                    m_sample,
                    disp,
                    self.simulator.m0,
                    self.simulator.b_field.z,
                    &self.pulse_config,
                )
            }
            MriPulseMode::HahnSpinEcho => {
                let (t_s, _, _) = self.pulse_workspace.playhead_sample();
                let mut frame = self.pulse_workspace.sample_sonification_frame(
                    MriPulseMode::HahnSpinEcho,
                    self.pulse_workspace.playhead_idx,
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
}

// [cite:graph_parameters_rust]
