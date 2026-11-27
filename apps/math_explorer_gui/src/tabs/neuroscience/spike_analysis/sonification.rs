//! Zero-allocation Hodgkin-Huxley spike-train sonification and preset state helpers for
//! [`crate::tabs::neuroscience::spike_analysis::SpikeAnalysisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SpikeAnalysisTool;
use crate::audio::mappings::spike_analysis_frame;
use crate::audio::{AcousticSettings, SonificationFrame};
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::hash::{Hash, Hasher};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SpikeAnalysisState {
    input_current: f64,
    spike_threshold: f64,
    simulation_speed: usize,
    #[serde(default)]
    is_running: bool,
    #[serde(
        default,
        skip_serializing_if = "crate::audio::AcousticSettings::is_default"
    )]
    audio: AcousticSettings,
}

impl SpikeAnalysisTool {
    pub(super) fn serialize_state(&self) -> Option<String> {
        serde_json::to_string(&SpikeAnalysisState {
            input_current: self.input_current,
            spike_threshold: self.spike_threshold,
            simulation_speed: self.simulation_speed,
            is_running: self.is_running,
            audio: self.audio_settings,
        })
        .ok()
    }

    pub(super) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: SpikeAnalysisState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid spike analysis state: {e}"))?;
        parsed.audio.validate()?;
        if !parsed.input_current.is_finite()
            || !(0.0..=50.0).contains(&parsed.input_current)
            || !parsed.spike_threshold.is_finite()
            || !(-50.0..=20.0).contains(&parsed.spike_threshold)
            || !(1..=100).contains(&parsed.simulation_speed)
        {
            return Err("Spike analysis parameters out of bounds".into());
        }
        self.reset();
        self.input_current = parsed.input_current;
        self.spike_threshold = parsed.spike_threshold;
        self.simulation_speed = parsed.simulation_speed;
        self.is_running = parsed.is_running;
        self.audio_settings = parsed.audio;
        Ok(())
    }

    pub(super) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.input_current.to_bits().hash(&mut h);
        self.spike_threshold.to_bits().hash(&mut h);
        self.simulation_speed.hash(&mut h);
        self.is_running.hash(&mut h);
        self.audio_settings.gain.to_bits().hash(&mut h);
        self.audio_settings.pitch_scale.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    pub(super) fn current_sonification_frame(&self) -> SonificationFrame {
        if self.sonification.is_audible() {
            self.sonification
        } else {
            spike_analysis_frame(
                self.neuron.v(),
                self.last_voltage,
                self.spike_threshold,
                self.mean_firing_rate_hz(),
            )
        }
    }

    pub(super) fn render_wav_frames(&self, frame_count: usize) -> Vec<SonificationFrame> {
        let count = frame_count.max(1);
        let mut out = Vec::with_capacity(count);
        let mut neuron = HodgkinHuxleyNeuron::new(self.neuron.v());
        let mut prev_v = neuron.v();
        let mut sim_time = 0.0_f64;
        let mut last_spike: Option<f64> = None;
        let mut rate_hz = self.mean_firing_rate_hz();
        let substeps = self.simulation_speed.clamp(1, 100).max(25);
        let current = if self.input_current.is_finite() {
            self.input_current
        } else {
            10.0
        };
        for _ in 0..count {
            let frame_start_v = prev_v;
            let mut spiked = false;
            for _ in 0..substeps {
                neuron.update(0.01, current);
                sim_time += 0.01;
                let v = neuron.v();
                if v >= self.spike_threshold && prev_v < self.spike_threshold {
                    spiked = true;
                    if let Some(prev_t) = last_spike {
                        let isi = (sim_time - prev_t).max(0.1);
                        rate_hz = 1000.0 / isi;
                    }
                    last_spike = Some(sim_time);
                }
                prev_v = v;
            }
            let cur_v = neuron.v();
            let ref_prev = if spiked {
                self.spike_threshold - 5.0
            } else {
                frame_start_v
            };
            let ref_cur = if spiked {
                cur_v.max(self.spike_threshold + 5.0)
            } else {
                cur_v
            };
            out.push(spike_analysis_frame(
                ref_cur,
                ref_prev,
                self.spike_threshold,
                rate_hz,
            ));
        }
        out
    }

    pub(super) const fn get_audio_settings(&self) -> AcousticSettings {
        self.audio_settings
    }

    pub(super) fn update_audio_settings(&mut self, settings: AcousticSettings) {
        self.audio_settings = settings.sanitized();
    }
}
