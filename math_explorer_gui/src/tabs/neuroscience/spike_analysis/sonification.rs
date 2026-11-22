//! Zero-allocation Hodgkin-Huxley patch-clamp sonification helpers for [`crate::tabs::neuroscience::spike_analysis::SpikeAnalysisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SpikeAnalysisTool;
use crate::audio::{AcousticSettings, SonificationFrame};
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::hash::{Hash, Hasher};

/// Maps Hodgkin-Huxley membrane voltage `v` and `(m, h, n)` ion-channel gating variables
/// into a zero-allocation patch-clamp [`SonificationFrame`].
#[inline]
#[must_use]
pub fn hodgkin_huxley_sonification_frame(
    v: f64,
    m: f64,
    h: f64,
    n: f64,
    spike_triggered: bool,
) -> SonificationFrame {
    if !v.is_finite() || !m.is_finite() || !h.is_finite() || !n.is_finite() {
        return SonificationFrame::silent();
    }
    let v_norm = ((v + 80.0) / 130.0).clamp(0.0, 1.0) as f32;
    let m_c = m.clamp(0.0, 1.0) as f32;
    let h_c = h.clamp(0.0, 1.0) as f32;
    let n_c = n.clamp(0.0, 1.0) as f32;
    let na_open = (m_c * m_c * m_c * h_c).clamp(0.0, 1.0);
    let k_open = (n_c * n_c * n_c * n_c).clamp(0.0, 1.0);

    let carrier_hz = 110.0 + v_norm * 520.0;
    let amplitude = (0.20 + 0.62 * na_open + 0.22 * v_norm).clamp(0.0, 0.95);
    let pan = ((na_open - k_open) * 0.85).clamp(-1.0, 1.0);
    let filter_cutoff_hz =
        (280.0 + 4_200.0 * m_c * (0.35 + 0.65 * h_c) + 900.0 * k_open).clamp(120.0, 7_500.0);
    let filter_q = (1.2 + 7.5 * (1.0 - h_c) + 3.0 * n_c).clamp(0.8, 14.0);
    let click_impulse = if spike_triggered {
        0.92
    } else if v > 10.0 {
        0.45
    } else {
        0.0
    };

    SonificationFrame {
        active: true,
        carrier_hz,
        amplitude,
        pan,
        fm_modulator_hz: carrier_hz * (1.0 + 0.5 * k_open),
        fm_index_hz: 12.0 + 95.0 * na_open,
        filter_cutoff_hz,
        filter_q,
        filter_bandpass: true,
        click_impulse,
        ..SonificationFrame::silent()
    }
}

impl SpikeAnalysisTool {
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
        let recent_window = (self.simulation_speed.max(1) as u32).max(15);
        let spike_triggered = self.steps_since_last_spike <= recent_window;
        hodgkin_huxley_sonification_frame(
            self.neuron.v(),
            self.neuron.m(),
            self.neuron.h(),
            self.neuron.n(),
            spike_triggered,
        )
    }

    pub(super) fn render_wav_frames(&self, frame_count: usize) -> Vec<SonificationFrame> {
        let count = frame_count.max(1);
        let mut out = Vec::with_capacity(count);
        let mut neuron = HodgkinHuxleyNeuron::new(self.neuron.v());
        let mut prev_v = neuron.v();
        let substeps = self.simulation_speed.clamp(1, 100).max(25);
        let current = if self.input_current.is_finite() {
            self.input_current
        } else {
            10.0
        };
        for _ in 0..count {
            let mut spiked = false;
            for _ in 0..substeps {
                neuron.update(0.01, current);
                let v = neuron.v();
                if v >= self.spike_threshold && prev_v < self.spike_threshold {
                    spiked = true;
                }
                prev_v = v;
            }
            out.push(hodgkin_huxley_sonification_frame(
                neuron.v(),
                neuron.m(),
                neuron.h(),
                neuron.n(),
                spiked,
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
