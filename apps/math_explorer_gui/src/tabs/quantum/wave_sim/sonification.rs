//! Periodic-wave modal sonification helpers and WAV frame sampling for [`crate::tabs::quantum::wave_sim::WaveSimulatorTool`].
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{WaveRunner, WaveSimulatorTool};
use std::sync::{atomic::AtomicUsize, Arc};

impl WaveRunner {
    pub(crate) fn refresh_sonification(&mut self, dt: f64) {
        let modes = self.psi.project_spatial_modes::<8>();
        self.sonification = crate::audio::mappings::quantum_wavepacket_frame(
            &modes,
            &mut self.prev_mode_phases,
            dt.max(1.0e-4),
        );
    }

    /// Current modal sonification frame emitted by the quantum wavepacket.
    #[must_use]
    pub(crate) const fn sonification_frame(&self) -> crate::audio::SonificationFrame {
        self.sonification
    }

    pub(crate) const fn active_sonification_frame(&self) -> crate::audio::SonificationFrame {
        if self.diverged {
            crate::audio::SonificationFrame::silent()
        } else {
            self.sonification
        }
    }
}

impl WaveSimulatorTool {
    pub(super) fn sample_wave_sonification_frames(
        &self,
        num_frames: usize,
    ) -> Vec<crate::audio::SonificationFrame> {
        let count = num_frames.max(1);
        let mut runner = WaveRunner::new_with_counter(
            Arc::clone(&self.shared_config),
            Arc::new(AtomicUsize::new(0)),
        );
        let sub_steps = self.steps_per_frame.clamp(1, 20);
        let dt = self.dt;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            for _ in 0..sub_steps {
                runner.step_with_dt(dt);
            }
            out.push(runner.sonification_frame().sanitize());
        }
        out
    }
}

#[cfg(test)]
mod sonification_tests {
    use super::super::{WavePotentialPreset, WaveSimulatorTool};
    use crate::framework::InteractiveTool;

    #[test]
    fn test_wave_simulator_sonification_and_audio_roundtrip() {
        let mut tool = WaveSimulatorTool::new();
        tool.preset = WavePotentialPreset::Custom;
        tool.math_input.set_expression("v0 * cos(x)^2");
        tool.last_valid_expr = tool.math_input.expr().unwrap().clone();
        tool.set_acoustic_settings(crate::audio::AcousticSettings {
            gain: 0.68,
            pitch_scale: 1.35,
        });
        tool.sync_config_and_reset(None);
        tool.step_simulation();

        let live_frame = tool.sonification_frame().expect("sonification frame");
        assert!(live_frame.is_audible());
        assert!(live_frame.custom_harmonics.is_some());

        let saved = tool.save_state().expect("save wave simulator state");
        assert!(saved.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);
        let mut restored = WaveSimulatorTool::new();
        restored
            .try_load_state(&saved)
            .expect("load wave simulator state");
        assert_eq!(restored.preset, WavePotentialPreset::Custom);
        assert!((restored.acoustic_settings().gain - 0.68).abs() < 1e-5);
        assert!((restored.acoustic_settings().pitch_scale - 1.35).abs() < 1e-5);
    }
}
