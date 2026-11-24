//! Deterministic 16-bit PCM `.wav` audio export dispatch for [`crate::framework::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::export_action_button;
use super::{ExportCapabilities, ExportFormat, ExportStatusSeverity, SimulationFramework};
use crate::audio::{
    synthesize_wav_16bit, DEFAULT_WAV_DURATION_SECS, DEFAULT_WAV_FRAME_COUNT,
    DEFAULT_WAV_SAMPLE_RATE,
};
use eframe::egui;
use std::hash::{Hash, Hasher};

fn deterministic_tool_seed(tool_name: &str, state_json: Option<&str>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tool_name.hash(&mut hasher);
    if let Some(json) = state_json {
        json.hash(&mut hasher);
    }
    hasher.finish() ^ 0x534F_4E49_4659_3136
}

impl SimulationFramework {
    pub(super) fn render_audio_export_items(
        &mut self,
        ui: &mut egui::Ui,
        caps: ExportCapabilities,
        close_on_click: bool,
    ) {
        let wav_tip = if caps.audio {
            "Synthesize and export 16-bit PCM WAV audio clip"
        } else {
            caps.audio_disabled_reason
        };
        if export_action_button(ui, caps.audio, "Export WAV", wav_tip, close_on_click) {
            self.queue_export(ExportFormat::Wav);
        }
    }

    pub(super) fn dispatch_wav_export(&mut self) {
        let (tool_name, caps, settings, saved_state) = {
            let Some(tool) = self.active_tool.as_ref() else {
                return;
            };
            (
                tool.name().to_string(),
                tool.export_capabilities(),
                tool.acoustic_settings(),
                tool.save_state(),
            )
        };
        if !caps.audio {
            self.set_export_status(ExportStatusSeverity::Warning, caps.audio_disabled_reason);
            return;
        }
        if self.is_diverged {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Cannot export WAV audio while {tool_name} is in a diverged state."),
            );
            return;
        }

        let frames = if let Some(tool) = self.active_tool.as_mut() {
            let sampled = tool.sample_sonification_frames(DEFAULT_WAV_FRAME_COUNT);
            if let Some(ref state_str) = saved_state {
                let _ = tool.try_load_recovery_state(state_str);
            }
            sampled
        } else {
            Vec::new()
        };

        if frames.is_empty() {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("No sonification frames available to export from {tool_name}."),
            );
            return;
        }

        let seed = deterministic_tool_seed(&tool_name, saved_state.as_deref());
        let wav_bytes = synthesize_wav_16bit(
            &frames,
            settings,
            DEFAULT_WAV_SAMPLE_RATE,
            DEFAULT_WAV_DURATION_SECS,
            seed,
        );
        let filename = Self::build_export_filename(&tool_name, ExportFormat::Wav);
        self.complete_file_export(
            ExportFormat::Wav,
            &filename,
            &wav_bytes,
            None,
            ExportFormat::Wav.filter_name(),
            "WAV audio export failed",
        );
    }
}
