//! Deferred end-of-frame `.wav`, `.mid`, `.scl`, and `.kbm` acoustic & microtonal export dispatch.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::export_action_button;
use super::{
    ExportCapabilities, ExportFormat, ExportStatusSeverity, ExportTarget, SimulationFramework,
};
use crate::audio::{
    synthesize_wav_16bit, DEFAULT_WAV_DURATION_SECS, DEFAULT_WAV_FRAME_COUNT,
    DEFAULT_WAV_SAMPLE_RATE, WAV_HEADER_BYTES,
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
    /// Renders `.wav`, `.mid`, `.scl`, and `.kbm` toolbar/menu buttons for simulation tools.
    pub(super) fn render_audio_export_items(
        &mut self,
        ui: &mut egui::Ui,
        caps: ExportCapabilities,
        close_on_click: bool,
    ) {
        let wav_enabled = caps.audio || caps.wav;
        let wav_tip = if wav_enabled {
            "Synthesize and export stereo 16-bit PCM WAV audio or impulse response"
        } else if !caps.audio_disabled_reason.is_empty() {
            caps.audio_disabled_reason
        } else {
            caps.wav_disabled_reason
        };
        if export_action_button(ui, wav_enabled, "Export WAV", wav_tip, close_on_click) {
            self.queue_export(ExportFormat::Wav);
        }
        if caps.midi
            && export_action_button(
                ui,
                true,
                "Export MIDI (.mid)",
                "Download SMF Format 1 polyphonic MIDI score with 14-bit microtonal pitch bends",
                close_on_click,
            )
        {
            self.queue_export(ExportFormat::Midi);
        }
        if caps.scala {
            if export_action_button(
                ui,
                true,
                "Export Scala (.scl)",
                "Download Huygens-Fokker Scala microtonal scale table (.scl) for Surge XT / Vital",
                close_on_click,
            ) {
                self.queue_export(ExportFormat::ScalaScl);
            }
            if export_action_button(
                ui,
                true,
                "Export Keymap (.kbm)",
                "Download Huygens-Fokker Scala keyboard root mapping (.kbm) for Surge XT / Vital",
                close_on_click,
            ) {
                self.queue_export(ExportFormat::ScalaKbm);
            }
        }
    }

    /// Dispatches a queued [`ExportFormat::Wav`], [`ExportFormat::Midi`], [`ExportFormat::ScalaScl`],
    /// or [`ExportFormat::ScalaKbm`] request at end-of-frame.
    pub(super) fn dispatch_audio_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        ctx: Option<&egui::Context>,
    ) {
        match format {
            ExportFormat::Wav => self.dispatch_wav_request(target),
            ExportFormat::Midi => self.dispatch_midi_request(target),
            ExportFormat::ScalaScl | ExportFormat::ScalaKbm => {
                self.dispatch_scala_request(format, target, ctx);
            }
            _ => {}
        }
    }

    fn dispatch_wav_request(&mut self, target: ExportTarget) {
        if target == ExportTarget::Clipboard {
            self.set_export_status(
                ExportStatusSeverity::Warning,
                "Clipboard export is unavailable for binary WAV audio; save .wav to disk.",
            );
            return;
        }
        let (tool_name, caps) = {
            let Some(tool) = self.active_tool.as_ref() else {
                return;
            };
            (tool.name().to_string(), tool.export_capabilities())
        };
        if !caps.wav && !caps.audio {
            let reason = if !caps.audio_disabled_reason.is_empty() {
                caps.audio_disabled_reason
            } else {
                caps.wav_disabled_reason
            };
            self.set_export_status(ExportStatusSeverity::Warning, reason);
            return;
        }
        if self.is_diverged {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Cannot export WAV audio while {tool_name} is in a diverged state."),
            );
            return;
        }
        let cfg = crate::audio::current_offline_config();
        let custom_res = self
            .active_tool
            .as_ref()
            .map(|t| t.export_wav_audio(cfg.sample_rate, cfg.duration_secs));
        match custom_res {
            Some(Ok(bytes)) if bytes.len() > WAV_HEADER_BYTES => {
                self.finish_wav_file(&tool_name, &bytes);
            }
            Some(Ok(_)) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("WAV export failed: {tool_name} produced an empty PCM payload."),
            ),
            Some(Err(err)) if err.starts_with("WAV audio export is not supported for ") => {
                self.dispatch_sampled_wav_fallback(&tool_name);
            }
            Some(Err(err)) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("WAV export failed: {err}"),
            ),
            None => {}
        }
    }

    fn dispatch_sampled_wav_fallback(&mut self, tool_name: &str) {
        let (settings, saved_state) = {
            let Some(tool) = self.active_tool.as_ref() else {
                return;
            };
            (tool.acoustic_settings(), tool.save_state())
        };
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
        let seed = deterministic_tool_seed(tool_name, saved_state.as_deref());
        let wav_bytes = synthesize_wav_16bit(
            &frames,
            settings,
            DEFAULT_WAV_SAMPLE_RATE,
            DEFAULT_WAV_DURATION_SECS,
            seed,
        );
        self.finish_wav_file(tool_name, &wav_bytes);
    }

    fn finish_wav_file(&mut self, tool_name: &str, wav_bytes: &[u8]) {
        let filename = Self::build_export_filename(tool_name, ExportFormat::Wav);
        self.complete_file_export(
            ExportFormat::Wav,
            &filename,
            wav_bytes,
            None,
            ExportFormat::Wav.filter_name(),
            "WAV export failed",
        );
    }

    fn dispatch_midi_request(&mut self, target: ExportTarget) {
        if target == ExportTarget::Clipboard {
            self.set_export_status(
                ExportStatusSeverity::Warning,
                "Clipboard export is unavailable for binary MIDI files; save .mid to disk.",
            );
            return;
        }
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.midi {
            self.set_export_status(ExportStatusSeverity::Warning, caps.midi_disabled_reason);
            return;
        }
        let cfg = crate::audio::current_offline_config();
        let tool_name = tool.name().to_string();
        match tool.export_midi_score(cfg.duration_secs) {
            Ok(bytes) if bytes.starts_with(b"MThd") => {
                let filename = Self::build_export_filename(&tool_name, ExportFormat::Midi);
                self.complete_file_export(
                    ExportFormat::Midi,
                    &filename,
                    &bytes,
                    None,
                    ExportFormat::Midi.filter_name(),
                    "MIDI export failed",
                );
            }
            Ok(_) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("MIDI export failed: {tool_name} did not produce a valid SMF MThd stream."),
            ),
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("MIDI export failed: {err}"),
            ),
        }
    }

    fn dispatch_scala_request(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        ctx: Option<&egui::Context>,
    ) {
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.scala {
            self.set_export_status(ExportStatusSeverity::Warning, caps.scala_disabled_reason);
            return;
        }
        let tool_name = tool.name().to_string();
        let (scl_src, kbm_src) = match tool.export_scala_tuning() {
            Ok(pair) => pair,
            Err(err) => {
                self.set_export_status(
                    ExportStatusSeverity::Error,
                    format!("Scala tuning export failed: {err}"),
                );
                return;
            }
        };
        let text = if format == ExportFormat::ScalaScl {
            scl_src
        } else {
            kbm_src
        };
        if text.trim().is_empty() {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Scala export failed: {tool_name} returned empty tuning text."),
            );
            return;
        }
        match target {
            ExportTarget::File => {
                let filename = Self::build_export_filename(&tool_name, format);
                let bytes = text.as_bytes().to_vec();
                self.complete_file_export(
                    format,
                    &filename,
                    &bytes,
                    Some(text),
                    format.filter_name(),
                    "Scala export failed",
                );
            }
            ExportTarget::Clipboard => {
                self.dispatch_clipboard_text_request(format, text, ctx);
            }
        }
    }
}
