//! Deferred end-of-frame `.wav`, `.mid`, `.scl`, and `.kbm` acoustic & microtonal export dispatch.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::export_action_button;
use super::*;
use eframe::egui;

impl SimulationFramework {
    /// Renders `.wav`, `.mid`, `.scl`, and `.kbm` toolbar/menu buttons for acoustic-capable tools.
    pub(super) fn render_audio_export_items(
        &mut self,
        ui: &mut egui::Ui,
        caps: ExportCapabilities,
        close_on_click: bool,
    ) {
        if !caps.has_audio_export() {
            return;
        }
        if caps.wav
            && export_action_button(
                ui,
                true,
                "Export WAV (.wav)",
                "Synthesize and download stereo 16-bit PCM WAV audio or impulse response",
                close_on_click,
            )
        {
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
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.wav {
            self.set_export_status(ExportStatusSeverity::Warning, caps.wav_disabled_reason);
            return;
        }
        let cfg = crate::audio::current_offline_config();
        let tool_name = tool.name().to_string();
        match tool.export_wav_audio(cfg.sample_rate, cfg.duration_secs) {
            Ok(bytes) if bytes.len() > 44 => {
                let filename = Self::build_export_filename(&tool_name, ExportFormat::Wav);
                self.complete_file_export(
                    ExportFormat::Wav,
                    &filename,
                    &bytes,
                    None,
                    ExportFormat::Wav.filter_name(),
                    "WAV export failed",
                );
            }
            Ok(_) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("WAV export failed: {tool_name} produced an empty PCM payload."),
            ),
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("WAV export failed: {err}"),
            ),
        }
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
