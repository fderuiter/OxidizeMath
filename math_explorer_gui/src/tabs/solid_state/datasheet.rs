//! Shared Markdown and CSV datasheet export controls for solid-state physics workbenches.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::web_io::{sanitize_download_filename, spawn_save_as_file_dialog, SaveAsDialogConfig};
use eframe::egui;
use std::sync::mpsc::{channel, Receiver, TryRecvError};

/// Non-blocking save/download channel and ARIA status state for `.md` and `.csv` datasheets.
#[derive(Default)]
pub struct DatasheetExportState {
    save_rx: Option<Receiver<Result<String, String>>>,
    /// Last polite confirmation message, if any.
    pub status_info: Option<String>,
    /// Last assertive save error message, if any.
    pub status_error: Option<String>,
}

impl DatasheetExportState {
    /// Records a polite status update and announces it via the live-region bridge.
    pub fn record_info(&mut self, msg: String) {
        self.status_error = None;
        crate::accessibility::announce_status(&msg);
        self.status_info = Some(msg);
    }

    /// Records an assertive error message and announces it via the live-region bridge.
    pub fn record_error(&mut self, err: String) {
        self.status_info = None;
        crate::accessibility::announce_status_with_priority(&err, "assertive");
        self.status_error = Some(err);
    }

    /// Polls any pending asynchronous file-dialog save completion.
    pub fn poll(&mut self) {
        let Some(rx) = &self.save_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(filename)) => {
                self.save_rx = None;
                self.record_info(format!("Saved datasheet to {filename}"));
            }
            Ok(Err(err)) => {
                self.save_rx = None;
                self.record_error(err);
            }
            Err(TryRecvError::Disconnected) => {
                self.save_rx = None;
            }
            Err(TryRecvError::Empty) => {}
        }
    }

    /// Launches a native Save-As dialog or browser `.md` download for a Markdown datasheet.
    pub fn trigger_save_markdown(&mut self, fallback_filename: &'static str, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(
            tx,
            SaveAsDialogConfig {
                filter_name: "Markdown Physics Datasheet (*.md)",
                extensions: &["md"],
                fallback_filename,
                write_error_prefix: "Failed to save Markdown physics datasheet",
            },
            fallback_filename,
            content,
            |path, fallback| sanitize_download_filename(&path.to_string_lossy(), fallback),
        );
    }

    /// Launches a native Save-As dialog or browser `.csv` download for a CSV datasheet.
    pub fn trigger_save_csv(&mut self, fallback_filename: &'static str, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(
            tx,
            SaveAsDialogConfig {
                filter_name: "CSV Physics Datasheet (*.csv)",
                extensions: &["csv"],
                fallback_filename,
                write_error_prefix: "Failed to save CSV physics datasheet",
            },
            fallback_filename,
            content,
            |path, fallback| sanitize_download_filename(&path.to_string_lossy(), fallback),
        );
    }

    /// Renders clipboard copy and `.md` / `.csv` datasheet download buttons plus semantic status text.
    pub fn render_controls(
        &mut self,
        ui: &mut egui::Ui,
        heading: &str,
        md_filename: &'static str,
        csv_filename: &'static str,
        build_md: impl Fn() -> String,
        build_csv: impl Fn() -> String,
    ) {
        ui.group(|ui| {
            ui.label(egui::RichText::new(heading).strong());
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button("📋 Copy .md Datasheet")
                    .accessible_hover_text(
                        "Copy standalone Markdown physics datasheet to clipboard",
                    )
                    .clicked()
                {
                    ui.ctx().copy_text(build_md());
                    self.record_info("Copied Markdown datasheet to clipboard".into());
                }
                if ui
                    .button("💾 Save .md Datasheet")
                    .accessible_hover_text("Save or download Markdown physics datasheet (.md)")
                    .clicked()
                {
                    self.trigger_save_markdown(md_filename, build_md());
                }
            });
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button("📋 Copy .csv Datasheet")
                    .accessible_hover_text("Copy CSV physics summary datasheet to clipboard")
                    .clicked()
                {
                    ui.ctx().copy_text(build_csv());
                    self.record_info("Copied CSV datasheet to clipboard".into());
                }
                if ui
                    .button("💾 Save .csv Datasheet")
                    .accessible_hover_text("Save or download CSV physics summary datasheet (.csv)")
                    .clicked()
                {
                    self.trigger_save_csv(csv_filename, build_csv());
                }
            });
            self.render_status(ui);
        });
    }

    fn render_status(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref info) = self.status_info {
            ui.colored_label(palette.success, format!("✓ {info}"));
        }
        if let Some(ref err) = self.status_error {
            ui.colored_label(palette.error, format!("⚠ {err}"));
        }
    }
}

// [cite:mmwave_radiotherapy_setup]
