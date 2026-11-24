//! Shared non-blocking file import and clinical artifact save helpers for `ClinicalTrialsTab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::SemanticPalette;
use crate::web_io::{
    sanitize_download_filename, spawn_open_file_dialog, spawn_save_as_file_dialog,
    OpenDialogConfig, SaveAsDialogConfig,
};
use eframe::egui;
use std::sync::mpsc::{channel, Receiver, TryRecvError};

/// Standard open-file filter configuration for local HIPAA-safe CSV and TSV cohort files.
pub const COHORT_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Clinical Cohort Table (*.csv, *.tsv, *.txt)",
    extensions: &["csv", "tsv", "txt"],
    accept: ".csv,.tsv,.txt,text/csv,text/tab-separated-values,text/plain",
    read_error_prefix: "Failed to read local cohort file",
};

/// Non-blocking channel and ARIA status state for clinical cohort imports and report downloads.
#[derive(Default)]
pub struct ClinicalIoState {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
    /// Latest success confirmation message, if any.
    pub status_info: Option<String>,
    /// Latest validation or I/O error message, if any.
    pub status_error: Option<String>,
}

impl ClinicalIoState {
    /// Records a polite ARIA status announcement and green confirmation message.
    pub fn record_info(&mut self, msg: String) {
        self.status_error = None;
        crate::accessibility::announce_status(&msg);
        self.status_info = Some(msg);
    }

    /// Records an assertive ARIA alert and red error banner.
    pub fn record_error(&mut self, err: String) {
        self.status_info = None;
        crate::accessibility::announce_status_with_priority(&err, "assertive");
        self.status_error = Some(err);
    }

    /// Polls any pending save-as dialog or browser download channel.
    pub fn poll_save(&mut self, artifact_label: &str) {
        let Some(rx) = &self.save_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(filename)) => {
                self.save_rx = None;
                self.record_info(format!("Saved {artifact_label} to {filename}"));
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

    /// Polls any pending local file-open picker channel, returning `(filename, raw_text)`.
    pub fn poll_open(&mut self) -> Option<Result<(String, String), String>> {
        let rx = self.open_rx.as_ref()?;
        match rx.try_recv() {
            Ok(outcome) => {
                self.open_rx = None;
                Some(outcome)
            }
            Err(TryRecvError::Disconnected) => {
                self.open_rx = None;
                None
            }
            Err(TryRecvError::Empty) => None,
        }
    }

    /// Opens a native `rfd` or browser `<input type="file">` picker for `.csv` / `.tsv` cohorts.
    pub fn trigger_open_cohort(&mut self) {
        let (tx, rx) = channel();
        self.open_rx = Some(rx);
        spawn_open_file_dialog(tx, COHORT_OPEN_CONFIG, |path| {
            sanitize_download_filename(&path.to_string_lossy(), "cohort.csv")
        });
    }

    /// Saves a UTF-8 text artifact using the given [`SaveAsDialogConfig`] and default filename.
    pub fn trigger_save_with_config(
        &mut self,
        config: SaveAsDialogConfig,
        default_filename: &str,
        content: String,
    ) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(tx, config, default_filename, content, |path, fallback| {
            sanitize_download_filename(&path.to_string_lossy(), fallback)
        });
    }

    /// Saves a Markdown clinical report or protocol card (`.md`).
    pub fn trigger_save_markdown(&mut self, fallback_filename: &'static str, content: String) {
        self.trigger_save_with_config(
            SaveAsDialogConfig {
                filter_name: "Markdown Clinical Report (*.md)",
                extensions: &["md"],
                fallback_filename,
                write_error_prefix: "Failed to save Markdown clinical report",
            },
            fallback_filename,
            content,
        );
    }

    /// Saves a CSV clinical table or schedule (`.csv`).
    pub fn trigger_save_csv(&mut self, fallback_filename: &'static str, content: String) {
        self.trigger_save_with_config(
            SaveAsDialogConfig {
                filter_name: "CSV Clinical Table (*.csv)",
                extensions: &["csv"],
                fallback_filename,
                write_error_prefix: "Failed to save CSV clinical table",
            },
            fallback_filename,
            content,
        );
    }

    /// Saves a JSON clinical allocation schedule or structured export (`.json`).
    pub fn trigger_save_json(&mut self, fallback_filename: &'static str, content: String) {
        self.trigger_save_with_config(
            SaveAsDialogConfig {
                filter_name: "JSON Clinical Schedule (*.json)",
                extensions: &["json"],
                fallback_filename,
                write_error_prefix: "Failed to save JSON clinical schedule",
            },
            fallback_filename,
            content,
        );
    }

    /// Saves an SVG clinical figure (`.svg`).
    pub fn trigger_save_svg(&mut self, fallback_filename: &'static str, content: String) {
        self.trigger_save_with_config(
            SaveAsDialogConfig {
                filter_name: "SVG Clinical Figure (*.svg)",
                extensions: &["svg"],
                fallback_filename,
                write_error_prefix: "Failed to save SVG clinical figure",
            },
            fallback_filename,
            content,
        );
    }

    /// Renders themed status feedback labels when present.
    pub fn render_status(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref info) = self.status_info {
            ui.colored_label(palette.success, format!("✓ {info}"));
        }
        if let Some(ref err) = self.status_error {
            ui.colored_label(palette.error, format!("⚠ {err}"));
        }
    }
}

// [cite:clinical_trials_statistics]
