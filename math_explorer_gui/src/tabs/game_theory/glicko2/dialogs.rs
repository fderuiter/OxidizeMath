//! Async open/save file dialog channel management for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::Glicko2TournamentTool;
use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, OpenDialogConfig, SaveAsDialogConfig,
};
use eframe::egui;
use std::sync::mpsc::{channel, Receiver};

const MATCH_LOG_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Match Log or Preset (.csv, .json)",
    extensions: &["csv", "json"],
    accept: ".csv,.json,text/csv,application/json",
    read_error_prefix: "Failed to read tournament file",
};

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Spreadsheet (.csv)",
    extensions: &["csv"],
    fallback_filename: "glicko2_export.csv",
    write_error_prefix: "Failed to write CSV export",
};

const JSON_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "JSON Document (.json)",
    extensions: &["json"],
    fallback_filename: "glicko2_export.json",
    write_error_prefix: "Failed to write JSON export",
};

#[derive(Default)]
pub(crate) struct TournamentIoState {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
    pub status_message: Option<(bool, String)>,
}

impl Glicko2TournamentTool {
    pub(crate) fn poll_io(&mut self, ctx: &egui::Context) {
        if let Some(Ok(outcome)) = self.io_state.open_rx.as_ref().map(Receiver::try_recv) {
            self.io_state.open_rx = None;
            let status = match outcome {
                Ok((name, raw)) => match self.import_match_log_str(&raw) {
                    Ok(count) => (false, format!("Imported {count} match(es) from {name}")),
                    Err(err) => (true, format!("Import rejected ({name}): {err}")),
                },
                Err(err) => (true, err),
            };
            self.io_state.status_message = Some(status);
            ctx.request_repaint();
        }
        if let Some(Ok(outcome)) = self.io_state.save_rx.as_ref().map(Receiver::try_recv) {
            self.io_state.save_rx = None;
            self.io_state.status_message = Some(match outcome {
                Ok(saved) => (false, format!("Exported tournament file: {saved}")),
                Err(err) => (true, err),
            });
            ctx.request_repaint();
        }
    }
}

impl TournamentIoState {
    pub(crate) fn trigger_import_dialog(&mut self) {
        let (tx, rx) = channel();
        self.open_rx = Some(rx);
        spawn_open_file_dialog(tx, MATCH_LOG_OPEN_CONFIG, |p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("match_log")
                .to_string()
        });
    }

    pub(crate) fn trigger_save_csv(&mut self, default_filename: &str, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(tx, CSV_SAVE_CONFIG, default_filename, content, |p, fb| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(fb)
                .to_string()
        });
    }

    pub(crate) fn trigger_save_json(
        &mut self,
        default_filename: &str,
        content: Result<String, String>,
    ) {
        match content {
            Ok(payload) => {
                let (tx, rx) = channel();
                self.save_rx = Some(rx);
                spawn_save_as_file_dialog(
                    tx,
                    JSON_SAVE_CONFIG,
                    default_filename,
                    payload,
                    |p, fb| {
                        p.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(fb)
                            .to_string()
                    },
                );
            }
            Err(err) => self.status_message = Some((true, err)),
        }
    }
}
