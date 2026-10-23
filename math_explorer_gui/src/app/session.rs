use crate::app::AppState;
use crate::app::MathExplorerApp;
use eframe::egui;
use federated_registry::{global_registry, Severity, TelemetryEvent};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};

/// Metadata envelope wrapping application state for versioned workspace session persistence.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WorkspaceEnvelope {
    /// Schema version for envelope serialization format (e.g. "1.0").
    pub schema_version: String,
    /// Application version emitting the envelope.
    pub app_version: String,
    /// ISO 8601 creation or export timestamp string.
    pub timestamp: String,
    /// Wrapped application state payload.
    pub app_state: AppState,
}

/// Manages non-blocking file dialog operations, active workspace file path tracking, and envelope parsing.
#[derive(Default)]
pub struct WorkspaceSessionManager {
    /// Active workspace file path for direct single-keystroke saves (`Cmd+S`).
    active_file_path: Option<PathBuf>,
    /// Channel receiver for pending file open dialog operations.
    open_rx: Option<Receiver<Result<(PathBuf, String), String>>>,
    /// Channel receiver for pending file save-as dialog operations.
    save_as_rx: Option<Receiver<Result<PathBuf, String>>>,
    /// Inline UI toast message and error flag.
    toast: Option<(String, bool)>,
}

impl WorkspaceSessionManager {
    /// Creates a new workspace session manager instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a reference to the active workspace file path if set.
    #[allow(dead_code)]
    pub fn active_file_path(&self) -> Option<&PathBuf> {
        self.active_file_path.as_ref()
    }

    /// Sets the active workspace file path.
    #[allow(dead_code)]
    pub fn set_active_file_path(&mut self, path: Option<PathBuf>) {
        self.active_file_path = path;
    }

    /// Sets an inline toast notification message.
    pub fn set_toast(&mut self, message: impl Into<String>, is_error: bool) {
        self.toast = Some((message.into(), is_error));
    }

    /// Clears the active toast notification message.
    #[allow(dead_code)]
    pub fn clear_toast(&mut self) {
        self.toast = None;
    }

    /// Returns the active toast notification message if present.
    #[allow(dead_code)]
    pub fn toast(&self) -> Option<&(String, bool)> {
        self.toast.as_ref()
    }

    /// Wraps the given application state inside a versioned workspace envelope and serializes it to JSON.
    pub fn create_envelope(&self, app_state: AppState) -> String {
        let envelope = WorkspaceEnvelope {
            schema_version: "1.0".to_string(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            app_state,
        };
        serde_json::to_string_pretty(&envelope).unwrap_or_default()
    }

    /// Validates and parses workspace session content from a JSON string.
    /// Supports both versioned `WorkspaceEnvelope` payloads and legacy/raw `AppState` JSON.
    pub fn parse_envelope(&self, json_str: &str) -> Result<AppState, String> {
        let val: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| format!("Invalid workspace file format: {}", e))?;

        if val.get("schema_version").is_some() {
            let envelope: WorkspaceEnvelope = serde_json::from_value(val)
                .map_err(|e| format!("Corrupted workspace envelope metadata: {}", e))?;

            let major_ver = envelope
                .schema_version
                .split('.')
                .next()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0);

            if major_ver > 1 {
                return Err(format!(
                    "Incompatible schema version '{}' (supported: <= 1.0)",
                    envelope.schema_version
                ));
            }
            Ok(envelope.app_state)
        } else {
            let state: AppState = serde_json::from_str(json_str)
                .map_err(|e| format!("Corrupted workspace state JSON: {}", e))?;
            Ok(state)
        }
    }

    /// Triggers a non-blocking background file picker for opening a workspace session file.
    pub fn trigger_open(&mut self) {
        let (tx, rx) = channel();
        self.open_rx = Some(rx);

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Workspace Session (*.mws, *.json)", &["mws", "json"])
                    .pick_file()
                {
                    match std::fs::read_to_string(&path) {
                        Ok(content) => {
                            let _ = tx.send(Ok((path, content)));
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Failed to read workspace file: {}", e)));
                        }
                    }
                }
            });
        }
    }

    /// Triggers a non-blocking background file dialog for saving a workspace session file to a new path.
    pub fn trigger_save_as(&mut self, envelope_json: String) {
        let (tx, rx) = channel();
        self.save_as_rx = Some(rx);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let default_name = "workspace.mws".to_string();
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .set_file_name(&default_name)
                    .add_filter("Workspace Session (*.mws, *.json)", &["mws", "json"])
                    .save_file()
                {
                    match std::fs::write(&path, envelope_json) {
                        Ok(_) => {
                            let _ = tx.send(Ok(path));
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Failed to write workspace file: {}", e)));
                        }
                    }
                }
            });
        }
    }

    /// Saves the workspace session state directly if an active file path exists, or prompts Save As otherwise.
    pub fn trigger_save(&mut self, envelope_json: String) {
        if let Some(path) = self.active_file_path.clone() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let path_clone = path.clone();
                std::thread::spawn(move || {
                    let _ = std::fs::write(path_clone, envelope_json);
                });
            }
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("workspace.mws");
            self.set_toast(format!("Workspace session saved to {}", file_name), false);
        } else {
            self.trigger_save_as(envelope_json);
        }
    }

    /// Registers global workspace session keyboard shortcuts and checks for user triggers.
    pub fn register_and_check_shortcuts(&mut self, ctx: &egui::Context, app: &mut MathExplorerApp) {
        let modifiers = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD
        } else {
            egui::Modifiers::CTRL
        };
        let shift_modifiers = modifiers | egui::Modifiers::SHIFT;

        let open_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Open Workspace Session...",
            "Open a workspace session file (.mws or .json)",
            egui_plot::commands::CommandTrigger::Shortcut(modifiers, egui::Key::O),
            true,
            "File",
            None,
            None,
        );

        let save_as_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Save Workspace Session As...",
            "Save active workspace session to a new file",
            egui_plot::commands::CommandTrigger::Shortcut(shift_modifiers, egui::Key::S),
            true,
            "File",
            None,
            None,
        );

        let save_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Save Workspace Session",
            "Save current workspace session state to disk",
            egui_plot::commands::CommandTrigger::Shortcut(modifiers, egui::Key::S),
            true,
            "File",
            None,
            None,
        );

        if open_triggered {
            self.trigger_open();
        } else if save_as_triggered {
            let state = app.get_app_state();
            let env_json = self.create_envelope(state);
            self.trigger_save_as(env_json);
        } else if save_triggered {
            let state = app.get_app_state();
            let env_json = self.create_envelope(state);
            self.trigger_save(env_json);
        }
    }

    /// Polls pending background file open channel.
    fn poll_open(&mut self, app: &mut MathExplorerApp, ctx: &egui::Context) {
        if let Some(rx) = &self.open_rx {
            if let Ok(res) = rx.try_recv() {
                self.open_rx = None;
                match res {
                    Ok((path, content)) => match self.parse_envelope(&content) {
                        Ok(app_state) => {
                            app.apply_app_state(app_state);
                            self.active_file_path = Some(path.clone());
                            let file_name = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("workspace.mws");
                            let msg = format!("Loaded workspace session from {}", file_name);
                            self.set_toast(&msg, false);

                            global_registry().emit(TelemetryEvent {
                                source: "WorkspaceSessionManager".to_string(),
                                severity: Severity::Info,
                                message: msg.clone(),
                                metadata: std::collections::HashMap::new(),
                                thread_name: None,
                            });
                            ctx.data_mut(|d| {
                                d.insert_temp(egui::Id::new("aria_live_message"), msg);
                            });
                        }
                        Err(err) => {
                            self.set_toast(&err, true);
                            global_registry().emit(TelemetryEvent {
                                source: "WorkspaceSessionManager".to_string(),
                                severity: Severity::Error,
                                message: err.clone(),
                                metadata: std::collections::HashMap::new(),
                                thread_name: None,
                            });
                            ctx.data_mut(|d| {
                                d.insert_temp(egui::Id::new("aria_live_assertive_message"), err);
                            });
                        }
                    },
                    Err(err) => {
                        self.set_toast(&err, true);
                        global_registry().emit(TelemetryEvent {
                            source: "WorkspaceSessionManager".to_string(),
                            severity: Severity::Error,
                            message: err.clone(),
                            metadata: std::collections::HashMap::new(),
                            thread_name: None,
                        });
                    }
                }
            }
        }
    }

    /// Polls pending background file save-as channel.
    fn poll_save_as(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.save_as_rx {
            if let Ok(res) = rx.try_recv() {
                self.save_as_rx = None;
                match res {
                    Ok(path) => {
                        self.active_file_path = Some(path.clone());
                        let file_name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("workspace.mws");
                        let msg = format!("Exported workspace session to {}", file_name);
                        self.set_toast(&msg, false);

                        global_registry().emit(TelemetryEvent {
                            source: "WorkspaceSessionManager".to_string(),
                            severity: Severity::Info,
                            message: msg.clone(),
                            metadata: std::collections::HashMap::new(),
                            thread_name: None,
                        });
                        ctx.data_mut(|d| {
                            d.insert_temp(egui::Id::new("aria_live_message"), msg);
                        });
                    }
                    Err(err) => {
                        self.set_toast(&err, true);
                        global_registry().emit(TelemetryEvent {
                            source: "WorkspaceSessionManager".to_string(),
                            severity: Severity::Error,
                            message: err.clone(),
                            metadata: std::collections::HashMap::new(),
                            thread_name: None,
                        });
                    }
                }
            }
        }
    }

    /// Polls pending background file dialog channels and applies loaded workspace state or logs diagnostic events.
    pub fn poll(&mut self, app: &mut MathExplorerApp, ctx: &egui::Context) {
        self.poll_open(app, ctx);
        self.poll_save_as(ctx);
    }

    /// Renders any active workspace toast notification message in the UI.
    pub fn render_toast(&mut self, ui: &mut egui::Ui) {
        if let Some((msg, is_error)) = &self.toast {
            let mut dismiss = false;
            ui.horizontal(|ui| {
                let color = if *is_error {
                    egui::Color32::RED
                } else {
                    egui::Color32::LIGHT_GREEN
                };
                let icon = if *is_error { "⚠️" } else { "ℹ️" };
                ui.label(egui::RichText::new(format!("{} {}", icon, msg)).color(color));
                if ui.small_button("✖").clicked() {
                    dismiss = true;
                }
            });
            if dismiss {
                self.toast = None;
            }
        }
    }
}
