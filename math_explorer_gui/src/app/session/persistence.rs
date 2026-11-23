use super::io::{
    file_name_str, handle_transition_error, notify, spawn_open_dialog, spawn_save_as_dialog,
    spawn_save_direct,
};
use super::{SessionLifecycleState, WorkspaceEnvelope, WorkspaceSessionManager};
use crate::app::{AppState, MathExplorerApp};
use eframe::egui;
use std::sync::mpsc::{channel, TryRecvError};

impl WorkspaceSessionManager {
    /// Wraps the given application state inside a versioned workspace envelope and serializes it to JSON.
    pub fn create_envelope(&self, app_state: AppState) -> String {
        let envelope = self.build_envelope_struct(app_state);
        serde_json::to_string_pretty(&envelope).unwrap_or_default()
    }

    /// Serializes a compact `WorkspaceEnvelope` (including undo/redo stacks) for `eframe::Storage`.
    pub fn create_compact_envelope(&self, app_state: AppState) -> String {
        let envelope = self.build_envelope_struct(app_state);
        serde_json::to_string(&envelope).unwrap_or_default()
    }

    fn build_envelope_struct(&self, app_state: AppState) -> WorkspaceEnvelope {
        WorkspaceEnvelope {
            schema_version: "1.0".to_string(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            app_state,
            history_undo_stack: self.undo_stack.clone(),
            history_redo_stack: self.redo_stack.clone(),
            callout_manager: self.callout_manager.clone(),
        }
    }

    /// Validates and parses workspace session content from a JSON string.
    pub fn parse_envelope(&mut self, json_str: &str) -> Result<AppState, String> {
        let cleaned = egui_plot::export::strip_utf8_bom(json_str);
        let val: serde_json::Value = serde_json::from_str(cleaned)
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
            self.undo_stack = envelope.history_undo_stack;
            self.redo_stack = envelope.history_redo_stack;
            self.callout_manager = envelope.callout_manager;
            self.last_committed_state = Some(envelope.app_state.clone());
            self.last_committed_fingerprint = None;
            self.set_is_restoring(true);
            Ok(envelope.app_state)
        } else {
            self.undo_stack.clear();
            self.redo_stack.clear();
            self.callout_manager = None;
            let state: AppState = serde_json::from_str(json_str)
                .map_err(|e| format!("Corrupted workspace state JSON: {}", e))?;
            self.last_committed_state = Some(state.clone());
            self.last_committed_fingerprint = None;
            self.set_is_restoring(true);
            Ok(state)
        }
    }

    fn is_dialog_busy(&self) -> bool {
        self.lifecycle_state == SessionLifecycleState::AwaitingFileDialog
            || self.open_rx.is_some()
            || self.save_as_rx.is_some()
            || self.save_rx.is_some()
    }

    /// Triggers a non-blocking background file picker for opening a workspace session file.
    pub fn trigger_open(&mut self) {
        if self.is_dialog_busy() {
            return;
        }
        self.lifecycle_state = SessionLifecycleState::AwaitingFileDialog;
        let (tx, rx) = channel();
        self.open_rx = Some(rx);
        spawn_open_dialog(tx);
    }

    /// Triggers a non-blocking background file dialog for saving a workspace session file to a new path.
    pub fn trigger_save_as(&mut self, envelope_json: String) {
        if self.is_dialog_busy() {
            return;
        }
        self.lifecycle_state = SessionLifecycleState::AwaitingFileDialog;
        let (tx, rx) = channel();
        self.save_as_rx = Some(rx);
        spawn_save_as_dialog(tx, envelope_json);
    }

    /// Saves the workspace session state directly if an active file path exists, or prompts Save As otherwise.
    pub fn trigger_save(&mut self, envelope_json: String) {
        if let Some(path) = self.active_file_path().cloned() {
            if self.lifecycle_state == SessionLifecycleState::AwaitingFileDialog
                || self.lifecycle_state == SessionLifecycleState::Saving
                || self.save_rx.is_some()
            {
                return;
            }
            self.lifecycle_state = SessionLifecycleState::Saving;
            let (tx, rx) = channel();
            self.save_rx = Some(rx);
            spawn_save_direct(tx, path, envelope_json);
        } else {
            self.trigger_save_as(envelope_json);
        }
    }

    fn poll_open(&mut self, app: &mut MathExplorerApp, ctx: &egui::Context) {
        if let Some(rx) = &self.open_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.open_rx = None;
                    if self.lifecycle_state != SessionLifecycleState::AwaitingFileDialog {
                        handle_transition_error(self, ctx, "open");
                        return;
                    }
                    self.lifecycle_state = SessionLifecycleState::Loading;
                    match res {
                        Ok((path, content)) => match self.parse_envelope(&content) {
                            Ok(app_state) => {
                                app.apply_app_state(app_state);
                                self.set_active_file_path(Some(path.clone()));
                                let name = file_name_str(&path);
                                notify(
                                    self,
                                    ctx,
                                    format!("Loaded workspace session from {name}"),
                                    false,
                                );
                            }
                            Err(err) => {
                                notify(self, ctx, err, true);
                                self.lifecycle_state = SessionLifecycleState::Error;
                            }
                        },
                        Err(err) => {
                            notify(self, ctx, err, true);
                            self.lifecycle_state = SessionLifecycleState::Error;
                        }
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    self.open_rx = None;
                    if self.lifecycle_state == SessionLifecycleState::AwaitingFileDialog {
                        self.lifecycle_state = SessionLifecycleState::Idle;
                    }
                }
                Err(TryRecvError::Empty) => {}
            }
        }
    }

    fn poll_save_as(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.save_as_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.save_as_rx = None;
                    if self.lifecycle_state != SessionLifecycleState::AwaitingFileDialog {
                        handle_transition_error(self, ctx, "save-as");
                        return;
                    }
                    self.lifecycle_state = SessionLifecycleState::Saving;
                    match res {
                        Ok(path) => {
                            self.set_active_file_path(Some(path.clone()));
                            let name = file_name_str(&path);
                            notify(
                                self,
                                ctx,
                                format!("Exported workspace session to {name}"),
                                false,
                            );
                            self.lifecycle_state = SessionLifecycleState::Idle;
                        }
                        Err(err) => {
                            notify(self, ctx, err, true);
                            self.lifecycle_state = SessionLifecycleState::Error;
                        }
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    self.save_as_rx = None;
                    if self.lifecycle_state == SessionLifecycleState::AwaitingFileDialog {
                        self.lifecycle_state = SessionLifecycleState::Idle;
                    }
                }
                Err(TryRecvError::Empty) => {}
            }
        }
    }

    fn poll_save(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.save_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.save_rx = None;
                    if self.lifecycle_state != SessionLifecycleState::Saving {
                        handle_transition_error(self, ctx, "save");
                        return;
                    }
                    match res {
                        Ok(path) => {
                            let name = file_name_str(&path);
                            notify(
                                self,
                                ctx,
                                format!("Workspace session saved to {name}"),
                                false,
                            );
                            self.lifecycle_state = SessionLifecycleState::Idle;
                        }
                        Err(err) => {
                            notify(self, ctx, err, true);
                            self.lifecycle_state = SessionLifecycleState::Error;
                        }
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    self.save_rx = None;
                    if self.lifecycle_state == SessionLifecycleState::Saving {
                        self.lifecycle_state = SessionLifecycleState::Idle;
                    }
                }
                Err(TryRecvError::Empty) => {}
            }
        }
    }

    /// Polls pending background file dialog channels and applies loaded workspace state or logs diagnostic events.
    pub fn poll(&mut self, app: &mut MathExplorerApp, ctx: &egui::Context) {
        self.poll_open(app, ctx);
        self.poll_save_as(ctx);
        self.poll_save(ctx);
    }

    /// Renders any active workspace toast notification message in the UI.
    pub fn render_toast(&mut self, ui: &mut egui::Ui) {
        if let Some((msg, is_error)) = self.toast() {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let mut dismiss = false;
            ui.horizontal(|ui| {
                let (color, icon) = if *is_error {
                    (palette.error, "⚠️")
                } else {
                    (palette.success, "ℹ️")
                };
                ui.label(egui::RichText::new(format!("{} {}", icon, msg)).color(color));
                if ui.small_button("✖").clicked() {
                    dismiss = true;
                }
            });
            if dismiss {
                self.clear_toast();
            }
        }
    }
}

impl MathExplorerApp {
    /// Persists a compact `WorkspaceEnvelope` (including undo/redo stacks) to `eframe::Storage`.
    pub(crate) fn save_envelope_to_storage(&mut self, storage: &mut dyn eframe::Storage) {
        self.clear_active_pane_caches();
        let envelope_json = self
            .session_manager
            .create_compact_envelope(self.get_app_state());
        storage.set_string(eframe::APP_KEY, envelope_json);
    }

    /// Parses and restores either a `WorkspaceEnvelope` or legacy `AppState` JSON payload,
    /// returning `true` when valid state was applied.
    pub(crate) fn try_load_state_from_json(&mut self, json_str: &str) -> bool {
        match self.session_manager.parse_envelope(json_str) {
            Ok(app_state) => {
                self.apply_app_state(app_state);
                true
            }
            Err(err) => {
                crate::telemetry::emit_telemetry(
                    "MathExplorerApp",
                    federated_registry::Severity::Warning,
                    format!(
                        "Failed to parse saved session state, resetting to defaults: {}",
                        err
                    ),
                    std::collections::HashMap::new(),
                );
                false
            }
        }
    }
}
