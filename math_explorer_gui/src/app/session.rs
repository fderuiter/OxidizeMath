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
    /// Recent undo history snapshots.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history_undo_stack: Vec<AppState>,
    /// Recent redo history snapshots.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history_redo_stack: Vec<AppState>,
}

/// Manages non-blocking file dialog operations, active workspace file path tracking, and envelope parsing.
pub struct WorkspaceSessionManager {
    /// Active workspace file path for direct single-keystroke saves (`Cmd+S`).
    active_file_path: Option<PathBuf>,
    /// Channel receiver for pending file open dialog operations.
    open_rx: Option<Receiver<Result<(PathBuf, String), String>>>,
    /// Channel receiver for pending file save-as dialog operations.
    save_as_rx: Option<Receiver<Result<PathBuf, String>>>,
    /// Inline UI toast message and error flag.
    toast: Option<(String, bool)>,
    /// Undo history stack.
    undo_stack: Vec<AppState>,
    /// Redo history stack.
    redo_stack: Vec<AppState>,
    /// Maximum depth of history stacks.
    max_history_depth: usize,
    /// Last committed state snapshot for diff tracking.
    last_committed_state: Option<AppState>,
    /// Flag indicating an active drag interaction for coalescing continuous slider edits.
    is_dragging: bool,
    /// Flag indicating an active undo/redo restoration.
    is_restoring: bool,
}

impl Default for WorkspaceSessionManager {
    fn default() -> Self {
        Self {
            active_file_path: None,
            open_rx: None,
            save_as_rx: None,
            toast: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_history_depth: 50,
            last_committed_state: None,
            is_dragging: false,
            is_restoring: false,
        }
    }
}

impl WorkspaceSessionManager {
    /// Creates a new workspace session manager instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a reference to the active workspace file path if set.
    pub fn active_file_path(&self) -> Option<&PathBuf> {
        self.active_file_path.as_ref()
    }

    /// Sets the active workspace file path.
    pub fn set_active_file_path(&mut self, path: Option<PathBuf>) {
        self.active_file_path = path;
    }

    /// Sets an inline toast notification message.
    pub fn set_toast(&mut self, message: impl Into<String>, is_error: bool) {
        self.toast = Some((message.into(), is_error));
    }

    /// Clears the active toast notification message.
    pub fn clear_toast(&mut self) {
        self.toast = None;
    }

    /// Returns the active toast notification message if present.
    pub fn toast(&self) -> Option<&(String, bool)> {
        self.toast.as_ref()
    }

    /// Returns true if an undo operation can be performed.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns true if a redo operation can be performed.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Pushes an application state snapshot onto the undo stack.
    pub fn push_undo(&mut self, state: AppState) {
        if self.is_restoring {
            return;
        }
        if self.undo_stack.last() == Some(&state) {
            return;
        }
        self.undo_stack.push(state);
        if self.undo_stack.len() > self.max_history_depth {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Pushes an application state snapshot onto the redo stack.
    pub fn push_redo(&mut self, state: AppState) {
        self.redo_stack.push(state);
    }

    /// Pops the top state snapshot from the undo stack.
    pub fn pop_undo(&mut self) -> Option<AppState> {
        self.undo_stack.pop()
    }

    /// Pops the top state snapshot from the redo stack.
    pub fn pop_redo(&mut self) -> Option<AppState> {
        self.redo_stack.pop()
    }

    /// Sets the restoration flag to avoid recording snapshots during undo/redo state restoration.
    pub fn set_is_restoring(&mut self, restoring: bool) {
        self.is_restoring = restoring;
    }

    /// Tracks application state changes and updates the history buffer.
    pub fn track_state_changes(&mut self, ctx: &egui::Context, current_state: AppState) {
        if self.is_restoring {
            self.last_committed_state = Some(current_state);
            self.is_restoring = false;
            return;
        }

        if let Some(ref last_state) = self.last_committed_state {
            if &current_state != last_state {
                let pointer_down = ctx.input(|i| i.pointer.any_down());
                if pointer_down {
                    if !self.is_dragging {
                        self.push_undo(last_state.clone());
                        self.is_dragging = true;
                    }
                } else {
                    if !self.is_dragging {
                        self.push_undo(last_state.clone());
                    } else {
                        self.is_dragging = false;
                    }
                }
                self.last_committed_state = Some(current_state);
            } else if !ctx.input(|i| i.pointer.any_down()) {
                self.is_dragging = false;
            }
        } else {
            self.last_committed_state = Some(current_state);
        }
    }

    /// Wraps the given application state inside a versioned workspace envelope and serializes it to JSON.
    pub fn create_envelope(&self, app_state: AppState) -> String {
        let envelope = WorkspaceEnvelope {
            schema_version: "1.0".to_string(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            app_state,
            history_undo_stack: self.undo_stack.clone(),
            history_redo_stack: self.redo_stack.clone(),
        };
        serde_json::to_string_pretty(&envelope).unwrap_or_default()
    }

    /// Validates and parses workspace session content from a JSON string.
    /// Supports both versioned `WorkspaceEnvelope` payloads and legacy/raw `AppState` JSON.
    pub fn parse_envelope(&mut self, json_str: &str) -> Result<AppState, String> {
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
            self.undo_stack = envelope.history_undo_stack;
            self.redo_stack = envelope.history_redo_stack;
            self.last_committed_state = Some(envelope.app_state.clone());
            self.is_restoring = true;
            Ok(envelope.app_state)
        } else {
            let state: AppState = serde_json::from_str(json_str)
                .map_err(|e| format!("Corrupted workspace state JSON: {}", e))?;
            self.undo_stack.clear();
            self.redo_stack.clear();
            self.last_committed_state = Some(state.clone());
            self.is_restoring = true;
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
        if let Some(path) = self.active_file_path().cloned() {
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

    fn reg_shortcut(
        ctx: &egui::Context,
        name: &str,
        cat: &str,
        k: egui::Key,
        m: egui::Modifiers,
    ) -> bool {
        egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            name,
            "",
            egui_plot::commands::CommandTrigger::Shortcut(m, k),
            true,
            cat,
            None,
            None,
        )
    }

    /// Registers global workspace session keyboard shortcuts and checks for user triggers.
    pub fn register_and_check_shortcuts(&mut self, ctx: &egui::Context, app: &mut MathExplorerApp) {
        let modifiers = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD
        } else {
            egui::Modifiers::CTRL
        };
        let shift_modifiers = modifiers | egui::Modifiers::SHIFT;

        let (undo, redo) = Self::check_history_shortcuts(ctx, modifiers, shift_modifiers);
        let (open, save_as, save) = Self::check_file_shortcuts(ctx, modifiers, shift_modifiers);

        if undo {
            app.undo();
        } else if redo {
            app.redo();
        } else if open {
            self.trigger_open();
        } else if save_as {
            let state = app.get_app_state();
            let env_json = self.create_envelope(state);
            self.trigger_save_as(env_json);
        } else if save {
            let state = app.get_app_state();
            let env_json = self.create_envelope(state);
            self.trigger_save(env_json);
        }
    }

    fn check_history_shortcuts(
        ctx: &egui::Context,
        m: egui::Modifiers,
        sm: egui::Modifiers,
    ) -> (bool, bool) {
        let undo = Self::reg_shortcut(ctx, "Undo", "Edit", egui::Key::Z, m);
        let redo_shift = Self::reg_shortcut(ctx, "Redo", "Edit", egui::Key::Z, sm);
        let redo_ctrl_y = !cfg!(target_os = "macos")
            && Self::reg_shortcut(ctx, "Redo (Ctrl+Y)", "Edit", egui::Key::Y, m);

        (undo, redo_shift || redo_ctrl_y)
    }

    fn check_file_shortcuts(
        ctx: &egui::Context,
        m: egui::Modifiers,
        sm: egui::Modifiers,
    ) -> (bool, bool, bool) {
        let open = Self::reg_shortcut(ctx, "Open Workspace Session...", "File", egui::Key::O, m);
        let save_as = Self::reg_shortcut(
            ctx,
            "Save Workspace Session As...",
            "File",
            egui::Key::S,
            sm,
        );
        let save = Self::reg_shortcut(ctx, "Save Workspace Session", "File", egui::Key::S, m);

        (open, save_as, save)
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
                            self.set_active_file_path(Some(path.clone()));
                            let file_name = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("workspace.mws");
                            let msg = format!("Loaded workspace session from {}", file_name);
                            self.set_toast(&msg, false);
                            log_telemetry(msg, Severity::Info, ctx, "aria_live_message");
                        }
                        Err(err) => {
                            self.set_toast(&err, true);
                            log_telemetry(err, Severity::Error, ctx, "aria_live_assertive_message");
                        }
                    },
                    Err(err) => {
                        self.set_toast(&err, true);
                        log_telemetry(err, Severity::Error, ctx, "aria_live_assertive_message");
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
                        self.set_active_file_path(Some(path.clone()));
                        let file_name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("workspace.mws");
                        let msg = format!("Exported workspace session to {}", file_name);
                        self.set_toast(&msg, false);
                        log_telemetry(msg, Severity::Info, ctx, "aria_live_message");
                    }
                    Err(err) => {
                        self.set_toast(&err, true);
                        log_telemetry(err, Severity::Error, ctx, "aria_live_assertive_message");
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
        if let Some((msg, is_error)) = self.toast() {
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
                self.clear_toast();
            }
        }
    }
}

fn log_telemetry(msg: String, severity: Severity, ctx: &egui::Context, aria_key: &str) {
    global_registry().emit(TelemetryEvent {
        source: "WorkspaceSessionManager".to_string(),
        severity,
        message: msg.clone(),
        metadata: std::collections::HashMap::new(),
        thread_name: None,
    });
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(aria_key), msg);
    });
}
