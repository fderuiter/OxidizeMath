mod io;
mod shortcuts;
mod types;

pub use types::{SessionLifecycleState, WorkspaceEnvelope};

use crate::app::AppState;
use crate::app::MathExplorerApp;
use eframe::egui;
use io::{file_name_str, handle_transition_error, notify};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, TryRecvError};

/// Manages non-blocking file dialog operations, active workspace file path tracking, and envelope parsing.
pub struct WorkspaceSessionManager {
    /// Active workspace file path for direct single-keystroke saves (`Cmd+S`).
    active_file_path: Option<PathBuf>,
    /// Channel receiver for pending file open dialog operations.
    pub(crate) open_rx: Option<Receiver<Result<(PathBuf, String), String>>>,
    /// Channel receiver for pending file save-as dialog operations.
    pub(crate) save_as_rx: Option<Receiver<Result<PathBuf, String>>>,
    /// Channel receiver for pending direct save operations.
    pub(crate) save_rx: Option<Receiver<Result<PathBuf, String>>>,
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
    /// Explicit session lifecycle state.
    pub(crate) lifecycle_state: SessionLifecycleState,
    /// Number of consecutive quiescent frames observed during history restoration.
    pub(crate) quiescent_frames: usize,
    /// Active callout manager instance.
    callout_manager: Option<crate::framework::CalloutManager>,
}

impl Default for WorkspaceSessionManager {
    fn default() -> Self {
        Self {
            active_file_path: None,
            open_rx: None,
            save_as_rx: None,
            save_rx: None,
            toast: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_history_depth: 50,
            last_committed_state: None,
            lifecycle_state: SessionLifecycleState::Idle,
            quiescent_frames: 0,
            callout_manager: None,
        }
    }
}

impl WorkspaceSessionManager {
    /// Creates a new workspace session manager instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current session lifecycle state.
    #[cfg(test)]
    #[must_use]
    pub fn lifecycle_state(&self) -> SessionLifecycleState {
        self.lifecycle_state
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
        if self.lifecycle_state == SessionLifecycleState::Error {
            self.lifecycle_state = SessionLifecycleState::Idle;
        }
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
        if self.lifecycle_state == SessionLifecycleState::RestoringHistory {
            return;
        }
        self.push_undo_preserving_redo(state);
        self.redo_stack.clear();
    }

    /// Pushes an application state snapshot onto the undo stack without clearing redo history.
    pub fn push_undo_preserving_redo(&mut self, state: AppState) {
        if self.undo_stack.last() == Some(&state) {
            return;
        }
        self.undo_stack.push(state);
        if self.undo_stack.len() > self.max_history_depth {
            self.undo_stack.remove(0);
        }
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

    /// Returns true if currently in state restoration.
    #[cfg(test)]
    #[must_use]
    pub fn is_restoring(&self) -> bool {
        self.lifecycle_state == SessionLifecycleState::RestoringHistory
    }

    /// Sets the restoration flag to avoid recording snapshots during undo/redo state restoration.
    pub fn set_is_restoring(&mut self, restoring: bool) {
        if restoring {
            self.lifecycle_state = SessionLifecycleState::RestoringHistory;
            self.quiescent_frames = 0;
        } else if self.lifecycle_state == SessionLifecycleState::RestoringHistory {
            self.lifecycle_state = SessionLifecycleState::Idle;
            self.quiescent_frames = 0;
        }
    }

    /// Tracks application state changes and updates the history buffer.
    pub fn track_state_changes(&mut self, ctx: &egui::Context, current_state: AppState) {
        if self.lifecycle_state == SessionLifecycleState::RestoringHistory {
            if self.last_committed_state.as_ref() == Some(&current_state) {
                self.quiescent_frames += 1;
                if self.quiescent_frames >= 2 {
                    self.lifecycle_state = SessionLifecycleState::Idle;
                    self.quiescent_frames = 0;
                }
            } else {
                self.last_committed_state = Some(current_state);
                self.quiescent_frames = 0;
            }
            return;
        }

        if let Some(ref last_state) = self.last_committed_state {
            if &current_state != last_state {
                let pointer_down = ctx.input(|i| i.pointer.any_down());
                if pointer_down {
                    if self.lifecycle_state != SessionLifecycleState::DragEditing {
                        self.push_undo(last_state.clone());
                        self.lifecycle_state = SessionLifecycleState::DragEditing;
                    }
                } else {
                    if self.lifecycle_state != SessionLifecycleState::DragEditing {
                        self.push_undo(last_state.clone());
                    } else {
                        self.lifecycle_state = SessionLifecycleState::Idle;
                    }
                }
                self.last_committed_state = Some(current_state);
            } else if !ctx.input(|i| i.pointer.any_down())
                && self.lifecycle_state == SessionLifecycleState::DragEditing
            {
                self.lifecycle_state = SessionLifecycleState::Idle;
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
            callout_manager: self.callout_manager.clone(),
        };
        serde_json::to_string_pretty(&envelope).unwrap_or_default()
    }

    /// Sets the active callout manager for workspace session persistence.
    #[cfg(test)]
    pub fn set_callout_manager(&mut self, manager: Option<crate::framework::CalloutManager>) {
        self.callout_manager = manager;
    }

    /// Returns a reference to the active callout manager if set.
    #[cfg(test)]
    pub fn callout_manager(&self) -> Option<&crate::framework::CalloutManager> {
        self.callout_manager.as_ref()
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
            self.set_is_restoring(true);
            Ok(envelope.app_state)
        } else {
            self.undo_stack.clear();
            self.redo_stack.clear();
            self.callout_manager = None;
            let state: AppState = serde_json::from_str(json_str)
                .map_err(|e| format!("Corrupted workspace state JSON: {}", e))?;
            self.last_committed_state = Some(state.clone());
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
        io::spawn_open_dialog(tx);
    }

    /// Triggers a non-blocking background file dialog for saving a workspace session file to a new path.
    pub fn trigger_save_as(&mut self, envelope_json: String) {
        if self.is_dialog_busy() {
            return;
        }
        self.lifecycle_state = SessionLifecycleState::AwaitingFileDialog;
        let (tx, rx) = channel();
        self.save_as_rx = Some(rx);
        io::spawn_save_as_dialog(tx, envelope_json);
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
            io::spawn_save_direct(tx, path, envelope_json);
        } else {
            self.trigger_save_as(envelope_json);
        }
    }

    /// Polls pending background file open channel.
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

    /// Polls pending background file save-as channel.
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

    /// Polls pending background direct save channel.
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
            let palette = super::SemanticPalette::from_visuals(ui.visuals());
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
