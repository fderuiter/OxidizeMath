use super::io::{
    file_name_str, handle_transition_error, notify, spawn_open_dialog, spawn_save_as_dialog,
    spawn_save_direct,
};
use super::{SessionLifecycleState, WorkspaceEnvelope, WorkspaceSessionManager};
use crate::app::{AppState, MathExplorerApp, SharedTabJson};
use eframe::egui;
use std::sync::mpsc::{channel, TryRecvError};

fn resolve_legacy_callout_target(app_state: &AppState) -> Option<(String, Option<String>)> {
    let tabs = crate::tabs::instantiate_tabs();
    if let Some(raw_name) = app_state
        .selected_tab_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if let Some(tab) = tabs
            .iter()
            .find(|t| t.framework().is_some() && t.name().eq_ignore_ascii_case(raw_name))
        {
            return Some((tab.name().to_string(), None));
        }
        let singular = raw_name.strip_suffix('s').unwrap_or(raw_name);
        for tab in &tabs {
            if let Some(meta) = tab.framework().and_then(|fw| {
                fw.available_tools.iter().find(|m| {
                    m.id.eq_ignore_ascii_case(raw_name)
                        || m.display_name.eq_ignore_ascii_case(raw_name)
                        || m.id.eq_ignore_ascii_case(singular)
                        || m.display_name.eq_ignore_ascii_case(singular)
                })
            }) {
                return Some((tab.name().to_string(), Some(meta.display_name.to_string())));
            }
        }
    }
    if let Some(tab) = tabs
        .get(app_state.selected_tab)
        .filter(|t| t.framework().is_some())
        .or_else(|| tabs.iter().find(|t| t.framework().is_some()))
    {
        return Some((tab.name().to_string(), None));
    }
    app_state
        .selected_tab_name
        .clone()
        .filter(|s| !s.trim().is_empty())
        .map(|name| (name, None))
}

fn migrate_legacy_callout_manager(
    app_state: &mut AppState,
    mgr: &crate::framework::CalloutManager,
) {
    let saved: Vec<egui_plot::SavedCallout> = mgr
        .callouts()
        .iter()
        .filter(|c| {
            c.visible
                && c.target_point[0].is_finite()
                && c.target_point[1].is_finite()
                && c.label_offset[0].is_finite()
                && c.label_offset[1].is_finite()
        })
        .map(|c| egui_plot::SavedCallout::new(c.target_point, c.text.clone(), c.label_offset))
        .collect();
    if saved.is_empty() {
        return;
    }
    let Some((tab_name, tool_name)) = resolve_legacy_callout_target(app_state) else {
        return;
    };
    if tool_name.is_some() {
        app_state.selected_tab_name = Some(tab_name.clone());
    }
    if let Some(merged) = crate::framework::plot_persistence::merge_migrated_callouts_into_tab_json(
        app_state
            .tab_states
            .get(&tab_name)
            .map(SharedTabJson::as_str),
        tool_name.as_deref(),
        &saved,
    ) {
        let merged = SharedTabJson::from(merged);
        app_state
            .tab_states
            .insert(tab_name.clone(), merged.clone());
        if let Some(primary_pane) = app_state.pane_tab_states.first_mut() {
            if primary_pane.contains_key(&tab_name) {
                primary_pane.insert(tab_name, merged);
            }
        }
    }
}

fn validate_envelope_bounds(envelope: &WorkspaceEnvelope) -> Result<(), String> {
    envelope.app_state.data_bus.validate_session_bounds()?;
    for hist in envelope
        .history_undo_stack
        .iter()
        .chain(envelope.history_redo_stack.iter())
    {
        hist.data_bus.validate_session_bounds()?;
    }
    Ok(())
}

impl WorkspaceSessionManager {
    /// Wraps the given application state inside a versioned workspace envelope and serializes it to JSON.
    pub fn create_envelope(&self, app_state: AppState) -> String {
        let envelope = WorkspaceEnvelope {
            schema_version: "1.0".to_string(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            app_state,
            history_undo_stack: self.undo_stack.clone(),
            history_redo_stack: self.redo_stack.clone(),
            callout_manager: None,
        };
        serde_json::to_string_pretty(&envelope).unwrap_or_default()
    }

    /// Validates and parses workspace session content from a JSON string.
    pub fn parse_envelope(&mut self, json_str: &str) -> Result<AppState, String> {
        let cleaned = egui_plot::export::strip_utf8_bom(json_str);
        crate::framework::ToolStateBounds::SESSION.validate_json(cleaned)?;
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
            validate_envelope_bounds(&envelope)?;
            let mut app_state = envelope.app_state;
            if let Some(ref legacy_mgr) = envelope.callout_manager {
                migrate_legacy_callout_manager(&mut app_state, legacy_mgr);
            }
            self.undo_stack = envelope.history_undo_stack;
            self.redo_stack = envelope.history_redo_stack;
            self.last_committed_state = Some(app_state.clone());
            self.last_committed_fingerprint = None;
            self.set_is_restoring(true);
            Ok(app_state)
        } else {
            self.undo_stack.clear();
            self.redo_stack.clear();
            let state: AppState = serde_json::from_str(cleaned)
                .map_err(|e| format!("Corrupted workspace state JSON: {}", e))?;
            state.data_bus.validate_session_bounds()?;
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
