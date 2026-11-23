mod io;
mod persistence;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod recovery;
pub(crate) mod shortcuts;
mod types;

#[cfg(any(target_arch = "wasm32", test))]
#[allow(unused_imports)]
pub(crate) use recovery::WorkspaceRecoverySnapshot;
pub use types::{SessionLifecycleState, WorkspaceEnvelope};

use crate::app::state::AppStateFingerprint;
use crate::app::{AppState, MathExplorerApp};
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

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
    pub(super) undo_stack: Vec<AppState>,
    /// Redo history stack.
    pub(super) redo_stack: Vec<AppState>,
    /// Maximum depth of history stacks.
    max_history_depth: usize,
    /// Last committed state snapshot for diff tracking.
    pub(super) last_committed_state: Option<AppState>,
    /// Fast O(1) revision fingerprint of `last_committed_state`.
    pub(super) last_committed_fingerprint: Option<AppStateFingerprint>,
    /// Explicit session lifecycle state.
    pub(crate) lifecycle_state: SessionLifecycleState,
    /// Number of consecutive quiescent frames observed during history restoration.
    pub(crate) quiescent_frames: usize,
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
            last_committed_fingerprint: None,
            lifecycle_state: SessionLifecycleState::Idle,
            quiescent_frames: 0,
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

    #[cfg(test)]
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
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

    /// Records `prev_state` onto `undo_stack` prior to an external URL navigation override
    /// (including during startup `RestoringHistory`), clearing `redo_stack` and committing `new_state`.
    pub fn record_external_url_override(&mut self, prev_state: AppState, new_state: AppState) {
        if self.lifecycle_state == SessionLifecycleState::DragEditing {
            if let Some(ref last_state) = self.last_committed_state {
                if last_state != &prev_state {
                    self.push_undo_preserving_redo(last_state.clone());
                }
            }
        }
        if prev_state != new_state {
            self.push_undo_preserving_redo(prev_state);
            self.redo_stack.clear();
        }
        self.last_committed_state = Some(new_state);
        self.set_is_restoring(true);
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
            self.last_committed_fingerprint = None;
        } else if self.lifecycle_state == SessionLifecycleState::RestoringHistory {
            self.lifecycle_state = SessionLifecycleState::Idle;
            self.quiescent_frames = 0;
        }
    }

    /// Tracks per-frame workspace changes lazily from `MathExplorerApp`:
    /// - Skips full `AppState::extract` on clean idle frames when revision fingerprints match.
    /// - Skips full `AppState::extract` during active `DragEditing` frames and commits once on pointer release.
    pub fn track_frame_changes(&mut self, ctx: &egui::Context, app: &MathExplorerApp) {
        let fingerprint = AppStateFingerprint::capture(app);
        if self.lifecycle_state == SessionLifecycleState::RestoringHistory
            || self.last_committed_state.is_none()
        {
            let current_state = app.get_app_state();
            self.track_state_changes(ctx, current_state);
            self.last_committed_fingerprint = Some(fingerprint);
            return;
        }

        let pointer_down = ctx.input(|i| i.pointer.any_down());
        let sweep_active = crate::reflective_ui::slider::has_active_sweeps(ctx);
        let continuous_edit = pointer_down || sweep_active;
        if self.lifecycle_state == SessionLifecycleState::DragEditing {
            if !continuous_edit {
                app.clear_active_pane_caches();
                let current_state = app.get_app_state();
                if let Some(ref last_state) = self.last_committed_state {
                    if &current_state != last_state {
                        self.push_undo(last_state.clone());
                    }
                }
                self.last_committed_state = Some(current_state);
                self.last_committed_fingerprint = Some(AppStateFingerprint::capture(app));
                self.lifecycle_state = SessionLifecycleState::Idle;
            }
            return;
        }

        if fingerprint.all_revisions_tracked && self.last_committed_fingerprint == Some(fingerprint)
        {
            if sweep_active && self.last_committed_state.is_some() {
                self.lifecycle_state = SessionLifecycleState::DragEditing;
            }
            return;
        }

        if continuous_edit && self.last_committed_state.is_some() {
            self.lifecycle_state = SessionLifecycleState::DragEditing;
            return;
        }

        let current_state = app.get_app_state();
        self.track_state_changes(ctx, current_state);
        self.last_committed_fingerprint = Some(fingerprint);
    }

    /// Executes an undo step directly on this session manager and restores `app` state.
    pub fn execute_undo(&mut self, app: &mut MathExplorerApp) {
        if self.lifecycle_state == SessionLifecycleState::DragEditing {
            app.clear_active_pane_caches();
            let current = app.get_app_state();
            if let Some(prev) = self.last_committed_state.clone() {
                if prev != current {
                    self.push_redo(current);
                    app.apply_app_state(prev);
                    self.set_is_restoring(true);
                    return;
                }
            }
        }
        let Some(prev) = self.pop_undo() else {
            return;
        };
        let current = app.get_app_state();
        self.push_redo(current);
        app.apply_app_state(prev);
        self.set_is_restoring(true);
    }

    /// Executes a redo step directly on this session manager and restores `app` state.
    pub fn execute_redo(&mut self, app: &mut MathExplorerApp) {
        let Some(next) = self.pop_redo() else {
            return;
        };
        if self.lifecycle_state == SessionLifecycleState::DragEditing {
            app.clear_active_pane_caches();
        }
        let current = app.get_app_state();
        self.push_undo_preserving_redo(current);
        app.apply_app_state(next);
        self.set_is_restoring(true);
    }

    /// Tracks application state changes and updates the history buffer.
    pub fn track_state_changes(&mut self, ctx: &egui::Context, current_state: AppState) {
        self.last_committed_fingerprint = None;
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

        let continuous_edit = ctx.input(|i| i.pointer.any_down())
            || crate::reflective_ui::slider::has_active_sweeps(ctx);
        if let Some(ref last_state) = self.last_committed_state {
            if &current_state != last_state {
                if continuous_edit {
                    if self.lifecycle_state != SessionLifecycleState::DragEditing {
                        self.push_undo(last_state.clone());
                        self.lifecycle_state = SessionLifecycleState::DragEditing;
                    }
                } else if self.lifecycle_state != SessionLifecycleState::DragEditing {
                    self.push_undo(last_state.clone());
                } else {
                    self.lifecycle_state = SessionLifecycleState::Idle;
                }
                self.last_committed_state = Some(current_state);
            } else if !continuous_edit && self.lifecycle_state == SessionLifecycleState::DragEditing
            {
                self.lifecycle_state = SessionLifecycleState::Idle;
            }
        } else {
            self.last_committed_state = Some(current_state);
        }
    }
}
