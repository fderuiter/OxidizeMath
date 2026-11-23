use super::{SessionLifecycleState, WorkspaceSessionManager};
use crate::app::{AppState, MathExplorerApp};
use crate::framework::CalloutManager;
use eframe::egui;
use std::path::PathBuf;

/// Complete in-memory workspace snapshot captured upon WebGL context loss so an in-place
/// `WebRunner` restart (or blocked-storage `.mws` backup download) preserves active tool
/// parameters, multi-pane layouts, undo/redo stacks, callouts, and active file metadata.
#[derive(Clone, Debug)]
pub(crate) struct WorkspaceRecoverySnapshot {
    pub app_state: AppState,
    pub undo_stack: Vec<AppState>,
    pub redo_stack: Vec<AppState>,
    pub callout_manager: Option<CalloutManager>,
    pub active_file_path: Option<PathBuf>,
    pub envelope_json: String,
}

impl WorkspaceRecoverySnapshot {
    /// Returns the preferred `.mws` download filename derived from `active_file_path` or a default.
    #[must_use]
    pub fn recovery_filename(&self) -> String {
        let raw = self
            .active_file_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("math-explorer-recovery.mws");
        crate::web_io::sanitize_download_filename(raw, "math-explorer-recovery.mws")
    }
}

impl WorkspaceSessionManager {
    fn finalize_drag_before_snapshot(&mut self, current_state: &AppState) {
        if self.lifecycle_state == SessionLifecycleState::DragEditing {
            if let Some(ref last_state) = self.last_committed_state {
                if current_state != last_state {
                    self.push_undo(last_state.clone());
                }
            }
            self.lifecycle_state = SessionLifecycleState::Idle;
        }
        self.last_committed_state = Some(current_state.clone());
        self.last_committed_fingerprint = None;
    }

    fn apply_recovery_snapshot(&mut self, snapshot: &WorkspaceRecoverySnapshot) {
        self.undo_stack = snapshot.undo_stack.clone();
        self.redo_stack = snapshot.redo_stack.clone();
        self.callout_manager = snapshot.callout_manager.clone();
        self.set_active_file_path(snapshot.active_file_path.clone());
        self.last_committed_state = Some(snapshot.app_state.clone());
        self.set_is_restoring(true);
    }
}

impl MathExplorerApp {
    /// Flushes active pane caches, finalizes any continuous drag edit, and extracts a full
    /// [`WorkspaceRecoverySnapshot`] plus serialized `.mws` [`super::WorkspaceEnvelope`] JSON.
    pub(crate) fn capture_recovery_snapshot(&mut self) -> WorkspaceRecoverySnapshot {
        self.clear_active_pane_caches();
        self.is_dragging_divider = false;
        self.dragging_quad_axes = (false, false);

        let app_state = self.get_app_state();
        self.session_manager
            .finalize_drag_before_snapshot(&app_state);
        let envelope_json = self.session_manager.create_envelope(app_state.clone());

        WorkspaceRecoverySnapshot {
            app_state,
            undo_stack: self.session_manager.undo_stack.clone(),
            redo_stack: self.session_manager.redo_stack.clone(),
            callout_manager: self.session_manager.callout_manager.clone(),
            active_file_path: self.session_manager.active_file_path().cloned(),
            envelope_json,
        }
    }

    /// Restores a previously captured [`WorkspaceRecoverySnapshot`] onto a freshly constructed
    /// [`MathExplorerApp`], preserving undo/redo history, multi-pane layout, theme sync, and URL sync.
    pub(crate) fn restore_recovery_snapshot(
        &mut self,
        snapshot: &WorkspaceRecoverySnapshot,
        ctx: &egui::Context,
    ) {
        self.is_dragging_divider = false;
        self.dragging_quad_axes = (false, false);
        self.apply_app_state(snapshot.app_state.clone());
        self.session_manager.apply_recovery_snapshot(snapshot);

        self.theme.apply_to_context(ctx);
        crate::app::theme::sync_document_theme(self.theme);
        self.last_synced_theme = Some(self.theme);

        let canonical_frag = self.encode_current_share_fragment().ok();
        self.url_sync
            .mark_synced(self.current_url_signature(), canonical_frag);
        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use crate::app::MathExplorerApp;
    use eframe::egui;

    #[cfg(feature = "pure_math")]
    #[test]
    fn test_capture_and_restore_recovery_snapshot_preserves_multi_pane_and_undo_history() {
        use crate::app::session::SessionLifecycleState;
        use crate::app::{ThemeMode, ViewportLayout};
        use std::path::PathBuf;
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let mut source_app = MathExplorerApp::default();
        let initial_state = source_app.get_app_state();
        source_app.session_manager.push_undo(initial_state.clone());

        source_app.theme = ThemeMode::HighContrast;
        source_app.set_layout(ViewportLayout::SplitHorizontal);
        source_app.set_pane_tab(0, 2);
        source_app.set_pane_tab(1, 3);
        source_app.split_ratio = 0.62;
        source_app.focused_pane = 1;
        source_app
            .session_manager
            .set_active_file_path(Some(PathBuf::from("/tmp/ising_study.mws")));

        let mid_state = source_app.get_app_state();
        source_app.session_manager.last_committed_state = Some(mid_state.clone());
        source_app.session_manager.lifecycle_state = SessionLifecycleState::DragEditing;
        source_app.split_ratio = 0.38;

        let snapshot = source_app.capture_recovery_snapshot();
        assert_eq!(snapshot.recovery_filename(), "ising_study.mws");
        assert_eq!(
            source_app.session_manager.lifecycle_state(),
            SessionLifecycleState::Idle
        );
        assert_eq!(snapshot.undo_stack.len(), 2);
        assert!(snapshot
            .envelope_json
            .contains("\"schema_version\": \"1.0\""));
        assert!(snapshot.envelope_json.contains("\"history_undo_stack\""));

        let fresh_ctx = egui::Context::default();
        let mut restored_app = MathExplorerApp::default();
        restored_app.restore_recovery_snapshot(&snapshot, &fresh_ctx);

        assert_eq!(restored_app.theme, ThemeMode::HighContrast);
        assert_eq!(
            restored_app.viewport_layout,
            ViewportLayout::SplitHorizontal
        );
        assert_eq!(restored_app.pane_tabs[0], 2);
        assert_eq!(restored_app.pane_tabs[1], 3);
        assert!((restored_app.split_ratio - 0.38).abs() < 1e-6);
        assert_eq!(restored_app.focused_pane, 1);
        assert_eq!(
            restored_app.session_manager.active_file_path(),
            Some(&PathBuf::from("/tmp/ising_study.mws"))
        );
        assert!(restored_app.session_manager.can_undo());

        restored_app.undo();
        assert!((restored_app.split_ratio - 0.62).abs() < 1e-6);
        assert!(restored_app.session_manager.can_redo());

        restored_app.redo();
        assert!((restored_app.split_ratio - 0.38).abs() < 1e-6);
    }

    #[test]
    fn test_fresh_app_rebuilds_clean_context_state_and_texture_handles() {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let old_ctx = egui::Context::default();
        let mut old_app = MathExplorerApp::default();
        let solid_state_idx = old_app
            .tabs
            .iter()
            .position(|t| t.url_slug() == "solid-state")
            .unwrap_or(0);
        old_app.select_tab(&old_ctx, solid_state_idx);
        let snapshot = old_app.capture_recovery_snapshot();
        assert_eq!(snapshot.recovery_filename(), "math-explorer-recovery.mws");
        assert!(snapshot.envelope_json.contains("\"schema_version\""));

        let restored_ctx = egui::Context::default();
        let mut restored_app = MathExplorerApp::default();
        restored_app.restore_recovery_snapshot(&snapshot, &restored_ctx);
        assert_eq!(restored_app.selected_tab, solid_state_idx);

        #[cfg(feature = "physics")]
        {
            use crate::framework::InteractiveTool;
            let mut old_ising = crate::tabs::solid_state::ising::IsingModelTool::default();
            let old_out = old_ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    old_ising.show_inside(ui);
                });
            });
            assert!(
                !old_out.textures_delta.set.is_empty(),
                "initial IsingModelTool frame should upload its lattice TextureHandle"
            );
            let saved_ising_json = old_ising
                .save_state()
                .expect("IsingModelTool should serialize state");

            let mut fresh_ising = crate::tabs::solid_state::ising::IsingModelTool::default();
            fresh_ising
                .try_load_state(&saved_ising_json)
                .expect("fresh IsingModelTool should restore saved state");
            let restored_out = restored_ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    fresh_ising.show_inside(ui);
                });
            });
            assert!(
                !restored_out.textures_delta.set.is_empty(),
                "freshly restored IsingModelTool must allocate a new TextureHandle on the restored Context"
            );
        }
    }
}
