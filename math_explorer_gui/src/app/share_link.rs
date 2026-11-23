//! Application-level URL fragment hydration, 300 ms debounced `replaceState` sync,
//! and `File` menu clipboard share-link controls for [`MathExplorerApp`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MathExplorerApp;
use crate::url_state::{
    install_browser_history_listeners, is_browser_location_self_synced,
    parse_workspace_or_share_location, read_browser_location_parts, UrlStateSignature,
    URL_SYNC_DEBOUNCE_MS,
};
use eframe::egui;
use federated_registry::Severity;
use std::time::Duration;

const APP_SHARE_CLIPBOARD_ID: &str = "app_share_clipboard_status";

impl MathExplorerApp {
    /// Captures the O(1) shareable URL signature for the active single-pane or multi-pane workspace.
    #[must_use]
    pub fn current_url_signature(&self) -> UrlStateSignature {
        self.capture_workspace_url_signature()
    }

    /// Encodes the active workspace (`#tab=...` for <= 4 KiB single panes, `#ws=...` for multi-pane
    /// or larger single-pane states up to 32 KiB).
    pub fn encode_current_share_fragment(&self) -> Result<String, String> {
        self.encode_workspace_or_single_fragment()
    }

    /// Encodes a full shareable web URL (`https://...#tab=...` or `https://...#ws=...`) for the workspace.
    pub fn encode_current_share_url(&self) -> Result<String, String> {
        let fragment = self.encode_current_share_fragment()?;
        Ok(crate::url_state::build_share_url_from_fragment(&fragment))
    }

    /// Copies a shareable web URL for the active workspace to the clipboard.
    pub fn copy_current_share_link(&mut self, ctx: &egui::Context) {
        match self.encode_current_share_url() {
            Ok(url) => {
                let status_id = egui::Id::new(APP_SHARE_CLIPBOARD_ID);
                let ctx_cb = ctx.clone();
                crate::url_state::copy_share_url_to_clipboard(ctx, url, move |res| {
                    let entry = match res {
                        Ok(()) => (false, String::new()),
                        Err(err) => (true, err),
                    };
                    ctx_cb.data_mut(|d| d.insert_temp(status_id, entry));
                });
                self.poll_share_clipboard_status(ctx);
            }
            Err(err) => {
                let msg = format!("Failed to copy share link: {err}");
                self.session_manager.set_toast(msg.clone(), true);
                crate::accessibility::announce_live_assertive_status(ctx, &msg);
            }
        }
    }

    pub(super) fn poll_share_clipboard_status(&mut self, ctx: &egui::Context) {
        let status_id = egui::Id::new(APP_SHARE_CLIPBOARD_ID);
        let Some((is_err, err)) = ctx.data_mut(|d| d.remove_temp::<(bool, String)>(status_id))
        else {
            return;
        };
        if is_err {
            let msg = format!("Failed to copy share link: {err}");
            self.session_manager.set_toast(msg.clone(), true);
            crate::accessibility::announce_live_assertive_status(ctx, &msg);
        } else {
            let msg = "Copied shareable simulation link to clipboard";
            self.session_manager.set_toast(msg, false);
            crate::accessibility::announce_live_status(ctx, msg);
        }
    }

    /// Hydrates from `window.location.hash` / `window.location.search` on startup:
    /// - Keeps the `WorkspaceEnvelope` + undo stack restored from `localStorage` when `History.state`
    ///   proves the current URL was self-synced by this tab (across reloads and WebGL recovery).
    /// - Pushes the restored workspace onto `undo_stack` before applying an external share URL.
    pub(crate) fn hydrate_from_browser_url(
        &mut self,
        ctx: &egui::Context,
        restored_from_storage: bool,
    ) {
        install_browser_history_listeners(ctx);
        if let Some((hash, search)) = read_browser_location_parts() {
            self.hydrate_startup_location_strings(&hash, &search, Some(ctx), restored_from_storage);
        }
    }

    pub(crate) fn hydrate_startup_location_strings(
        &mut self,
        hash: &str,
        search: &str,
        ctx: Option<&egui::Context>,
        restored_from_storage: bool,
    ) {
        if restored_from_storage && is_browser_location_self_synced(hash, search) {
            self.url_sync
                .mark_synced(self.current_url_signature(), Some(hash.to_string()));
            return;
        }
        let prev_state = restored_from_storage.then(|| self.get_app_state());
        let parsed = match parse_workspace_or_share_location(hash, search) {
            Ok(Some(p)) => p,
            Ok(None) => return,
            Err(err) => {
                self.record_invalid_share_url(err, ctx);
                return;
            }
        };
        let _ = self.hydrate_parsed_location(parsed, ctx, prev_state);
    }

    /// Parses and applies `hash` (with `search` query-parameter fallback) to the workspace,
    /// returning `Ok(true)` if a shared simulation/workspace link was hydrated, `Ok(false)` if no
    /// share parameters were present, or `Err(reason)` after safely recording warnings on invalid input.
    #[cfg(test)]
    pub fn apply_share_location_strings(
        &mut self,
        hash: &str,
        search: &str,
        ctx: Option<&egui::Context>,
    ) -> Result<bool, String> {
        let parsed = match parse_workspace_or_share_location(hash, search) {
            Ok(Some(params)) => params,
            Ok(None) => return Ok(false),
            Err(err) => {
                self.record_invalid_share_url(err.clone(), ctx);
                return Err(err);
            }
        };
        self.hydrate_parsed_location(parsed, ctx, None)
    }

    pub(super) fn record_invalid_share_url(&mut self, err: String, ctx: Option<&egui::Context>) {
        self.url_sync
            .suppress_initial_overwrite(self.current_url_signature());
        let warning = format!("Invalid shared simulation link ignored: {err}");
        self.session_manager.set_toast(warning.clone(), true);
        crate::telemetry::emit_telemetry(
            "UrlState",
            Severity::Warning,
            warning.clone(),
            std::collections::HashMap::new(),
        );
        if let Some(egui_ctx) = ctx {
            crate::accessibility::announce_live_assertive_status(egui_ctx, &warning);
        }
        crate::accessibility::announce_status_with_priority(&warning, "assertive");
    }

    /// Returns the most recently flushed URL fragment in tests.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn url_sync_last_fragment_for_test(&self) -> Option<&str> {
        self.url_sync.last_synced_fragment()
    }

    /// Drains queued `hashchange`/`popstate` navigations, registers `Ctrl+Shift+L` / `Cmd+Shift+L`
    /// in `CMD_REGISTRY`, and drains clipboard callbacks prior to rendering menus and viewports.
    pub(crate) fn poll_share_shortcut_and_clipboard(&mut self, ctx: &egui::Context) {
        self.drain_browser_url_navigations(ctx);
        self.poll_share_clipboard_status(ctx);

        let modifiers = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD | egui::Modifiers::SHIFT
        } else {
            egui::Modifiers::CTRL | egui::Modifiers::SHIFT
        };
        let shortcut_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Copy Share Link",
            "Copy a shareable simulation URL for the active workspace",
            egui_plot::commands::CommandTrigger::Shortcut(modifiers, egui::Key::L),
            true,
            "Global",
            None,
            None,
        );
        if shortcut_triggered {
            self.copy_current_share_link(ctx);
        }
    }

    /// Polls end-of-frame clipboard status and 300 ms debounced `History::replaceState` sync.
    pub(crate) fn tick_url_sync(&mut self, ctx: &egui::Context) {
        self.poll_share_clipboard_status(ctx);

        let pointer_or_sweep_active = ctx.input(|i| i.pointer.any_down())
            || self.is_dragging_divider
            || self.dragging_quad_axes != (false, false)
            || crate::reflective_ui::slider::has_active_sweeps(ctx);
        let now = crate::web_io::PlatformInstant::now();
        self.poll_url_sync_at(ctx, pointer_or_sweep_active, now);
    }

    /// Advances the debounced URL synchronization state machine at `now`.
    pub fn poll_url_sync_at(
        &mut self,
        ctx: &egui::Context,
        pointer_down: bool,
        now: crate::web_io::PlatformInstant,
    ) {
        let mut sync = std::mem::take(&mut self.url_sync);
        let sig = self.current_url_signature();
        let remaining = sync.poll_with_now(
            sig,
            pointer_down,
            now,
            Duration::from_millis(URL_SYNC_DEBOUNCE_MS),
            || self.encode_current_share_fragment(),
        );
        self.url_sync = sync;
        if let Some(wait) = remaining {
            ctx.request_repaint_after(wait.max(Duration::from_millis(16)));
        }
    }
}

/// Renders the `🔗 Copy Share Link` entry inside the non-macOS `File` menu.
#[cfg(any(not(target_os = "macos"), test))]
pub(super) fn render_file_share_menu_item(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
) {
    let shortcut = if cfg!(target_os = "macos") {
        "Cmd+Shift+L"
    } else {
        "Ctrl+Shift+L"
    };
    if ui
        .add(egui::Button::new("🔗 Copy Share Link").shortcut_text(shortcut))
        .clicked()
    {
        app.copy_current_share_link(ctx);
        ui.close();
    }
}

// theory_verification!
