//! Application-level URL fragment hydration, 300 ms debounced `replaceState` sync,
//! and `File` menu clipboard share-link controls for [`MathExplorerApp`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MathExplorerApp, ViewportLayout};
use crate::url_state::{ShareUrlParams, UrlStateSignature, URL_SYNC_DEBOUNCE_MS};
use eframe::egui;
use federated_registry::Severity;
use std::time::Duration;

const APP_SHARE_CLIPBOARD_ID: &str = "app_share_clipboard_status";

impl MathExplorerApp {
    fn active_share_tab(&self) -> Option<(usize, &dyn crate::tabs::ExplorerTab)> {
        let pane = self.active_focused_pane();
        let tab_idx = self.focused_tab_index();
        let tab = self
            .pane_tab(pane)
            .or_else(|| self.tabs.get(tab_idx).map(Box::as_ref))?;
        Some((tab_idx, tab))
    }

    /// Captures the O(1) shareable URL signature for the active focused simulation pane.
    #[must_use]
    pub fn current_url_signature(&self) -> UrlStateSignature {
        if let Some((selected_tab, tab)) = self.active_share_tab() {
            UrlStateSignature {
                selected_tab,
                active_tool_id: tab.active_tool_id(),
                state_revision: tab.url_state_revision(),
            }
        } else {
            UrlStateSignature {
                selected_tab: self.selected_tab,
                active_tool_id: None,
                state_revision: 0,
            }
        }
    }

    /// Encodes the active focused simulation pane's state as a `#tab=...` URL fragment.
    pub fn encode_current_share_fragment(&self) -> Result<String, String> {
        let (_, tab) = self
            .active_share_tab()
            .ok_or_else(|| "No active simulation tab selected".to_string())?;
        let preset_json = tab.export_url_preset_json()?;
        crate::url_state::encode_share_fragment_with_challenge(
            tab.url_slug(),
            tab.active_tool_id(),
            tab.active_challenge_id(),
            preset_json.as_deref(),
        )
    }

    /// Encodes a full shareable web URL (`https://...#tab=...`) for the primary simulation tab.
    pub fn encode_current_share_url(&self) -> Result<String, String> {
        let fragment = self.encode_current_share_fragment()?;
        Ok(crate::url_state::build_share_url_from_fragment(&fragment))
    }

    /// Copies a shareable web simulation link for the active primary tab to the clipboard.
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

    /// Reads `window.location.hash` / `window.location.search` on startup after `cc.storage`
    /// restoration so shared links override the active view while preserving other cached tabs.
    pub(super) fn hydrate_from_browser_url(&mut self, ctx: &egui::Context) {
        if let Some((hash, search)) = crate::url_state::read_browser_location_parts() {
            let _ = self.apply_share_location_strings(&hash, &search, Some(ctx));
        }
    }

    /// Parses and applies `hash` (with `search` query-parameter fallback) to the primary view,
    /// returning `Ok(true)` if a shared simulation link was hydrated, `Ok(false)` if no share
    /// parameters were present, or `Err(reason)` after safely recording warnings on invalid input.
    pub fn apply_share_location_strings(
        &mut self,
        hash: &str,
        search: &str,
        ctx: Option<&egui::Context>,
    ) -> Result<bool, String> {
        let parsed = match crate::url_state::parse_share_location(hash, search) {
            Ok(Some(params)) => params,
            Ok(None) => return Ok(false),
            Err(err) => {
                self.record_invalid_share_url(err.clone(), ctx);
                return Err(err);
            }
        };
        self.hydrate_parsed_share_params(parsed, hash, ctx)
    }

    fn hydrate_parsed_share_params(
        &mut self,
        params: ShareUrlParams,
        raw_hash: &str,
        ctx: Option<&egui::Context>,
    ) -> Result<bool, String> {
        let Some(tab_idx) = self.tabs.iter().position(|tab| {
            crate::url_state::matches_tab_slug(&params.tab, tab.url_slug(), tab.name())
        }) else {
            let err = format!("Unknown simulation tab '{}' in shared URL", params.tab);
            self.record_invalid_share_url(err.clone(), ctx);
            return Err(err);
        };

        if let Err(err) = self.tabs[tab_idx].import_url_share_params(
            params.tool.as_deref(),
            params.challenge.as_deref(),
            params.state_json.as_deref(),
        ) {
            self.record_invalid_share_url(err.clone(), ctx);
            return Err(err);
        }

        let tab_name = self.tabs[tab_idx].name();
        self.tab_state_cache.invalidate_tab(tab_name);
        self.selected_tab = tab_idx;
        self.focused_pane = 0;
        self.set_layout(ViewportLayout::Single);
        self.session_manager.clear_toast();

        let canonical_frag = self.encode_current_share_fragment().ok();
        #[cfg(target_arch = "wasm32")]
        // theory_verification!
        if raw_hash.trim_start_matches('#').trim().is_empty() {
            if let Some(ref frag) = canonical_frag {
                let _ = crate::url_state::replace_browser_url_fragment(frag);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = raw_hash;

        self.url_sync
            .mark_synced(self.current_url_signature(), canonical_frag);
        Ok(true)
    }

    fn record_invalid_share_url(&mut self, err: String, ctx: Option<&egui::Context>) {
        self.focused_pane = 0;
        self.set_layout(ViewportLayout::Single);
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

    /// Returns the most recently flushed `#tab=...` fragment in tests.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn url_sync_last_fragment_for_test(&self) -> Option<&str> {
        self.url_sync.last_synced_fragment()
    }

    /// Registers `Ctrl+Shift+L` / `Cmd+Shift+L` in `CMD_REGISTRY` and drains clipboard callbacks
    /// prior to rendering menus, toasts, and the Keyboard Shortcuts modal.
    pub(crate) fn poll_share_shortcut_and_clipboard(&mut self, ctx: &egui::Context) {
        self.poll_share_clipboard_status(ctx);

        let modifiers = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD | egui::Modifiers::SHIFT
        } else {
            egui::Modifiers::CTRL | egui::Modifiers::SHIFT
        };
        let shortcut_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Copy Share Link",
            "Copy a shareable simulation URL for the active view",
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
