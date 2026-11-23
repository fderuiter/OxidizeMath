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

    /// Captures the O(1) shareable URL signature for the active focused simulation pane (plus active pipeline DAG).
    #[must_use]
    pub fn current_url_signature(&self) -> UrlStateSignature {
        let mut base = if let Some((selected_tab, tab)) = self.active_share_tab() {
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
        };
        if self.viewport_layout != ViewportLayout::Single && !self.pipeline_dag.is_empty() {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            base.state_revision.hash(&mut h);
            std::mem::discriminant(&self.viewport_layout).hash(&mut h);
            self.active_focused_pane().hash(&mut h);
            self.pipeline_dag.links().hash(&mut h);
            for p in 0..self.viewport_layout.pane_count() {
                p.hash(&mut h);
                self.pane_tabs.get(p).copied().unwrap_or(0).hash(&mut h);
                if let Some(tab) = self.pane_tab(p) {
                    tab.url_state_revision().hash(&mut h);
                }
            }
            base.state_revision = h.finish().max(1);
        }
        base
    }

    fn export_pane_for_pipeline_url(
        &self,
        pane_idx: usize,
    ) -> Result<crate::url_state::pipeline::PipelinePaneUrlEntry, String> {
        use crate::url_state::pipeline::build_pane_url_entry;
        if let Some(tab) = self.pane_tab(pane_idx) {
            let json = tab.export_url_preset_json()?;
            return build_pane_url_entry(tab.url_slug(), tab.active_tool_id(), json.as_deref());
        }
        let tab_idx = self
            .pane_tabs
            .get(pane_idx)
            .copied()
            .ok_or_else(|| format!("Missing tab assignment for pane {pane_idx}"))?;
        let mut temp = crate::tabs::instantiate_tab(tab_idx)
            .ok_or_else(|| format!("Unknown tab index {tab_idx} for pane {pane_idx}"))?;
        if let Some(saved) = self
            .pane_slots
            .slot(pane_idx)
            .and_then(|s| s.saved_states.get(temp.name()))
        {
            temp.load_state(saved);
        }
        let json = temp.export_url_preset_json()?;
        build_pane_url_entry(temp.url_slug(), temp.active_tool_id(), json.as_deref())
    }

    /// Encodes the active focused simulation pane's state (and active multi-pane pipeline when linked) as `#tab=...`.
    pub fn encode_current_share_fragment(&self) -> Result<String, String> {
        let (_, tab) = self
            .active_share_tab()
            .ok_or_else(|| "No active simulation tab selected".to_string())?;
        if self.viewport_layout != ViewportLayout::Single && !self.pipeline_dag.is_empty() {
            let pane_count = self.viewport_layout.pane_count();
            let mut panes = Vec::with_capacity(pane_count);
            for p in 0..pane_count {
                panes.push(self.export_pane_for_pipeline_url(p)?);
            }
            let env = crate::url_state::pipeline::PipelineUrlEnvelope {
                layout: self.viewport_layout,
                focused_pane: self.active_focused_pane(),
                panes,
                links: self.pipeline_dag.links().to_vec(),
            };
            return crate::url_state::pipeline::encode_pipeline_share_fragment(
                tab.url_slug(),
                tab.active_tool_id(),
                env,
            );
        }
        tab.encode_url_share_fragment()
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
        if let Some(pipe) = params.pipeline {
            return self.hydrate_pipeline_share_url(pipe, raw_hash, ctx);
        }
        let Some(tab_idx) = self.tabs.iter().position(|tab| {
            crate::url_state::matches_tab_slug(&params.tab, tab.url_slug(), tab.name())
        }) else {
            let err = format!("Unknown simulation tab '{}' in shared URL", params.tab);
            self.record_invalid_share_url(err.clone(), ctx);
            return Err(err);
        };

        if let Err(err) = self.tabs[tab_idx].import_url_share_params(&params, ctx) {
            self.record_invalid_share_url(err.clone(), ctx);
            return Err(err);
        }

        let tab_name = self.tabs[tab_idx].name();
        self.tab_state_cache.invalidate_tab(tab_name);
        self.selected_tab = tab_idx;
        self.focused_pane = 0;
        self.set_layout(ViewportLayout::Single);
        self.session_manager.clear_toast();
        self.finish_hydrated_url_sync(raw_hash);
        Ok(true)
    }

    fn hydrate_pipeline_share_url(
        &mut self,
        pipe: crate::url_state::pipeline::ValidatedPipelineUrl,
        raw_hash: &str,
        ctx: Option<&egui::Context>,
    ) -> Result<bool, String> {
        let pane_count = pipe.layout.pane_count();
        let mut built: Vec<(usize, Box<dyn crate::tabs::ExplorerTab>)> =
            Vec::with_capacity(pane_count);
        for desc in &pipe.panes {
            let Some(tab_idx) = self.tabs.iter().position(|tab| {
                crate::url_state::matches_tab_slug(&desc.tab, tab.url_slug(), tab.name())
            }) else {
                let err = format!("Unknown simulation tab '{}' in pipeline URL", desc.tab);
                self.record_invalid_share_url(err.clone(), ctx);
                return Err(err);
            };
            let Some(mut instance) = crate::tabs::instantiate_tab(tab_idx) else {
                let err = format!("Failed to instantiate tab '{}' for pipeline URL", desc.tab);
                self.record_invalid_share_url(err.clone(), ctx);
                return Err(err);
            };
            if let Err(err) =
                instance.import_url_preset_json(desc.tool.as_deref(), desc.state_json.as_deref())
            {
                self.record_invalid_share_url(err.clone(), ctx);
                return Err(err);
            }
            built.push((tab_idx, instance));
        }
        for link in &pipe.links {
            let src_tool = built[link.from_pane].1.active_tool_id();
            let dst_tool = built[link.to_pane].1.active_tool_id();
            if super::dataflow_bus::compatible_stream_kind(src_tool, dst_tool) != Some(link.kind) {
                let err = "Incompatible tool stream link in shared pipeline URL".to_string();
                self.record_invalid_share_url(err.clone(), ctx);
                return Err(err);
            }
        }

        self.apply_validated_pipeline_instances(pipe.layout, pipe.focused_pane, built, &pipe.links);
        self.session_manager.clear_toast();
        self.finish_hydrated_url_sync(raw_hash);
        Ok(true)
    }

    fn apply_validated_pipeline_instances(
        &mut self,
        layout: ViewportLayout,
        focused_pane: usize,
        built: Vec<(usize, Box<dyn crate::tabs::ExplorerTab>)>,
        links: &[super::dataflow_bus::PaneLink],
    ) {
        while self.pane_tabs.len() < 4 {
            self.pane_tabs.push(0);
        }
        let mut iter = built.into_iter().enumerate();
        if let Some((_, (tab0_idx, tab0))) = iter.next() {
            let name0 = tab0.name().to_string();
            self.tabs[tab0_idx] = tab0;
            self.selected_tab = tab0_idx;
            self.pane_tabs[0] = tab0_idx;
            self.tab_state_cache.invalidate_tab(&name0);
        }
        for (pane_idx, (tab_idx, tab_box)) in iter {
            self.pane_tabs[pane_idx] = tab_idx;
            if pane_idx == 1 {
                self.secondary_tab = tab_idx;
            }
            if let Some(slot) = self.pane_slots.slot_mut(pane_idx) {
                slot.initialized = true;
                if let Some(st) = tab_box.save_state() {
                    slot.saved_states
                        .insert(tab_box.name().to_string(), st.into());
                }
                slot.active = Some((tab_idx, tab_box));
                slot.clear_active_cache();
            }
        }
        self.viewport_layout = layout;
        self.sync_layout();
        self.prune_inactive_pane_slots(layout.pane_count());
        self.focused_pane = focused_pane.min(layout.pane_count().saturating_sub(1));
        let _ = self.pipeline_dag.restore_links(links, layout.pane_count());
        self.prune_incompatible_pipeline_links();
    }

    fn finish_hydrated_url_sync(&mut self, raw_hash: &str) {
        let canonical_frag = self.encode_current_share_fragment().ok();
        #[cfg(target_arch = "wasm32")]
        if raw_hash.trim_start_matches('#').trim().is_empty() {
            if let Some(ref frag) = canonical_frag {
                let _ = crate::url_state::replace_browser_url_fragment(frag);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = raw_hash;

        self.url_sync
            .mark_synced(self.current_url_signature(), canonical_frag);
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
        let allowed = !self.has_active_modal_or_popup(ctx);
        let shortcut_triggered = super::global_shortcuts::register_and_check_shortcut_if_allowed(
            ctx,
            super::global_shortcuts::DesktopShortcutSpec {
                name: "Copy Share Link",
                description: "Copy a shareable simulation URL for the active view",
                trigger: egui_plot::commands::CommandTrigger::Shortcut(modifiers, egui::Key::L),
                context_name: "Global",
            },
            None,
            allowed,
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
