//! Multi-pane `#ws=<base64url>` workspace signature capture, encoding, atomic candidate-tab
//! hydration, and same-document `hashchange` / `popstate` navigation handling.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ComparisonSyncConfig, MathExplorerApp, ViewportLayout};
use crate::tabs::ExplorerTab;
use crate::url_state::{
    encode_single_pane_smart_fragment_from_params, encode_workspace_share_fragment,
    matches_tab_slug, parse_workspace_or_share_location, replace_browser_url_fragment,
    PaneUrlSignature, ParsedShareLocation, ShareUrlParams, UrlStateSignature, WorkspaceShareParams,
};
use eframe::egui;
use std::hash::{Hash, Hasher};

impl MathExplorerApp {
    fn pane_url_signature(&self, pane_idx: usize) -> PaneUrlSignature {
        if pane_idx == 0 {
            let tab_idx = self.active_primary_tab_index();
            if let Some(tab) = self.tabs.get(tab_idx) {
                return PaneUrlSignature {
                    tab_idx,
                    active_tool_id: tab.active_tool_id(),
                    state_revision: tab.url_state_revision(),
                };
            }
            return PaneUrlSignature::default();
        }
        let tab_idx = self.pane_tabs.get(pane_idx).copied().unwrap_or(0);
        if let Some(slot) = self.pane_slots.slot(pane_idx) {
            if let Some((active_idx, ref tab)) = slot.active {
                return PaneUrlSignature {
                    tab_idx: active_idx,
                    active_tool_id: tab.active_tool_id(),
                    state_revision: tab.url_state_revision(),
                };
            }
            if let Some(tab_meta) = self.tabs.get(tab_idx) {
                if let Some(saved) = slot.saved_states.get(tab_meta.name()) {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    saved.hash(&mut h);
                    return PaneUrlSignature {
                        tab_idx,
                        active_tool_id: None,
                        state_revision: h.finish().max(1),
                    };
                }
            }
        }
        PaneUrlSignature {
            tab_idx,
            active_tool_id: None,
            state_revision: 0,
        }
    }

    pub(super) fn capture_workspace_url_signature(&self) -> UrlStateSignature {
        let layout = self.viewport_layout;
        let pane_count = layout.pane_count();
        let focused = self.active_focused_pane();
        let mut pane_revisions = [PaneUrlSignature::default(); 4];
        for (idx, slot) in pane_revisions.iter_mut().enumerate().take(pane_count) {
            *slot = self.pane_url_signature(idx);
        }
        let focused_sig = pane_revisions[focused];
        let half = 0.5_f32.to_bits();
        let split_bits = match layout {
            ViewportLayout::SplitHorizontal | ViewportLayout::SplitVertical => {
                self.split_ratio.clamp(0.20, 0.80).to_bits()
            }
            ViewportLayout::Single | ViewportLayout::QuadGrid => half,
        };
        let quad_bits = if layout == ViewportLayout::QuadGrid {
            (
                self.quad_split_ratios.0.clamp(0.20, 0.80).to_bits(),
                self.quad_split_ratios.1.clamp(0.20, 0.80).to_bits(),
            )
        } else {
            (half, half)
        };
        let compare_sync = if layout == ViewportLayout::Single {
            ComparisonSyncConfig::default()
        } else {
            self.compare_sync
        };
        UrlStateSignature {
            selected_tab: focused_sig.tab_idx,
            active_tool_id: focused_sig.active_tool_id,
            state_revision: focused_sig.state_revision,
            viewport_layout: layout,
            split_ratio_bits: split_bits,
            quad_split_ratio_bits: quad_bits,
            focused_pane: focused,
            compare_sync,
            pane_revisions,
        }
    }

    fn export_pane_share_params(&self, pane_idx: usize) -> Result<ShareUrlParams, String> {
        if pane_idx == 0 {
            let tab_idx = self.active_primary_tab_index();
            let tab = self
                .tabs
                .get(tab_idx)
                .ok_or_else(|| "No active primary simulation tab selected".to_string())?;
            return tab.export_share_url_params();
        }
        let tab_idx = self.pane_tabs.get(pane_idx).copied().unwrap_or(0);
        if let Some(tab) = self.pane_tab(pane_idx) {
            return tab.export_share_url_params();
        }
        let mut temp_tab = crate::tabs::instantiate_tab(tab_idx)
            .ok_or_else(|| format!("Invalid tab index {tab_idx} in secondary pane {pane_idx}"))?;
        if let Some(slot) = self.pane_slots.slot(pane_idx) {
            if let Some(saved) = slot.saved_states.get(temp_tab.name()) {
                temp_tab.load_state(saved);
            }
        }
        temp_tab.export_share_url_params()
    }

    pub(super) fn encode_workspace_or_single_fragment(&self) -> Result<String, String> {
        let layout = self.viewport_layout;
        if layout == ViewportLayout::Single {
            let primary = self.export_pane_share_params(0)?;
            return encode_single_pane_smart_fragment_from_params(&primary);
        }
        let pane_count = layout.pane_count();
        let mut panes = Vec::with_capacity(pane_count);
        for pane_idx in 0..pane_count {
            panes.push(self.export_pane_share_params(pane_idx)?);
        }
        let ws_params = WorkspaceShareParams {
            layout,
            split_ratio: self.split_ratio.clamp(0.20, 0.80),
            quad_split_ratios: (
                self.quad_split_ratios.0.clamp(0.20, 0.80),
                self.quad_split_ratios.1.clamp(0.20, 0.80),
            ),
            focused_pane: self.active_focused_pane(),
            compare_sync: self.compare_sync,
            panes,
        };
        encode_workspace_share_fragment(&ws_params)
    }

    fn build_validated_candidate_tab(
        &self,
        params: &ShareUrlParams,
        allow_workspace_bounds: bool,
        preserve_primary_subtools: bool,
        ctx: Option<&egui::Context>,
    ) -> Result<(usize, Box<dyn ExplorerTab>), String> {
        let Some(tab_idx) = self
            .tabs
            .iter()
            .position(|tab| matches_tab_slug(&params.tab, tab.url_slug(), tab.name()))
        else {
            return Err(format!(
                "Unknown simulation tab '{}' in shared URL",
                params.tab
            ));
        };
        let mut candidate = crate::tabs::instantiate_tab(tab_idx).ok_or_else(|| {
            format!(
                "Failed to instantiate simulation tab '{}' for shared URL",
                params.tab
            )
        })?;
        if preserve_primary_subtools {
            if let Some(existing_json) = self.tabs[tab_idx].save_state() {
                candidate.load_state(&existing_json);
            }
        }
        let res = if allow_workspace_bounds {
            candidate.import_workspace_url_share_params(params, ctx)
        } else {
            candidate.import_url_share_params(params, ctx)
        };
        res?;
        Ok((tab_idx, candidate))
    }

    pub(super) fn hydrate_parsed_location(
        &mut self,
        parsed: ParsedShareLocation,
        ctx: Option<&egui::Context>,
        record_undo_from: Option<super::AppState>,
    ) -> Result<bool, String> {
        match parsed {
            ParsedShareLocation::SinglePane(single) => {
                self.hydrate_single_pane_params(single, ctx, record_undo_from)
            }
            ParsedShareLocation::Workspace(ws) => {
                self.hydrate_workspace_params(ws, ctx, record_undo_from)
            }
        }
    }

    fn hydrate_single_pane_params(
        &mut self,
        params: ShareUrlParams,
        ctx: Option<&egui::Context>,
        record_undo_from: Option<super::AppState>,
    ) -> Result<bool, String> {
        let (tab_idx, candidate) =
            match self.build_validated_candidate_tab(&params, false, true, ctx) {
                Ok(pair) => pair,
                Err(err) => {
                    self.record_invalid_share_url(err.clone(), ctx);
                    return Err(err);
                }
            };
        let tab_name = candidate.name();
        self.tab_state_cache.invalidate_tab(tab_name);
        self.tabs[tab_idx] = candidate;
        self.selected_tab = tab_idx;
        self.focused_pane = 0;
        self.set_layout(ViewportLayout::Single);
        if let Some(egui_ctx) = ctx {
            egui_ctx.data_mut(|d| {
                d.insert_temp(
                    egui::Id::new(super::viewport::focus::PENDING_PANE_FOCUS_KEY),
                    0_usize,
                );
                d.insert_temp(
                    egui::Id::new(super::viewport::focus::ACTIVE_FOCUSED_PANE_KEY),
                    0_usize,
                );
            });
        }
        self.finalize_hydrated_url_state(ctx, record_undo_from);
        Ok(true)
    }

    fn hydrate_workspace_params(
        &mut self,
        params: WorkspaceShareParams,
        ctx: Option<&egui::Context>,
        record_undo_from: Option<super::AppState>,
    ) -> Result<bool, String> {
        let single_pane = params.layout == ViewportLayout::Single;
        let mut candidates = Vec::with_capacity(params.panes.len());
        for (idx, pane_param) in params.panes.iter().enumerate() {
            let preserve_subtools = single_pane && idx == 0;
            match self.build_validated_candidate_tab(pane_param, true, preserve_subtools, ctx) {
                Ok(pair) => candidates.push(pair),
                Err(err) => {
                    self.record_invalid_share_url(err.clone(), ctx);
                    return Err(err);
                }
            }
        }
        self.commit_validated_workspace_panes(&params, candidates, ctx);
        self.finalize_hydrated_url_state(ctx, record_undo_from);
        Ok(true)
    }

    fn commit_validated_workspace_panes(
        &mut self,
        params: &WorkspaceShareParams,
        candidates: Vec<(usize, Box<dyn ExplorerTab>)>,
        ctx: Option<&egui::Context>,
    ) {
        let pane_count = params.layout.pane_count();
        self.is_dragging_divider = false;
        self.dragging_quad_axes = (false, false);

        let mut iter = candidates.into_iter().enumerate();
        if let Some((_, (tab_idx_0, candidate_0))) = iter.next() {
            self.tab_state_cache.invalidate_tab(candidate_0.name());
            self.tabs[tab_idx_0] = candidate_0;
            self.selected_tab = tab_idx_0;
            if !self.pane_tabs.is_empty() {
                self.pane_tabs[0] = tab_idx_0;
            }
        }
        for (pane_idx, (tab_idx, candidate)) in iter {
            if pane_idx < self.pane_tabs.len() {
                self.pane_tabs[pane_idx] = tab_idx;
            }
            if pane_idx == 1 {
                self.secondary_tab = tab_idx;
            }
            if let Some(slot) = self.pane_slots.slot_mut(pane_idx) {
                slot.reset_active_without_snapshot();
                slot.initialized = true;
                slot.active = Some((tab_idx, candidate));
                slot.snapshot_active();
            }
        }

        self.viewport_layout = params.layout;
        self.sync_layout();
        self.prune_inactive_pane_slots(pane_count);
        self.split_ratio = params.split_ratio.clamp(0.20, 0.80);
        self.quad_split_ratios = (
            params.quad_split_ratios.0.clamp(0.20, 0.80),
            params.quad_split_ratios.1.clamp(0.20, 0.80),
        );
        self.focused_pane = params.focused_pane.min(pane_count.saturating_sub(1));
        self.compare_sync = params.compare_sync;

        if let Some(egui_ctx) = ctx {
            let target_focus = self.focused_pane;
            egui_ctx.data_mut(|d| {
                d.insert_temp(
                    egui::Id::new(super::viewport::focus::PENDING_PANE_FOCUS_KEY),
                    target_focus,
                );
                d.insert_temp(
                    egui::Id::new(super::viewport::focus::ACTIVE_FOCUSED_PANE_KEY),
                    target_focus,
                );
            });
        }
    }

    fn finalize_hydrated_url_state(
        &mut self,
        ctx: Option<&egui::Context>,
        record_undo_from: Option<super::AppState>,
    ) {
        self.clear_active_pane_caches();
        self.session_manager.clear_toast();

        if let Some(prev_state) = record_undo_from {
            let new_state = self.get_app_state();
            self.session_manager
                .record_external_url_override(prev_state, new_state);
        }

        let canonical_frag = self.encode_current_share_fragment().ok();
        if let Some(ref frag) = canonical_frag {
            let _ = replace_browser_url_fragment(frag);
        } else {
            let _ = replace_browser_url_fragment("");
        }
        self.url_sync
            .mark_synced(self.current_url_signature(), canonical_frag);
        if let Some(egui_ctx) = ctx {
            egui_ctx.request_repaint();
        }
    }

    /// Drains queued browser `popstate` / `hashchange` navigations and applies valid external share
    /// hashes with workspace undo support while ignoring non-share document anchors.
    pub(super) fn drain_browser_url_navigations(&mut self, ctx: &egui::Context) {
        for (hash, search) in crate::url_state::drain_pending_browser_navigations() {
            self.apply_external_browser_navigation(&hash, &search, ctx);
        }
    }

    pub(crate) fn apply_external_browser_navigation(
        &mut self,
        hash: &str,
        search: &str,
        ctx: &egui::Context,
    ) {
        match parse_workspace_or_share_location(hash, search) {
            Ok(None) => {
                // Non-share anchor (e.g. `#overview` or `#utm_source=docs`): keep active multi-pane
                // layout without warnings and avoid immediately overwriting the anchor until edited.
                self.url_sync
                    .suppress_initial_overwrite(self.current_url_signature());
            }
            Ok(Some(parsed)) => {
                self.clear_active_pane_caches();
                let prev_state = self.get_app_state();
                let _ = self.hydrate_parsed_location(parsed, Some(ctx), Some(prev_state));
            }
            Err(err) => {
                self.record_invalid_share_url(err, Some(ctx));
            }
        }
    }
}
