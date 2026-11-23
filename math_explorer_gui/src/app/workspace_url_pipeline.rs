//! Cross-pane dataflow pipeline URL (`&pipe=<base64url>`) encoding and atomic hydration
//! bridged into [`MathExplorerApp`]'s workspace URL and undo/history lifecycle.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::dataflow_bus::{PaneLink, compatible_stream_kind};
use super::{ComparisonSyncConfig, MathExplorerApp, ViewportLayout};
use crate::tabs::ExplorerTab;
use crate::url_state::pipeline::{
    PipelineUrlEnvelope, ValidatedPipelineUrl, build_pane_url_entry, encode_pipeline_share_fragment,
};
use crate::url_state::{
    MAX_DECODED_URL_STATE_BYTES, ShareUrlParams, WorkspaceShareParams, matches_tab_slug,
};
use eframe::egui;
use std::hash::{Hash, Hasher};

impl MathExplorerApp {
    pub(super) fn compute_pipeline_links_hash(&self, layout: ViewportLayout) -> u64 {
        if layout == ViewportLayout::Single || self.pipeline_dag.is_empty() {
            return 0;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.pipeline_dag.links().hash(&mut h);
        h.finish().max(1)
    }

    pub(super) fn try_encode_compact_pipeline_fragment(
        &self,
        layout: ViewportLayout,
        focused_pane: usize,
        panes: &[ShareUrlParams],
    ) -> Option<Result<String, String>> {
        if layout == ViewportLayout::Single || self.pipeline_dag.is_empty() {
            return None;
        }
        let default_ratios = match layout {
            ViewportLayout::SplitHorizontal | ViewportLayout::SplitVertical => {
                (self.split_ratio - 0.5).abs() < 1e-6
            }
            ViewportLayout::QuadGrid => {
                (self.quad_split_ratios.0 - 0.5).abs() < 1e-6
                    && (self.quad_split_ratios.1 - 0.5).abs() < 1e-6
            }
            ViewportLayout::Single => false,
        };
        let has_hud_params = panes
            .iter()
            .any(|p| p.preset.is_some() || p.challenge.is_some() || p.step.is_some());
        if !default_ratios || self.compare_sync != ComparisonSyncConfig::default() || has_hud_params
        {
            return None;
        }

        let mut pipe_panes = Vec::with_capacity(panes.len());
        for pane in panes {
            if let Some(ref json) = pane.state_json {
                if let Err(cap_err) = crate::url_state::pipeline::enforce_url_graph_node_cap(json) {
                    return Some(Err(cap_err));
                }
                if json.len() > MAX_DECODED_URL_STATE_BYTES {
                    return None;
                }
            }
            match build_pane_url_entry(&pane.tab, pane.tool.as_deref(), pane.state_json.as_deref())
            {
                Ok(entry) => pipe_panes.push(entry),
                Err(_) => return None,
            }
        }
        let focused = panes.get(focused_pane)?;
        let env = PipelineUrlEnvelope {
            layout,
            focused_pane,
            panes: pipe_panes,
            links: self.pipeline_dag.links().to_vec(),
        };
        match encode_pipeline_share_fragment(&focused.tab, focused.tool.as_deref(), env) {
            Ok(frag) if frag.len() <= MAX_DECODED_URL_STATE_BYTES => Some(Ok(frag)),
            _ => None,
        }
    }

    pub(super) fn validate_candidate_pipeline_links(
        candidates: &[(usize, Box<dyn ExplorerTab>)],
        links: &[PaneLink],
    ) -> Result<(), String> {
        for link in links {
            let Some((_, src_tab)) = candidates.get(link.from_pane) else {
                return Err("Out-of-bounds source pane in shared pipeline URL".to_string());
            };
            let Some((_, dst_tab)) = candidates.get(link.to_pane) else {
                return Err("Out-of-bounds target pane in shared pipeline URL".to_string());
            };
            if compatible_stream_kind(src_tab.active_tool_id(), dst_tab.active_tool_id())
                != Some(link.kind)
            {
                return Err("Incompatible tool stream link in shared pipeline URL".to_string());
            }
        }
        Ok(())
    }

    pub(super) fn hydrate_pipeline_share_params(
        &mut self,
        pipe: ValidatedPipelineUrl,
        ctx: Option<&egui::Context>,
        record_undo_from: Option<super::AppState>,
    ) -> Result<bool, String> {
        let mut candidates: Vec<(usize, Box<dyn ExplorerTab>)> =
            Vec::with_capacity(pipe.panes.len());
        for desc in &pipe.panes {
            let Some(tab_idx) = self
                .tabs
                .iter()
                .position(|tab| matches_tab_slug(&desc.tab, tab.url_slug(), tab.name()))
            else {
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
            candidates.push((tab_idx, instance));
        }
        if let Err(err) = Self::validate_candidate_pipeline_links(&candidates, &pipe.links) {
            self.record_invalid_share_url(err.clone(), ctx);
            return Err(err);
        }

        let ws_params = WorkspaceShareParams {
            layout: pipe.layout,
            split_ratio: 0.5,
            quad_split_ratios: (0.5, 0.5),
            focused_pane: pipe.focused_pane,
            compare_sync: ComparisonSyncConfig::default(),
            panes: Vec::new(),
            links: pipe.links,
        };
        self.commit_validated_workspace_panes(&ws_params, candidates, ctx);
        self.finalize_hydrated_url_state(ctx, record_undo_from);
        Ok(true)
    }
}
