//! Compact multi-pane workspace link URL descriptor (`&ws=<base64url>`) encoding and atomic restore.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::{MathExplorerApp, ViewportLayout};
use crate::url_state::URL_SYNC_DEBOUNCE_MS;
use std::hash::{Hash, Hasher};

const _KEEP_SYNC_IMPORT: u64 = URL_SYNC_DEBOUNCE_MS;

/// Per-pane compact URL descriptor inside [`WorkspaceLinkUrlDescriptor`].
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkspacePaneUrlEntry {
    pub tab: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<serde_json::Value>,
}

/// Compact multi-pane layout and `SharedWorkspaceBus` link descriptor serialized under `&ws=`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkspaceLinkUrlDescriptor {
    pub layout: String,
    pub links: Vec<bool>,
    #[serde(default)]
    pub focus: usize,
    #[serde(default = "default_ratio")]
    pub ratio: f32,
    #[serde(default = "default_quad_ratios")]
    pub quad: (f32, f32),
    pub panes: Vec<WorkspacePaneUrlEntry>,
}

const fn default_ratio() -> f32 {
    0.5
}

const fn default_quad_ratios() -> (f32, f32) {
    (0.5, 0.5)
}

fn encode_layout_code(layout: ViewportLayout) -> Option<&'static str> {
    match layout {
        ViewportLayout::Single => None,
        ViewportLayout::SplitHorizontal => Some("h"),
        ViewportLayout::SplitVertical => Some("v"),
        ViewportLayout::QuadGrid => Some("q"),
    }
}

fn decode_layout_code(code: &str) -> Result<ViewportLayout, String> {
    match code {
        "h" => Ok(ViewportLayout::SplitHorizontal),
        "v" => Ok(ViewportLayout::SplitVertical),
        "q" => Ok(ViewportLayout::QuadGrid),
        other => Err(format!("Unsupported workspace URL layout code '{other}'")),
    }
}

impl MathExplorerApp {
    pub(super) fn workspace_url_revision_mix(&self, base_rev: u64) -> u64 {
        if !self.has_active_multi_pane_bus_links() {
            return base_rev;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        base_rev.hash(&mut h);
        encode_layout_code(self.viewport_layout).hash(&mut h);
        self.active_focused_pane().hash(&mut h);
        self.split_ratio.clamp(0.20, 0.80).to_bits().hash(&mut h);
        self.quad_split_ratios
            .0
            .clamp(0.20, 0.80)
            .to_bits()
            .hash(&mut h);
        self.quad_split_ratios
            .1
            .clamp(0.20, 0.80)
            .to_bits()
            .hash(&mut h);
        let count = self.viewport_layout.pane_count();
        for pane_idx in 0..count {
            self.is_pane_bus_linked(pane_idx).hash(&mut h);
            if let Some(tab) = self.pane_tab(pane_idx) {
                tab.url_slug().hash(&mut h);
                tab.active_tool_id().hash(&mut h);
                tab.url_state_revision().hash(&mut h);
            } else {
                self.pane_tabs.get(pane_idx).copied().hash(&mut h);
            }
        }
        h.finish().max(1)
    }

    fn export_pane_url_entry(&self, pane_idx: usize) -> Result<WorkspacePaneUrlEntry, String> {
        if let Some(tab) = self.pane_tab(pane_idx) {
            return build_entry_from_tab(tab);
        }
        let assigned = if pane_idx == 0 {
            self.active_primary_tab_index()
        } else {
            self.pane_tabs.get(pane_idx).copied().unwrap_or(0)
        };
        let proto = self
            .tabs
            .get(assigned)
            .ok_or_else(|| format!("Invalid tab index {assigned} on pane {pane_idx}"))?;
        if pane_idx > 0 {
            if let Some(mut temp_tab) = crate::tabs::instantiate_tab(assigned) {
                if let Some(saved) = self
                    .pane_slots
                    .slot(pane_idx)
                    .and_then(|s| s.saved_states.get(proto.name()))
                {
                    temp_tab.load_state(saved);
                }
                return build_entry_from_tab(temp_tab.as_ref());
            }
        }
        build_entry_from_tab(proto.as_ref())
    }

    pub(super) fn encode_workspace_link_url_json(&self) -> Result<Option<String>, String> {
        if !self.has_active_multi_pane_bus_links() {
            return Ok(None);
        }
        let Some(layout_code) = encode_layout_code(self.viewport_layout) else {
            return Ok(None);
        };
        let count = self.viewport_layout.pane_count();
        let mut panes = Vec::with_capacity(count);
        for pane_idx in 0..count {
            panes.push(self.export_pane_url_entry(pane_idx)?);
        }
        let desc = WorkspaceLinkUrlDescriptor {
            layout: layout_code.to_string(),
            links: self.pane_bus_links[..count].to_vec(),
            focus: self.active_focused_pane(),
            ratio: self.split_ratio.clamp(0.20, 0.80),
            quad: (
                self.quad_split_ratios.0.clamp(0.20, 0.80),
                self.quad_split_ratios.1.clamp(0.20, 0.80),
            ),
            panes,
        };
        serde_json::to_string(&desc)
            .map(Some)
            .map_err(|e| format!("Failed to serialize workspace URL descriptor: {e}"))
    }

    pub(super) fn apply_workspace_link_url_json(&mut self, ws_json: &str) -> Result<(), String> {
        let desc: WorkspaceLinkUrlDescriptor = serde_json::from_str(ws_json)
            .map_err(|e| format!("Invalid workspace URL descriptor: {e}"))?;
        let layout = decode_layout_code(&desc.layout)?;
        let count = layout.pane_count();
        if desc.links.len() != count || desc.panes.len() != count {
            return Err("Workspace URL descriptor pane count mismatch".to_string());
        }
        if desc.focus >= count
            || !desc.ratio.is_finite()
            || !desc.quad.0.is_finite()
            || !desc.quad.1.is_finite()
        {
            return Err("Workspace URL descriptor focus or split ratios out of bounds".to_string());
        }
        let mut resolved_indices = Vec::with_capacity(count);
        for entry in &desc.panes {
            let tab_idx = self
                .tabs
                .iter()
                .position(|t| {
                    crate::url_state::matches_tab_slug(&entry.tab, t.url_slug(), t.name())
                })
                .ok_or_else(|| {
                    format!("Unknown simulation tab '{}' in workspace URL", entry.tab)
                })?;
            resolved_indices.push(tab_idx);
        }

        let rollback = self.get_app_state();
        if let Err(err) = self.commit_workspace_url_descriptor(layout, &desc, &resolved_indices) {
            self.apply_app_state(rollback);
            return Err(err);
        }
        Ok(())
    }

    fn commit_workspace_url_descriptor(
        &mut self,
        layout: ViewportLayout,
        desc: &WorkspaceLinkUrlDescriptor,
        resolved_indices: &[usize],
    ) -> Result<(), String> {
        self.split_ratio = desc.ratio.clamp(0.20, 0.80);
        self.quad_split_ratios = (desc.quad.0.clamp(0.20, 0.80), desc.quad.1.clamp(0.20, 0.80));
        self.set_layout(layout);
        for (pane_idx, (&tab_idx, entry)) in resolved_indices.iter().zip(&desc.panes).enumerate() {
            self.set_pane_tab(pane_idx, tab_idx);
            let preset_str = entry
                .preset
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .map_err(|e| format!("Invalid pane preset JSON: {e}"))?;
            let tab = self
                .pane_tab_mut(pane_idx)
                .ok_or_else(|| format!("Failed to allocate pane {pane_idx}"))?;
            tab.import_url_preset_json(entry.tool.as_deref(), preset_str.as_deref())?;
        }
        self.pane_bus_links = [false; 4];
        for (i, &linked) in desc.links.iter().enumerate() {
            self.pane_bus_links[i] = linked;
        }
        self.focused_pane = desc.focus;
        self.tab_state_cache.clear();
        self.workspace_bus.reset();
        self.sync_linked_panes_bus();
        Ok(())
    }
}

fn build_entry_from_tab(
    tab: &dyn crate::tabs::ExplorerTab,
) -> Result<WorkspacePaneUrlEntry, String> {
    let preset = match tab.export_url_preset_json()? {
        Some(raw) => Some(
            serde_json::from_str(&raw)
                .map_err(|e| format!("Invalid URL preset JSON from '{}': {e}", tab.name()))?,
        ),
        None => None,
    };
    Ok(WorkspacePaneUrlEntry {
        tab: tab.url_slug().to_string(),
        tool: tab.active_tool_id().map(str::to_string),
        preset,
    })
}
