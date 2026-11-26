#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Scoped per-tool, per-plot overlay snapshot persistence for [`super::SimulationFramework`].

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use eframe::egui;
use serde::Deserializer;
use serde_json::value::RawValue;

use super::{
    SimulationFramework, ToolStateBounds, MAX_SESSION_STATE_BYTES, MAX_SESSION_TOOL_ENTRIES,
    MAX_TOOL_KEY_BYTES,
};

/// Maximum number of distinct `Plot` keys persisted for a single tool.
pub const MAX_PLOTS_PER_TOOL: usize = 16;

/// Deterministically ordered map from `plot_key` to [`egui_plot::PlotOverlaySnapshot`].
pub type ToolPlotSnapshotMap = BTreeMap<String, egui_plot::PlotOverlaySnapshot>;

/// Deterministically ordered map from tool identifier/name to [`ToolPlotSnapshotMap`].
pub type SavedPlotStatesMap = BTreeMap<String, ToolPlotSnapshotMap>;

fn cap_tool_plot_map(plot_map: &mut ToolPlotSnapshotMap) -> Result<(), String> {
    if plot_map.len() > MAX_PLOTS_PER_TOOL {
        return Err("plot_states per-tool plot count limit exceeded".to_owned());
    }
    let mut non_default_keys = Vec::with_capacity(plot_map.len());
    let mut total_points = 0_usize;
    for (plot_key, snap) in plot_map.iter_mut() {
        if plot_key.len() > MAX_TOOL_KEY_BYTES {
            return Err("plot_states plot_key length limit exceeded".to_owned());
        }
        snap.sanitize_and_validate()?;
        let pts: usize = snap.imported_datasets.iter().map(|d| d.points.len()).sum();
        let next_total = total_points.saturating_add(pts);
        if next_total > egui_plot::snapshot::MAX_SNAPSHOT_TOTAL_POINTS {
            let allowed =
                egui_plot::snapshot::MAX_SNAPSHOT_TOTAL_POINTS.saturating_sub(total_points);
            trim_snapshot_points(snap, allowed);
        }
        total_points = total_points
            .saturating_add(snap.imported_datasets.iter().map(|d| d.points.len()).sum());
        if !snap.is_default() {
            non_default_keys.push(plot_key.clone());
        }
    }
    plot_map.retain(|k, _| non_default_keys.iter().any(|keep| keep == k));
    Ok(())
}

fn trim_snapshot_points(snap: &mut egui_plot::PlotOverlaySnapshot, mut budget: usize) {
    for ds in &mut snap.imported_datasets {
        if budget == 0 {
            ds.points.clear();
        } else if ds.points.len() > budget {
            let last = ds.points.len() - 1;
            ds.points = if budget == 1 {
                vec![ds.points[0]]
            } else {
                (0..budget)
                    .map(|i| ds.points[(i * last) / (budget - 1)])
                    .collect()
            };
            budget = 0;
        } else {
            budget -= ds.points.len();
        }
    }
    snap.imported_datasets.retain(|d| !d.points.is_empty());
}

pub(super) fn deserialize_plot_states<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<SavedPlotStatesMap, D::Error> {
    struct PlotStatesVisitor;

    impl<'de> serde::de::Visitor<'de> for PlotStatesVisitor {
        type Value = SavedPlotStatesMap;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a bounded map of tool names to per-plot PlotOverlaySnapshot maps")
        }

        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            use serde::de::Error;
            let mut out = SavedPlotStatesMap::new();
            let mut entry_count = 0_usize;
            let mut total_bytes = 0_usize;

            while let Some((tool_key, raw)) = map.next_entry::<String, Box<RawValue>>()? {
                entry_count = entry_count.saturating_add(1);
                if entry_count > MAX_SESSION_TOOL_ENTRIES {
                    return Err(M::Error::custom("plot_states tool count limit exceeded"));
                }
                if tool_key.len() > MAX_TOOL_KEY_BYTES {
                    return Err(M::Error::custom(
                        "plot_states tool key length limit exceeded",
                    ));
                }
                let raw_text = raw.get();
                total_bytes = total_bytes
                    .saturating_add(tool_key.len())
                    .saturating_add(raw_text.len());
                if total_bytes > MAX_SESSION_STATE_BYTES {
                    return Err(M::Error::custom("plot_states session byte limit exceeded"));
                }
                ToolStateBounds::SESSION
                    .validate_json(raw_text)
                    .map_err(M::Error::custom)?;
                let mut plot_map: ToolPlotSnapshotMap =
                    serde_json::from_str(raw_text).map_err(M::Error::custom)?;
                cap_tool_plot_map(&mut plot_map).map_err(M::Error::custom)?;
                if !plot_map.is_empty() {
                    out.insert(tool_key, plot_map);
                }
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(PlotStatesVisitor)
}

/// Merges migrated legacy [`crate::app::WorkspaceEnvelope::callout_manager`] callouts into a serialized
/// [`SimulationFramework`] tab state JSON string under `plot_states`.
pub(crate) fn merge_migrated_callouts_into_tab_json(
    existing_tab_json: Option<&str>,
    preferred_tool_name: Option<&str>,
    callouts: &[egui_plot::SavedCallout],
) -> Option<String> {
    if callouts.is_empty() {
        return existing_tab_json.map(str::to_owned);
    }
    let mut root: serde_json::Value = existing_tab_json
        .and_then(|s| serde_json::from_str(s).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| {
            serde_json::json!({
                "selected_tool_index": Option::<usize>::None,
                "selected_tool_name": preferred_tool_name,
                "show_theory_portal": false,
                "show_starter_gallery": false,
                "tool_states": {},
            })
        });
    let root_obj = root.as_object_mut()?;
    if root_obj
        .get("selected_tool_name")
        .is_none_or(serde_json::Value::is_null)
    {
        if let Some(tool_name) = preferred_tool_name {
            root_obj.insert(
                "selected_tool_name".to_owned(),
                serde_json::Value::String(tool_name.to_owned()),
            );
        }
    }
    let target_tool = root_obj
        .get("selected_tool_name")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .or(preferred_tool_name)
        .unwrap_or("*")
        .to_owned();

    let mut plot_states: SavedPlotStatesMap = root_obj
        .get("plot_states")
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();
    let tool_entry = plot_states.entry(target_tool).or_default();
    let target_plot_key = tool_entry
        .keys()
        .next()
        .cloned()
        .unwrap_or_else(|| "*".to_owned());
    let snap = tool_entry.entry(target_plot_key).or_default();
    snap.callouts.extend(callouts.iter().cloned());
    let _ = snap.sanitize_and_validate();
    let _ = cap_tool_plot_map(tool_entry);
    if tool_entry.is_empty() {
        plot_states.retain(|_, m| !m.is_empty());
    }
    if let Ok(val) = serde_json::to_value(&plot_states) {
        root_obj.insert("plot_states".to_owned(), val);
    }
    serde_json::to_string(&root).ok()
}

impl SimulationFramework {
    pub(super) fn bump_plot_sync_generation(&mut self) {
        self.plot_sync_generation = self.plot_sync_generation.wrapping_add(1).max(1);
    }

    pub(super) fn hash_saved_plot_states(&self, hasher: &mut impl Hasher) {
        self.saved_plot_states.len().hash(hasher);
        for (tool_key, plot_map) in &self.saved_plot_states {
            tool_key.hash(hasher);
            plot_map.len().hash(hasher);
            for (plot_key, snap) in plot_map {
                plot_key.hash(hasher);
                snap.hash_state(hasher);
            }
        }
    }

    pub(super) fn promote_legacy_plot_keys_for_active_tool(&mut self) {
        let Some(tool_name) = self.active_tool.as_ref().map(|t| t.name().to_owned()) else {
            return;
        };
        let mut promoted = false;
        let alias_id = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
            .map(|meta| meta.id)
            .filter(|&id| id != tool_name);
        if let Some(alias_map) = alias_id.and_then(|id| self.saved_plot_states.remove(id)) {
            let target = self.saved_plot_states.entry(tool_name.clone()).or_default();
            for (k, v) in alias_map {
                target.entry(k).or_insert(v);
            }
            promoted = true;
        }
        if let Some(wildcard_tool_map) = self.saved_plot_states.remove("*") {
            let target = self.saved_plot_states.entry(tool_name).or_default();
            for (k, v) in wildcard_tool_map {
                match target.get_mut(&k) {
                    Some(existing) => {
                        existing.callouts.extend(v.callouts);
                        let _ = existing.sanitize_and_validate();
                    }
                    None => {
                        target.insert(k, v);
                    }
                }
            }
            promoted = true;
        }
        if promoted {
            self.bump_plot_sync_generation();
            self.bump_framework_revision();
        }
    }

    pub(super) fn begin_plot_scope(&mut self, ctx: &egui::Context) {
        self.promote_legacy_plot_keys_for_active_tool();
        if self.plot_sync_generation == 0 {
            self.plot_sync_generation = 1;
        }
        let empty = ToolPlotSnapshotMap::new();
        let seed = self
            .active_tool
            .as_ref()
            .and_then(|tool| self.saved_plot_states.get(tool.name()))
            .unwrap_or(&empty);
        egui_plot::begin_plot_persistence_scope(
            ctx,
            self.instance_id,
            self.plot_sync_generation,
            seed,
        );
    }

    pub(super) fn end_plot_scope(&mut self, ctx: &egui::Context) {
        let mut updated = egui_plot::end_plot_persistence_scope(ctx, self.instance_id);
        let Some(tool_name) = self.active_tool.as_ref().map(|t| t.name()) else {
            return;
        };
        let _ = cap_tool_plot_map(&mut updated);
        let changed = match self.saved_plot_states.get(tool_name) {
            Some(existing) => existing != &updated,
            None => !updated.is_empty(),
        };
        if changed {
            if updated.is_empty() {
                self.saved_plot_states.remove(tool_name);
            } else {
                self.saved_plot_states.insert(tool_name.to_owned(), updated);
            }
            self.bump_framework_revision();
        }
    }

    /// Clears persisted manual 2D viewport bounds for the active tool while retaining overlays
    /// (imported CSV datasets, callouts, log scales, fits, and table state).
    pub fn reset_active_tool_manual_bounds(&mut self) {
        let mut keys_to_reset = Vec::with_capacity(2);
        if let Some(tool) = &self.active_tool {
            keys_to_reset.push(tool.name().to_owned());
        }
        if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            keys_to_reset.push(meta.id.to_owned());
        }
        for key in keys_to_reset {
            if let Some(plot_map) = self.saved_plot_states.get_mut(&key) {
                for snap in plot_map.values_mut() {
                    snap.clear_manual_bounds();
                }
                plot_map.retain(|_, snap| !snap.is_default());
                if plot_map.is_empty() {
                    self.saved_plot_states.remove(&key);
                }
            }
        }
        self.bump_plot_sync_generation();
        self.bump_framework_revision();
    }

    pub(super) fn poll_preset_import_and_sync(&mut self, ctx: &egui::Context) {
        let imported = self.active_tool.as_mut().is_some_and(|tool| {
            self.preset_dialog
                .poll_pending_import(|json_str| tool.import_preset_json(json_str))
        });
        if imported {
            self.reset_active_tool_manual_bounds();
            self.sync_active_baseline();
            ctx.request_repaint();
        }
    }
}
