#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

struct RawToolStates<'a> {
    saved: &'a HashMap<String, String>,
    active: Option<(&'a str, &'a str)>,
}

impl Serialize for RawToolStates<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let extra = usize::from(
            self.active
                .is_some_and(|(name, _)| !self.saved.contains_key(name)),
        );
        let mut map = serializer.serialize_map(Some(self.saved.len() + extra))?;
        for (key, val) in self.saved {
            if self.active.is_some_and(|(active_key, _)| active_key == key) {
                continue;
            }
            serialize_raw_or_str(&mut map, key, val)?;
        }
        if let Some((active_key, active_val)) = self.active {
            serialize_raw_or_str(&mut map, active_key, active_val)?;
        }
        map.end()
    }
}

fn serialize_raw_or_str<M: SerializeMap>(
    map: &mut M,
    key: &str,
    json_or_text: &str,
) -> Result<(), M::Error> {
    if let Ok(raw) = serde_json::from_str::<&RawValue>(json_or_text) {
        map.serialize_entry(key, raw)
    } else {
        map.serialize_entry(key, json_or_text)
    }
}

fn deserialize_tool_states<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<HashMap<String, String>, D::Error> {
    let raw_map = HashMap::<String, Box<RawValue>>::deserialize(deserializer)?;
    let mut out = HashMap::with_capacity(raw_map.len());
    for (key, raw) in raw_map {
        let raw_text = raw.get();
        if raw_text.starts_with('"') {
            if let Ok(unescaped) = serde_json::from_str::<String>(raw_text) {
                out.insert(key, unescaped);
                continue;
            }
        }
        let boxed_str: Box<str> = raw.into();
        out.insert(key, String::from(boxed_str));
    }
    Ok(out)
}

#[derive(Serialize)]
struct FrameworkStateSerialize<'a> {
    selected_tool_index: Option<usize>,
    selected_tool_name: Option<&'a str>,
    show_theory_portal: bool,
    show_starter_gallery: bool,
    tool_states: RawToolStates<'a>,
}

#[derive(Deserialize)]
struct FrameworkStateDeserialize {
    selected_tool_index: Option<usize>,
    selected_tool_name: Option<String>,
    show_theory_portal: bool,
    #[serde(default)]
    show_starter_gallery: bool,
    #[serde(deserialize_with = "deserialize_tool_states")]
    tool_states: HashMap<String, String>,
}

pub(super) fn capture_isolated_tool_datasets(
    tool: &dyn InteractiveTool,
) -> Vec<(String, Vec<[f64; 2]>)> {
    let prev = egui_plot::get_last_plot_datasets();
    let prev_prims = egui_plot::get_last_plot_primitives();
    let prev_subplots = egui_plot::get_last_plot_subplots();
    egui_plot::clear_last_plot_datasets();
    egui_plot::clear_last_plot_primitives();
    let datasets = tool.export_datasets();
    egui_plot::clear_last_plot_datasets();
    egui_plot::clear_last_plot_primitives();
    if !prev.is_empty() {
        egui_plot::append_last_plot_datasets(&prev);
    }
    if !prev_prims.is_empty() {
        egui_plot::append_last_plot_primitives(&prev_prims);
    }
    for subplot in prev_subplots {
        egui_plot::append_last_plot_subplot(subplot);
    }
    datasets
}

impl SimulationFramework {
    pub(super) fn bump_framework_revision(&mut self) {
        self.rev_cache.framework_revision = self.rev_cache.framework_revision.wrapping_add(1);
        self.rev_cache.untracked_tool_revision =
            self.rev_cache.untracked_tool_revision.wrapping_add(1);
        *self.rev_cache.active_tool_json.borrow_mut() = None;
        self.rev_cache.last_snapshot_rev = None;
    }

    /// Computes a deterministic O(1) revision token for this framework's persisted state.
    pub fn state_revision(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.rev_cache.framework_revision.hash(&mut hasher);
        self.selected_tool_index.hash(&mut hasher);
        self.show_theory_portal.hash(&mut hasher);
        self.show_starter_gallery.hash(&mut hasher);
        self.saved_tool_states.len().hash(&mut hasher);
        if let Some(tool) = &self.active_tool {
            tool.name().hash(&mut hasher);
            let tool_rev = tool.state_revision();
            if tool_rev > 0 {
                1_u8.hash(&mut hasher);
                tool_rev.hash(&mut hasher);
            } else {
                0_u8.hash(&mut hasher);
                self.rev_cache.untracked_tool_revision.hash(&mut hasher);
            }
        }
        hasher.finish().max(1)
    }

    pub(super) fn serialized_active_tool(&self) -> Option<(&'static str, Arc<str>)> {
        let active_tool = self.active_tool.as_ref()?;
        let name = active_tool.name();
        let rev = active_tool.state_revision();
        if rev > 0 {
            if let Some((cached_name, cached_rev, cached_json)) =
                self.rev_cache.active_tool_json.borrow().as_ref()
            {
                if *cached_name == name && *cached_rev == rev {
                    return Some((name, Arc::clone(cached_json)));
                }
            }
        }
        let state_str = active_tool.save_state()?;
        let arc_json: Arc<str> = Arc::from(state_str);
        if rev > 0 {
            *self.rev_cache.active_tool_json.borrow_mut() =
                Some((name, rev, Arc::clone(&arc_json)));
        }
        Some((name, arc_json))
    }

    /// Serializes the framework state to JSON, embedding `saved_tool_states` as `RawValue` entries.
    pub fn save_state(&self) -> Option<String> {
        let active_entry = self.serialized_active_tool();
        let selected_tool_name = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx).map(|meta| meta.id));

        let state = FrameworkStateSerialize {
            selected_tool_index: self.selected_tool_index,
            selected_tool_name,
            show_theory_portal: self.show_theory_portal,
            show_starter_gallery: self.show_starter_gallery,
            tool_states: RawToolStates {
                saved: &self.saved_tool_states,
                active: active_entry
                    .as_ref()
                    .map(|(name, json)| (*name, json.as_ref())),
            },
        };
        serde_json::to_string(&state).ok()
    }

    /// Deserializes and restores the framework state from JSON.
    pub fn load_state(&mut self, state_str: &str) {
        let Ok(state) = serde_json::from_str::<FrameworkStateDeserialize>(state_str) else {
            return;
        };
        let _ = state.show_starter_gallery;
        self.show_theory_portal = state.show_theory_portal;
        self.show_starter_gallery = false;
        self.saved_tool_states = state.tool_states;
        self.bump_framework_revision();

        let tool_idx = state
            .selected_tool_name
            .as_deref()
            .and_then(|name| {
                let trimmed = name.trim();
                self.available_tools.iter().position(|m| {
                    m.id.eq_ignore_ascii_case(trimmed)
                        || m.display_name.eq_ignore_ascii_case(trimmed)
                })
            })
            .or(state.selected_tool_index);

        let Some(idx) = tool_idx.filter(|&i| i < self.available_tools.len()) else {
            if tool_idx.is_none() {
                self.selected_tool_index = None;
                self.active_tool = None;
            }
            return;
        };

        if self.selected_tool_index == Some(idx) && self.active_tool.is_some() {
            self.reload_existing_active_tool(idx);
        } else {
            self.select_tool(idx);
        }
    }

    fn reload_existing_active_tool(&mut self, idx: usize) {
        let meta = self.available_tools[idx];
        self.is_diverged = false;
        self.diagnostic_report = None;
        if let Some(tool) = &mut self.active_tool {
            if let Some(saved) = self
                .saved_tool_states
                .get(tool.name())
                .or_else(|| self.saved_tool_states.get(meta.id))
            {
                tool.load_state(saved);
            }
            let refreshed = capture_isolated_tool_datasets(tool.as_ref());
            let recovery = tool.save_recovery_state();
            let rev = tool.state_revision();
            self.sync_state.datasets_owner_tool = Some(tool.name());
            self.current_valid_plot_datasets = refreshed.clone();
            self.rev_cache.last_snapshot_rev = (rev > 0).then_some((tool.name(), rev));
            self.last_valid_snapshot = Some(StateSnapshot {
                tool_state: recovery,
                plot_datasets: refreshed,
                step_index: self.step_counter,
                timestamp: crate::web_io::PlatformInstant::now(),
            });
        }
    }

    /// Selects and activates a tool by index in `available_tools`.
    pub fn select_tool(&mut self, index: usize) {
        if index >= self.available_tools.len()
            || (self.selected_tool_index == Some(index) && self.active_tool.is_some())
        {
            return;
        }
        if let Some((old_name, old_json)) = self.serialized_active_tool() {
            self.saved_tool_states
                .insert(old_name.to_string(), old_json.to_string());
        }
        self.bump_framework_revision();
        self.selected_tool_index = Some(index);
        let meta = self.available_tools[index];
        let mut new_tool = (meta.build)();
        let factory_baseline_state = new_tool.save_state();
        if let Some(state) = self
            .saved_tool_states
            .get(new_tool.name())
            .or_else(|| self.saved_tool_states.get(meta.id))
        {
            new_tool.load_state(state);
        }
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.export_status_message = None;
        self.export_queue.status = None;
        self.step_counter = 0;
        self.baseline_tool_state = factory_baseline_state;
        let recovery_state = new_tool.save_recovery_state();
        let rev = new_tool.state_revision();
        let initial_datasets = capture_isolated_tool_datasets(new_tool.as_ref());
        self.sync_state.datasets_owner_tool = Some(new_tool.name());
        self.current_valid_plot_datasets = initial_datasets.clone();
        self.rev_cache.last_snapshot_rev = (rev > 0).then_some((new_tool.name(), rev));
        self.last_valid_snapshot = Some(StateSnapshot {
            tool_state: recovery_state,
            plot_datasets: initial_datasets,
            step_index: 0,
            timestamp: crate::web_io::PlatformInstant::now(),
        });
        scientific_metadata::citation_registry::CitationRegistry::register(
            new_tool.name().to_string(),
            new_tool.theory().theory_citation(),
        );
        self.active_tool = Some(new_tool);
    }

    /// Selects and activates a tool by ID, display name, or tag in `available_tools` (case-insensitively).
    pub fn select_tool_by_name(&mut self, name: &str) -> bool {
        let name_trimmed = name.trim();
        if let Some(index) = self.available_tools.iter().position(|t| {
            t.id.eq_ignore_ascii_case(name_trimmed)
                || t.display_name.eq_ignore_ascii_case(name_trimmed)
                || t.tags
                    .iter()
                    .any(|tag| tag.eq_ignore_ascii_case(name_trimmed))
        }) {
            self.select_tool(index);
            true
        } else {
            false
        }
    }
}
