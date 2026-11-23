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

impl RawToolStates<'_> {
    fn is_empty(&self) -> bool {
        self.saved.is_empty() && self.active.is_none()
    }
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

fn decode_bounded_tool_entry(raw: Box<RawValue>) -> Option<String> {
    let raw_text = raw.get();
    let max_quoted_len = MAX_TOOL_STATE_BYTES.saturating_mul(2).saturating_add(2);
    if raw_text.starts_with('"') {
        if raw_text.len() > max_quoted_len {
            return None;
        }
        if let Ok(unescaped) = serde_json::from_str::<String>(raw_text) {
            return ToolStateBounds::DEFAULT
                .validate_json(&unescaped)
                .is_ok()
                .then_some(unescaped);
        }
    }
    if ToolStateBounds::DEFAULT.validate_json(raw_text).is_err() {
        return None;
    }
    let boxed_str: Box<str> = raw.into();
    Some(String::from(boxed_str))
}

fn deserialize_tool_states<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<HashMap<String, String>, D::Error> {
    struct ToolStatesVisitor;

    impl<'de> serde::de::Visitor<'de> for ToolStatesVisitor {
        type Value = HashMap<String, String>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a bounded map of tool names to serialized JSON states")
        }

        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            use serde::de::Error;
            let cap = map.size_hint().unwrap_or(0).min(MAX_SESSION_TOOL_ENTRIES);
            let mut out = HashMap::with_capacity(cap);
            let mut entry_count = 0_usize;
            let mut total_bytes = 0_usize;

            while let Some((key, raw)) = map.next_entry::<String, Box<RawValue>>()? {
                entry_count = entry_count.saturating_add(1);
                if entry_count > MAX_SESSION_TOOL_ENTRIES {
                    return Err(M::Error::custom("tool_states entry count limit exceeded"));
                }
                if key.len() > MAX_TOOL_KEY_BYTES {
                    return Err(M::Error::custom("tool_states key length limit exceeded"));
                }
                total_bytes = total_bytes
                    .saturating_add(key.len())
                    .saturating_add(raw.get().len());
                if total_bytes > MAX_SESSION_STATE_BYTES {
                    return Err(M::Error::custom("tool_states session byte limit exceeded"));
                }
                if let Some(decoded) = decode_bounded_tool_entry(raw) {
                    out.insert(key, decoded);
                }
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(ToolStatesVisitor)
}

#[derive(Serialize)]
struct FrameworkStateSerialize<'a> {
    selected_tool_index: Option<usize>,
    selected_tool_name: Option<&'a str>,
    show_theory_portal: bool,
    show_starter_gallery: bool,
    tool_states: RawToolStates<'a>,
    #[serde(skip_serializing_if = "RawToolStates::is_empty")]
    preset_tool_states: RawToolStates<'a>,
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
    #[serde(default, deserialize_with = "deserialize_tool_states")]
    preset_tool_states: HashMap<String, String>,
}

pub(super) fn capture_isolated_tool_datasets(
    tool: &dyn InteractiveTool,
) -> Vec<(String, Vec<[f64; 2]>)> {
    let prev = egui_plot::get_last_plot_datasets();
    let prev_3d = egui_plot::get_last_plot_datasets_3d();
    let prev_prims = egui_plot::get_last_plot_primitives();
    egui_plot::clear_last_plot_datasets();
    let datasets = tool.export_datasets();
    egui_plot::clear_last_plot_datasets();
    if !prev.is_empty() {
        egui_plot::append_last_plot_datasets(&prev);
    }
    if !prev_3d.is_empty() {
        egui_plot::append_last_plot_datasets_3d(&prev_3d);
    }
    if !prev_prims.is_empty() {
        egui_plot::append_last_plot_primitives(&prev_prims);
    }
    datasets
}

fn capture_isolated_tool_datasets_3d(tool: &dyn InteractiveTool) -> Vec<(String, Vec<[f64; 3]>)> {
    let prev = egui_plot::get_last_plot_datasets();
    let prev_3d = egui_plot::get_last_plot_datasets_3d();
    let prev_prims = egui_plot::get_last_plot_primitives();
    egui_plot::clear_last_plot_datasets();
    let datasets_3d = tool.export_datasets_3d();
    egui_plot::clear_last_plot_datasets();
    if !prev.is_empty() {
        egui_plot::append_last_plot_datasets(&prev);
    }
    if !prev_3d.is_empty() {
        egui_plot::append_last_plot_datasets_3d(&prev_3d);
    }
    if !prev_prims.is_empty() {
        egui_plot::append_last_plot_primitives(&prev_prims);
    }
    datasets_3d
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
        self.preset_tool_states.len().hash(&mut hasher);
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
            preset_tool_states: RawToolStates {
                saved: &self.preset_tool_states,
                active: None,
            },
        };
        serde_json::to_string(&state).ok()
    }

    /// Deserializes and restores the framework state from JSON.
    pub fn load_state(&mut self, state_str: &str) {
        let cleaned = egui_plot::export::strip_utf8_bom(state_str);
        if ToolStateBounds::SESSION.validate_json(cleaned).is_err() {
            return;
        }
        let Ok(state) = serde_json::from_str::<FrameworkStateDeserialize>(cleaned) else {
            return;
        };
        let _ = state.show_starter_gallery;
        self.show_theory_portal = state.show_theory_portal;
        self.show_starter_gallery = false;
        self.saved_tool_states = state.tool_states;
        self.preset_tool_states = state.preset_tool_states;
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
        if self.factory_tool_state.is_none() {
            self.factory_tool_state = (meta.build)().save_state();
        }
        if let Some(tool) = &mut self.active_tool {
            let preset_state = self
                .preset_tool_states
                .get(tool.name())
                .or_else(|| self.preset_tool_states.get(meta.id))
                .cloned();
            self.preset_tool_state = preset_state.clone();
            self.baseline_tool_state = preset_state.or_else(|| self.factory_tool_state.clone());
            if let Some(saved) = self
                .saved_tool_states
                .get(tool.name())
                .or_else(|| self.saved_tool_states.get(meta.id))
            {
                tool.load_state(saved);
            }
            if !tool.has_finite_parameters() {
                if let Some(ref base) = self.baseline_tool_state {
                    let _ = tool.try_load_recovery_state(base);
                }
                if !tool.has_finite_parameters() {
                    let fresh = (meta.build)();
                    let factory = fresh.save_state();
                    self.factory_tool_state = factory.clone();
                    self.baseline_tool_state = self.preset_tool_state.clone().or(factory);
                    *tool = fresh;
                }
            }
            let refreshed = capture_isolated_tool_datasets(tool.as_ref());
            let refreshed_3d = capture_isolated_tool_datasets_3d(tool.as_ref());
            let recovery = tool.save_recovery_state();
            let rev = tool.state_revision();
            self.sync_state.datasets_owner_tool = Some(tool.name());
            self.current_valid_plot_datasets = refreshed.clone();
            self.current_valid_plot_datasets_3d = refreshed_3d;
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
        let preset_baseline_state = self
            .preset_tool_states
            .get(new_tool.name())
            .or_else(|| self.preset_tool_states.get(meta.id))
            .cloned();
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
        self.factory_tool_state = factory_baseline_state.clone();
        self.preset_tool_state = preset_baseline_state.clone();
        self.baseline_tool_state = preset_baseline_state.or(factory_baseline_state);
        let recovery_state = new_tool.save_recovery_state();
        let rev = new_tool.state_revision();
        let initial_datasets = capture_isolated_tool_datasets(new_tool.as_ref());
        let initial_datasets_3d = capture_isolated_tool_datasets_3d(new_tool.as_ref());
        self.sync_state.datasets_owner_tool = Some(new_tool.name());
        self.current_valid_plot_datasets = initial_datasets.clone();
        self.current_valid_plot_datasets_3d = initial_datasets_3d;
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

    /// Returns the active tool's interactive canvas `Id` if applicable.
    pub fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.active_tool
            .as_ref()
            .and_then(|t| t.focused_canvas_id())
    }
}
