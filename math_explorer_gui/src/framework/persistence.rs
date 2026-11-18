#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;

#[derive(serde::Serialize, serde::Deserialize)]
struct FrameworkState {
    selected_tool_index: Option<usize>,
    selected_tool_name: Option<String>,
    show_theory_portal: bool,
    #[serde(default)]
    show_starter_gallery: bool,
    tool_states: std::collections::HashMap<String, String>,
}

impl SimulationFramework {
    /// Serializes the framework state (selected tool index, active tool state, theory portal state) to JSON.
    pub fn save_state(&self) -> Option<String> {
        let mut tool_states = self.saved_tool_states.clone();
        if let Some(active_tool) = &self.active_tool {
            if let Some(state) = active_tool.save_state() {
                tool_states.insert(active_tool.name().to_string(), state);
            }
        }
        let selected_tool_name = self.selected_tool_index.and_then(|idx| {
            self.available_tools
                .get(idx)
                .map(|meta| meta.name.to_string())
        });

        let state = FrameworkState {
            selected_tool_index: self.selected_tool_index,
            selected_tool_name,
            show_theory_portal: self.show_theory_portal,
            show_starter_gallery: self.show_starter_gallery,
            tool_states,
        };
        serde_json::to_string(&state).ok()
    }

    /// Deserializes and restores the framework state from JSON.
    pub fn load_state(&mut self, state_str: &str) {
        let state: FrameworkState = match serde_json::from_str(state_str) {
            Ok(s) => s,
            Err(_) => return,
        };
        self.show_theory_portal = state.show_theory_portal;
        // Saved workspace session bypasses the empty-state starter gallery
        self.show_starter_gallery = false;
        self.saved_tool_states = state.tool_states;

        let tool_idx = if let Some(ref name) = state.selected_tool_name {
            self.available_tools
                .iter()
                .position(|m| m.name == *name)
                .or(state.selected_tool_index)
        } else {
            state.selected_tool_index
        };

        if let Some(idx) = tool_idx {
            if idx < self.available_tools.len() {
                if self.selected_tool_index == Some(idx) && self.active_tool.is_some() {
                    if let Some(tool) = &mut self.active_tool {
                        if let Some(saved) = self.saved_tool_states.get(tool.name()) {
                            tool.load_state(saved);
                        }
                        let prev = egui_plot::get_last_plot_datasets();
                        egui_plot::clear_last_plot_datasets();
                        let refreshed = tool.export_datasets();
                        if !prev.is_empty() {
                            egui_plot::append_last_plot_datasets(&prev);
                        }
                        (self.overlay_sync.2, self.current_valid_plot_datasets) =
                            (Some(tool.name()), refreshed.clone());
                        self.last_valid_snapshot = Some(StateSnapshot {
                            tool_state: tool.save_state(),
                            plot_datasets: refreshed,
                            step_index: self.step_counter,
                            timestamp: crate::web_io::PlatformInstant::now(),
                        });
                    }
                } else {
                    self.select_tool(idx);
                }
            }
        } else {
            self.selected_tool_index = None;
            self.active_tool = None;
        }
    }
}
