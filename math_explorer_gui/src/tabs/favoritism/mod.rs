// @explorer_feature = "applied"
// @explorer_order = 32
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod state;
pub mod theory;
pub mod tool;

pub use state::{FavoritismToolState, LegacyFavoritismState};
pub use tool::FavoritismTool;

/// Applied mathematics tab hosting [`FavoritismTool`] inside [`crate::framework::SimulationFramework`].
pub struct FavoritismTab {
    /// Underlying simulation framework managing exports, presets, theory portal, and state guard.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for FavoritismTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("favoritism"),
        }
    }
}

impl FavoritismTab {
    fn try_load_legacy_state(&mut self, state: &str) -> bool {
        let Ok(legacy) = serde_json::from_str::<LegacyFavoritismState>(state) else {
            return false;
        };
        let Some(tool) = self.framework.active_tool.as_mut() else {
            return true;
        };
        let Some(current_json) = tool.save_state() else {
            return true;
        };
        let Ok(mut current_state) = serde_json::from_str::<FavoritismToolState>(&current_json)
        else {
            return true;
        };
        current_state.selected_child_index = state::clamp_selected_child_index(
            legacy.selected_child_index,
            current_state.children.len(),
        );
        current_state.new_child_name = legacy.new_child_name;
        if let Ok(updated_json) = serde_json::to_string(&current_state) {
            if tool.try_load_state(&updated_json).is_ok() {
                if let Some(saved) = tool.save_state() {
                    self.framework
                        .saved_tool_states
                        .insert(tool.name().to_string(), saved);
                }
                self.framework.cache_valid_snapshot();
            }
        }
        true
    }
}

impl ExplorerTab for FavoritismTab {
    fn name(&self) -> &'static str {
        "Favoritism"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "favoritism");
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        if !self.try_load_legacy_state(state) {
            self.framework.load_state(state);
        }
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        self.framework.select_tool_by_name(name)
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

#[cfg(test)]
mod tests;

// [cite:graph_parameters_rust]
