// @explorer_feature = "physics"
// @explorer_order = 11
use crate::async_sim::SimWorkerState;
use crate::tabs::ExplorerTab;
use eframe::egui;
use serde::{Deserialize, Serialize};

pub mod kspace_bloch_view;
pub mod kspace_phantom;
pub mod kspace_runner;
pub mod kspace_tool;
pub mod kspace_types;
pub mod kspace_ui;
pub mod kspace_worker;
pub mod runner;
pub mod theory;
pub mod tool;
pub mod ui;

#[cfg(test)]
pub use kspace_tool::MriKSpaceTool;
pub use runner::sanitize_mri_time_scale;
#[cfg(test)]
pub use runner::{MriRunner, MriSharedParams, MriSnapshot, MAX_MRI_TIME_SCALE, MIN_MRI_TIME_SCALE};
#[cfg(test)]
pub use tool::MriTool;
pub use tool::MriToolState;

/// Legacy flat `MriState` schema saved by pre-framework `.mws` workspaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyMriState {
    /// Optional worker execution state.
    #[serde(default)]
    pub state: Option<SimWorkerState>,
    /// Legacy boolean pause flag (`paused: false` => `Running`, `paused: true` => `Ready`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Previously persisted simulation speed multiplier.
    pub time_scale: f64,
}

/// Physics tab hosting [`tool::MriTool`] inside [`crate::framework::SimulationFramework`].
pub struct MriTab {
    /// Underlying simulation framework managing exports, presets, theory portal, and state guard.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for MriTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("mri"),
        }
    }
}

impl MriTab {
    fn try_load_legacy_state(&mut self, state: &str) -> bool {
        let Ok(legacy) = serde_json::from_str::<LegacyMriState>(state) else {
            return false;
        };
        let _ = self.framework.select_tool_by_name("MriTool");
        let Some(tool) = self.framework.active_tool.as_mut() else {
            return true;
        };
        let Some(current_json) = tool.save_state() else {
            return true;
        };
        let Ok(mut current_state) = serde_json::from_str::<MriToolState>(&current_json) else {
            return true;
        };
        current_state.time_scale = sanitize_mri_time_scale(legacy.time_scale);
        current_state.state = Some(legacy.state.unwrap_or_else(|| {
            if legacy.paused.unwrap_or(false) {
                SimWorkerState::Ready
            } else {
                SimWorkerState::Running
            }
        }));
        current_state.paused = None;
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

impl ExplorerTab for MriTab {
    fn name(&self) -> &'static str {
        "MRI Physics"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "mri");
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
mod kspace_tests;
#[cfg(test)]
mod tests;

// [cite:graph_parameters_rust]
