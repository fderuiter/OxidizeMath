// @explorer_feature = "biology"
use crate::async_sim::SimWorkerState;
use crate::tabs::ExplorerTab;
use eframe::egui;
use serde::{Deserialize, Serialize};

pub mod atlas;
pub mod lenia;
pub mod presets;
pub mod runner;
mod seeding;
pub mod tool;
pub mod ui;

#[cfg(test)]
pub use atlas::{
    classify_schnakenberg_regime, grid_u_variance, MorphogenesisCartographer, MorphogenesisRegime,
};
#[cfg(test)]
pub use lenia::LeniaTool;
#[cfg(test)]
pub use presets::MorphogenesisTheory;
pub use presets::SimpleRng;
#[cfg(test)]
pub use runner::{MorphogenesisRunner, MorphogenesisSharedParams};
pub use tool::{MorphogenesisTool, MorphogenesisToolState};

/// Legacy flat `MorphoState` schema saved by pre-framework `.mws` workspaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyMorphoState {
    /// Time step size `dt`.
    pub dt: f64,
    /// Optional worker execution state.
    #[serde(default)]
    pub state: Option<SimWorkerState>,
    /// Legacy boolean pause flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Legacy starter gallery visibility flag (ignored on load; managed by framework).
    #[serde(default)]
    pub show_starter_gallery: bool,
    /// Deferred start flag keeping hidden tabs idle until first rendered frame.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pending_initial_start: bool,
}

/// Biology tab hosting [`MorphogenesisTool`] and [`lenia::LeniaTool`] inside [`crate::framework::SimulationFramework`].
pub struct MorphogenesisTab {
    /// Underlying simulation framework managing exports, presets, theory portal, and state guard.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for MorphogenesisTab {
    fn default() -> Self {
        let mut framework = crate::framework::SimulationFramework::new("biology");
        let _ = framework.select_tool_by_name("MorphogenesisTool");
        Self { framework }
    }
}

impl MorphogenesisTab {
    fn try_load_legacy_state(&mut self, state: &str) -> bool {
        let Ok(legacy) = serde_json::from_str::<LegacyMorphoState>(state) else {
            return false;
        };
        if self.framework.active_tool_id() != Some("MorphogenesisTool") {
            let _ = self.framework.select_tool_by_name("MorphogenesisTool");
        }
        let Some(tool) = self.framework.active_tool.as_mut() else {
            return true;
        };
        let Some(current_json) = tool.save_state() else {
            return true;
        };
        let Ok(mut current_state) = serde_json::from_str::<MorphogenesisToolState>(&current_json)
        else {
            return true;
        };
        current_state.dt = legacy.dt.clamp(1e-4, 0.1);
        current_state.state = legacy.state;
        current_state.paused = legacy.paused;
        current_state.pending_initial_start = legacy.pending_initial_start;
        if let Ok(updated_json) = serde_json::to_string(&current_state) {
            if tool.try_load_state(&updated_json).is_ok() {
                self.framework.show_starter_gallery = false;
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

impl ExplorerTab for MorphogenesisTab {
    fn name(&self) -> &'static str {
        "Morphogenesis"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "biology");
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

// [cite:favorite_child]
