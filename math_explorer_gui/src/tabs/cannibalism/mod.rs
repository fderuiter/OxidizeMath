// @explorer_feature = "applied"
use crate::tabs::ExplorerTab;
use eframe::egui;

/// Preallocated Lexis carpet texture and plot rendering helpers for [`CannibalismTool`].
pub mod plots;
/// Serializable state schema, bounds validation, and starter presets for [`CannibalismTool`].
pub mod state;
/// Academic citations and equations for [`CannibalismTool`].
pub mod theory;
/// Interactive McKendrick-von Foerster PDE and *Tribolium* bifurcation lab.
pub mod tool;

pub use tool::CannibalismTool;

/// Applied mathematics tab hosting [`CannibalismTool`] inside [`crate::framework::SimulationFramework`].
pub struct CannibalismTab {
    /// Underlying simulation framework managing exports, presets, theory portal, and state guard.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for CannibalismTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("cannibalism"),
        }
    }
}

impl ExplorerTab for CannibalismTab {
    fn name(&self) -> &'static str {
        "Cannibalism"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "cannibalism");
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
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

// [cite:cannibalism]
