// @explorer_feature = "applied"
use crate::tabs::ExplorerTab;
use eframe::egui;

/// Validated state schema, bounds, and federated presets for age-structured cannibalism.
pub mod state;
/// Theoretical equations and citations for McKendrick-von Foerster and van den Bosch models.
pub mod theory;
/// In-thread simulation runner, Lexis heatmap builder, and [`FiniteStateGuard`](crate::framework::FiniteStateGuard).
pub mod tool;
/// Side-panel controls and scientific plots (Lexis heatmap, age pyramid, Hopf cycles, 2D ODE).
pub mod views;

pub use tool::PopulationStructureTool;

/// Applied mathematics tab hosting [`PopulationStructureTool`] inside [`crate::framework::SimulationFramework`].
pub struct PopulationStructureTab {
    /// Underlying simulation framework managing exports, presets, theory portal, and state guard.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for PopulationStructureTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("cannibalism"),
        }
    }
}

impl ExplorerTab for PopulationStructureTab {
    fn name(&self) -> &'static str {
        "Population Structure"
    }

    fn academic_domain(&self) -> &'static str {
        "Applied Math"
    }

    fn url_slug(&self) -> &'static str {
        "cannibalism"
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
