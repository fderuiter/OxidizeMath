// @explorer_feature = "pure_math"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod bayesian_analytical;
pub mod bayesian_tool;
pub mod bayesian_ui;
pub mod copula_tool;
pub mod copula_ui;
pub mod distribution_eval;
pub mod distribution_tool;
pub mod distribution_ui;
pub mod presets;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

/// Top-level `Pure Math` tab auto-discovered by `math_explorer_gui/build.rs`.
pub struct ProbabilityStatisticsTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for ProbabilityStatisticsTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("probability_statistics"),
        }
    }
}

impl ExplorerTab for ProbabilityStatisticsTab {
    fn name(&self) -> &'static str {
        "Probability & Statistics"
    }

    fn academic_domain(&self) -> &'static str {
        "Pure Math"
    }

    fn url_slug(&self) -> &'static str {
        "probability_statistics"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "probability_statistics");
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

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework.focused_canvas_id()
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

// [cite:clinical_trials_statistics]
