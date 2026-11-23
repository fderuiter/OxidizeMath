// @explorer_feature = "pure_math"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod controls;
pub mod plots;
pub mod presets;
pub mod state;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

/// Top-level `Pure Math` tab auto-discovered by `math_explorer_gui/build.rs`.
pub struct LinearAlgebraTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for LinearAlgebraTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("linear_algebra"),
        }
    }
}

impl ExplorerTab for LinearAlgebraTab {
    fn name(&self) -> &'static str {
        "Linear Algebra"
    }

    fn academic_domain(&self) -> &'static str {
        "Pure Math"
    }

    fn url_slug(&self) -> &'static str {
        "linear_algebra"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "linear_algebra");
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

// [cite:advanced_linear_algebra]
