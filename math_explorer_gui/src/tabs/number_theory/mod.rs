// @explorer_feature = "pure_math"
// @explorer_order = 21
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod elliptic_curve_canvas;
pub mod elliptic_curve_hasse;
pub mod elliptic_curve_playground;
#[cfg(test)]
mod elliptic_curve_tests;
pub mod factorization;
pub mod partitions_widget;
pub mod prime_spiral;
pub mod ualbf_widget;

pub struct NumberTheoryTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for NumberTheoryTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("number_theory"),
        }
    }
}

impl ExplorerTab for NumberTheoryTab {
    fn name(&self) -> &'static str {
        "Number Theory"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "number_theory");
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

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework.focused_canvas_id()
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

// [cite:graph_parameters_rust]
