#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

pub mod controls;
pub mod plots;
pub mod presets;
pub mod state;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

/// Top-level `Pure Math` tab registered via [`TabDescriptor`].
pub struct AlgorithmicInfoTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for AlgorithmicInfoTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("algorithmic_information"),
        }
    }
}

impl ExplorerTab for AlgorithmicInfoTab {
    fn name(&self) -> &'static str {
        "Algorithmic Information"
    }

    fn academic_domain(&self) -> &'static str {
        "Pure Math"
    }

    fn url_slug(&self) -> &'static str {
        "algorithmic_information"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "algorithmic_information");
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

inventory::submit! {
    TabDescriptor {
        id: "algorithmic_information",
        order: 100,
        build: || Box::new(AlgorithmicInfoTab::default()),
    }
}

// [cite:algorithmic_information_rust]
