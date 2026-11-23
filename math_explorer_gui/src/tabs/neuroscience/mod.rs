// @explorer_feature = "biology"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod hodgkin_huxley;
pub mod neural_network_viz;
pub mod spike_analysis;

pub struct NeuroscienceTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for NeuroscienceTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("neuroscience"),
        }
    }
}

impl ExplorerTab for NeuroscienceTab {
    fn name(&self) -> &'static str {
        "Neuroscience"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "neuroscience");
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

// [cite:graph_parameters_rust]
