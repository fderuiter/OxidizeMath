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

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "neuroscience");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:graph_parameters_rust]
