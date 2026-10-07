// @explorer_feature = "climate"
pub mod cera;
pub mod co2_projections;
pub mod temperature;

use crate::tabs::ExplorerTab;
use eframe::egui;

pub struct ClimateTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for ClimateTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("climate"),
        }
    }
}

impl ExplorerTab for ClimateTab {
    fn name(&self) -> &'static str {
        "Climate"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "climate");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:graph_parameters_rust]
